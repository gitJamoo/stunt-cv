"""Pose data types, COCO keypoint constants, and pure geometry/kinematics
helpers shared by tracking, stats, exports, and analysis. Nothing in this
module touches Tkinter or YOLO — it is fully headless and unit-testable.

Coordinates convention: landmarks are normalized 0-1 relative to the full
frame; helpers that need pixels take explicit (w, h).
"""
from collections import deque

import numpy as np

# COCO 17-keypoint skeleton connections used for drawing
YOLO_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),    # face
    (5, 6),                              # shoulders
    (5, 7), (7, 9),                      # left arm
    (6, 8), (8, 10),                     # right arm
    (5, 11), (6, 12),                    # torso sides
    (11, 12),                            # hips
    (11, 13), (13, 15),                  # left leg
    (12, 14), (14, 16),                  # right leg
]

# COCO keypoint indices used throughout stats calculations
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW,    KP_R_ELBOW    = 7, 8
KP_L_WRIST,    KP_R_WRIST    = 9, 10
KP_L_HIP,      KP_R_HIP      = 11, 12
KP_L_KNEE,     KP_R_KNEE     = 13, 14
KP_L_ANKLE,    KP_R_ANKLE    = 15, 16

COM_KEYPOINTS = (KP_L_SHOULDER, KP_R_SHOULDER, KP_L_HIP, KP_R_HIP,   # torso
                 KP_L_KNEE, KP_R_KNEE, KP_L_ANKLE, KP_R_ANKLE,       # legs
                 KP_L_ELBOW, KP_R_ELBOW, KP_L_WRIST, KP_R_WRIST)     # arms


class Landmark:
    __slots__ = ('x', 'y', 'visibility')
    def __init__(self, x=0.0, y=0.0, visibility=0.0):
        self.x = x
        self.y = y
        self.visibility = visibility


class LandmarkList:
    def __init__(self, landmarks):
        self.landmark = landmarks


class SmoothedResults:
    """Result wrapper kept for MediaPipe-era call-shape compatibility:
    callers read `.pose_landmarks.landmark[i].x/.y/.visibility`."""
    def __init__(self, landmarks):
        self.pose_landmarks = landmarks


class PoseSmoother:
    """Boxcar-averages landmarks over a rolling window, and holds the last
    smoothed pose through short detection gaps (up to max_gap frames) so a
    one-frame dropout doesn't wipe the history and snap on re-acquire."""
    def __init__(self, window, max_gap=5):
        self.frames = deque(maxlen=window)
        self.max_gap = max_gap
        self.miss = 0

    def set_window(self, window):
        if window != self.frames.maxlen:
            self.frames = deque(self.frames, maxlen=window)

    def clear(self):
        self.frames.clear()
        self.miss = 0

    def smooth(self, results):
        if not results or not results.pose_landmarks:
            self.miss += 1
            if self.miss > self.max_gap:
                self.frames.clear()
            return self._average()
        self.miss = 0
        self.frames.append(results.pose_landmarks.landmark)
        return self._average()

    def _average(self):
        if not self.frames:
            return None
        n = len(self.frames[0])
        smoothed = [
            Landmark(
                x=sum(f[i].x for f in self.frames) / len(self.frames),
                y=sum(f[i].y for f in self.frames) / len(self.frames),
                visibility=sum(f[i].visibility for f in self.frames) / len(self.frames),
            )
            for i in range(n)
        ]
        return SmoothedResults(LandmarkList(smoothed))


def yolo_to_landmarks(kp_xyn, kp_conf):
    """Convert YOLO normalized keypoints + confidences to a LandmarkList."""
    return LandmarkList([
        Landmark(x=float(kp_xyn[i][0]), y=float(kp_xyn[i][1]), visibility=float(kp_conf[i]))
        for i in range(len(kp_xyn))
    ])


def translate_landmarks(lm_list, crop_x, crop_y, crop_w, crop_h, frame_w, frame_h):
    """Remap crop-relative normalized landmarks to full-frame normalized coords (in place)."""
    for lm in lm_list.landmark:
        lm.x = (lm.x * crop_w + crop_x) / frame_w
        lm.y = (lm.y * crop_h + crop_y) / frame_h


def pose_bottom_y(pose):
    """Lowest visible point of a person (max y), preferring ankles/hips.
    Robust to upper-body-only detections, where a mean over visible
    keypoints makes an occluded base look like the highest person."""
    lm = pose.landmark
    lower = [lm[i].y for i in (KP_L_ANKLE, KP_R_ANKLE, KP_L_HIP, KP_R_HIP)
             if i < len(lm) and lm[i].visibility > 0.3]
    if lower:
        return max(lower)
    visible = [p.y for p in lm if p.visibility > 0.3]
    return max(visible) if visible else 0.5


def pose_avg_x(pose):
    visible = [lm for lm in pose.landmark if lm.visibility > 0.3]
    if not visible:
        return 0.5
    return sum(lm.x for lm in visible) / len(visible)


def pose_centroid(pose):
    """(x, y) mean of visible landmarks in normalized coords, or None."""
    visible = [lm for lm in pose.landmark if lm.visibility > 0.3]
    if not visible:
        return None
    return (sum(lm.x for lm in visible) / len(visible),
            sum(lm.y for lm in visible) / len(visible))


def get_bounding_box(landmarks, w, h):
    # Filter out undetected keypoints (YOLO returns 0,0 with conf~0)
    pts = [(lm.x * w, lm.y * h) for lm in landmarks if lm.visibility > 0.3]
    if not pts:
        pts = [(lm.x * w, lm.y * h) for lm in landmarks]
    x_coords, y_coords = zip(*pts)
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)


def calculate_iou(box1, box2):
    x1_inter, y1_inter = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2_inter, y2_inter = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / (box1_area + box2_area + 1e-6)


def center_of_mass(landmarks, w, h):
    """Unweighted average of visible torso/leg/arm keypoints, in pixels."""
    if not landmarks:
        return None
    total, com_x, com_y = 0, 0, 0
    for idx in COM_KEYPOINTS:
        if idx < len(landmarks.landmark):
            lm = landmarks.landmark[idx]
            if lm.visibility > 0.5:
                com_x += lm.x * w
                com_y += lm.y * h
                total += 1
    if total == 0:
        return None
    return (com_x / total, com_y / total)


def torso_length(landmarks, w, h):
    """Pixel distance from avg shoulder to avg hip — used to normalize all stats."""
    if not landmarks:
        return None
    lm = landmarks.landmark
    sx = (lm[KP_L_SHOULDER].x + lm[KP_R_SHOULDER].x) / 2 * w
    sy = (lm[KP_L_SHOULDER].y + lm[KP_R_SHOULDER].y) / 2 * h
    hx = (lm[KP_L_HIP].x + lm[KP_R_HIP].x) / 2 * w
    hy = (lm[KP_L_HIP].y + lm[KP_R_HIP].y) / 2 * h
    length = np.hypot(sx - hx, sy - hy)
    return length if length > 0 else None


def base_alignment_tl(base_lm, frame_w, base_torso):
    """Base joint alignment: horizontal deviation of the shoulder/hip/ankle
    stack, in torso lengths. Returns None without a base or torso reference."""
    if not base_lm or not base_torso:
        return None
    lm = base_lm.landmark
    shoulder_x = (lm[KP_L_SHOULDER].x + lm[KP_R_SHOULDER].x) / 2
    hip_x      = (lm[KP_L_HIP].x      + lm[KP_R_HIP].x)      / 2
    ankle_x    = (lm[KP_L_ANKLE].x     + lm[KP_R_ANKLE].x)    / 2
    return (abs(shoulder_x - hip_x) + abs(hip_x - ankle_x)) * frame_w / base_torso
