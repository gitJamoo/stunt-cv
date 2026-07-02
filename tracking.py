"""Person detection and base/flyer role tracking, headless (no Tkinter).

PoseTracker wraps two YOLOv8-pose model handles:
- the tracking model, run through .track() so ByteTrack assigns persistent
  per-person IDs across frames;
- a lazily created plain-detection model for ROI mode, kept separate because
  .track() registers tracker callbacks on the model's predictor and per-crop
  detections must not feed the tracker.

Roles are sticky: base/flyer are bound to track IDs once (heuristic on the
first frame with 2+ people, or by the user) and follow those IDs — they are
never silently reassigned by per-frame geometry. Role state lives in a plain
dict (see new_state) owned by the caller, so live playback and export passes
can each maintain their own without interfering.
"""
from ultralytics import YOLO

from pose_data import (SmoothedResults, yolo_to_landmarks, translate_landmarks,
                       pose_bottom_y, pose_avg_x, get_bounding_box, calculate_iou)

REBIND_AFTER_MISSES = 5   # frames a role's track must be gone before overlap re-bind
REBIND_MIN_IOU = 0.25


class PoseTracker:
    def __init__(self, model_path='yolov8m-pose.pt'):
        # Weights download automatically on first run
        self.model_path = model_path
        self.yolo = YOLO(model_path)
        self._roi_model = None

        # Per-frame outputs of the last detect_auto call, for drawing and
        # click-to-assign in the UI
        self.last_poses = []   # all people detected last frame (LandmarkLists)
        self.last_ids = []     # parallel list of ByteTrack IDs
        self.raw_base = None   # unsmoothed LandmarkList currently bound to base
        self.raw_flyer = None  # unsmoothed LandmarkList currently bound to flyer

    @staticmethod
    def new_state():
        return {
            'base_id': None, 'flyer_id': None,   # ByteTrack IDs bound to each role
            'base_lm': None, 'flyer_lm': None,   # last known landmarks per role
            'base_miss': 0, 'flyer_miss': 0,     # consecutive frames the role's track was absent
            'initialized': False,                # roles picked (heuristic or user)
        }

    @staticmethod
    def invalidate_role_ids(state):
        """Track IDs restart after a tracker reset: drop the stale bindings and
        let the overlap re-bind recover each role from its last known box."""
        state['base_id'] = state['flyer_id'] = None
        state['base_miss'] = state['flyer_miss'] = REBIND_AFTER_MISSES

    def reset(self):
        """Clear ByteTrack state so a new video / export pass starts fresh."""
        predictor = getattr(self.yolo, 'predictor', None)
        for tracker in getattr(predictor, 'trackers', None) or []:
            if hasattr(tracker, 'reset'):
                tracker.reset()

    def detect_auto(self, frame, state, ui_conf, crop_rect=None):
        """Single-pass multi-person detection with ByteTrack persistent IDs.
        crop_rect: optional (x1, y1, x2, y2) detection area in frame pixels.
        Mutates `state`; returns (base_results, flyer_results) as
        SmoothedResults or None."""
        h, w = frame.shape[:2]

        detect_frame, dc_x, dc_y = frame, 0, 0
        if crop_rect:
            rx, ry, rx2, ry2 = crop_rect
            if rx2 > rx and ry2 > ry:
                detect_frame = frame[ry:ry2, rx:rx2]
                dc_x, dc_y = rx, ry

        # conf=0.1 feeds low-confidence boxes to ByteTrack, whose second
        # association stage uses them to hold tracks through occlusion —
        # critical when the flyer covers the base. iou=0.7 keeps NMS from
        # suppressing the stacked pair down to one detection. The caller's
        # confidence threshold is applied below, but never to the boxes
        # carrying the pair's track IDs.
        yolo_out = self.yolo.track(detect_frame, persist=True, verbose=False, conf=0.1, iou=0.7)[0]

        role_ids = {state['base_id'], state['flyer_id']}
        poses, ids = [], []
        if yolo_out.keypoints is not None and yolo_out.boxes.id is not None:
            for i in range(len(yolo_out.boxes)):
                tid = int(yolo_out.boxes.id[i])
                if float(yolo_out.boxes.conf[i]) < ui_conf and tid not in role_ids:
                    continue
                kp_xyn = yolo_out.keypoints.xyn[i].cpu().numpy()
                kp_conf = yolo_out.keypoints.conf[i].cpu().numpy()
                lm_list = yolo_to_landmarks(kp_xyn, kp_conf)
                if dc_x or dc_y:
                    dh, dw = detect_frame.shape[:2]
                    translate_landmarks(lm_list, dc_x, dc_y, dw, dh, w, h)
                poses.append(lm_list)
                ids.append(tid)
        self.last_poses = poses
        self.last_ids = ids
        by_id = dict(zip(ids, poses))

        # One-shot role initialization
        if not state['initialized']:
            if len(poses) >= 2:
                state['flyer_id'], state['base_id'] = self._best_stack_pair(ids, by_id)
                state['initialized'] = True
            elif len(poses) == 1:
                state['base_id'] = ids[0]  # solo person: treat as base until a pair shows up

        for role in ('base', 'flyer'):
            current = by_id.get(state[f'{role}_id'])
            if current is None and (state[f'{role}_id'] is not None or state[f'{role}_lm'] is not None):
                state[f'{role}_miss'] += 1
                if state[f'{role}_miss'] >= REBIND_AFTER_MISSES:
                    new_id = self._rebind_role(state, role, by_id, w, h)
                    if new_id is not None:
                        state[f'{role}_id'] = new_id
                        current = by_id[new_id]
            if current is not None:
                state[f'{role}_miss'] = 0
                state[f'{role}_lm'] = current

        self.raw_base = by_id.get(state['base_id'])
        self.raw_flyer = by_id.get(state['flyer_id'])

        base_results = SmoothedResults(self.raw_base) if self.raw_base else None
        flyer_results = SmoothedResults(self.raw_flyer) if self.raw_flyer else None
        return base_results, flyer_results

    def detect_in_roi(self, frame, rect, ui_conf):
        """Plain (untracked) detection of the single best person inside
        rect=(x1, y1, x2, y2) frame pixels. Returns SmoothedResults or None."""
        fh, fw = frame.shape[:2]
        rx, ry, rx2, ry2 = rect
        crop = frame[ry:ry2, rx:rx2]
        if crop.size == 0:
            return None
        if self._roi_model is None:
            self._roi_model = YOLO(self.model_path)
        yolo_out = self._roi_model(crop, verbose=False)[0]
        if yolo_out.keypoints is None or len(yolo_out.boxes) == 0:
            return None
        best_i = int(yolo_out.boxes.conf.argmax())
        if float(yolo_out.boxes.conf[best_i]) < ui_conf:
            return None
        kp_xyn = yolo_out.keypoints.xyn[best_i].cpu().numpy()
        kp_conf = yolo_out.keypoints.conf[best_i].cpu().numpy()
        lm_list = yolo_to_landmarks(kp_xyn, kp_conf)
        translate_landmarks(lm_list, rx, ry, rx2 - rx, ry2 - ry, fw, fh)
        return SmoothedResults(lm_list)

    def _best_stack_pair(self, ids, by_id):
        """Pick (flyer_id, base_id): flyer = person whose lowest point is highest
        in frame; base = person most horizontally aligned directly below them."""
        ordered = sorted(ids, key=lambda t: pose_bottom_y(by_id[t]))
        flyer_id = ordered[0]
        flyer_x = pose_avg_x(by_id[flyer_id])
        flyer_bottom = pose_bottom_y(by_id[flyer_id])
        below = [t for t in ordered[1:] if pose_bottom_y(by_id[t]) > flyer_bottom]
        if not below:
            below = ordered[1:]
        base_id = min(below, key=lambda t: abs(pose_avg_x(by_id[t]) - flyer_x))
        return flyer_id, base_id

    def _rebind_role(self, state, role, by_id, w, h):
        """A role's track died and ByteTrack couldn't recover it: re-bind the
        role to the unassigned track that best overlaps its last known box."""
        last_lm = state[f'{role}_lm']
        if last_lm is None:
            return None
        other_id = state['flyer_id'] if role == 'base' else state['base_id']
        ref_box = get_bounding_box(last_lm.landmark, w, h)
        best_id, best_iou = None, REBIND_MIN_IOU
        for tid, pose in by_id.items():
            if tid == other_id:
                continue
            iou = calculate_iou(ref_box, get_bounding_box(pose.landmark, w, h))
            if iou > best_iou:
                best_iou, best_id = iou, tid
        return best_id
