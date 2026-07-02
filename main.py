import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import csv
from collections import deque
from ultralytics import YOLO

import pandas as pd
import plotly.express as px
import os

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


class StuntCVApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stunt CV - YOLOv8 Tracking")

        # Core App State
        self.video_path = None
        self.cap = None
        self.playing = False
        self.video_width, self.video_height, self.total_frames = 0, 0, 0
        self.display_width, self.display_height = 480, 360
        self.current_frame_data = None
        self._resize_job = None
        self._last_detected_poses = []  # all people detected last frame, for click-to-assign
        self._last_detected_ids = []    # parallel list of ByteTrack IDs
        self._raw_base_pose  = None    # unsmoothed LandmarkList for the current base
        self._raw_flyer_pose = None    # unsmoothed LandmarkList for the current flyer

        # YOLOv8 Pose model — downloads yolov8m-pose.pt automatically on first run.
        # Auto mode runs it through .track() (ByteTrack persistent IDs). ROI mode
        # uses a separate plain-detection instance (created lazily), because
        # .track() registers tracker callbacks on the model's predictor and
        # per-crop detections must not feed the tracker.
        self.yolo = YOLO('yolov8m-pose.pt')
        self.yolo_roi = None

        # UI-Controlled Tracking Parameters
        self.smoothing_window_var = tk.IntVar(value=8)
        self.conf_threshold_var   = tk.DoubleVar(value=0.4)
        self.keypoint_vis_var     = tk.DoubleVar(value=0.4)

        # Crop State
        self.detection_crop_enabled = tk.BooleanVar(value=False)
        self.output_crop_enabled    = tk.BooleanVar(value=False)
        self.detection_crop = {"x": 0, "y": 0, "w": self.display_width, "h": self.display_height, "name": "crop"}

        # Smoothing
        self.base_smoother = PoseSmoother(self.smoothing_window_var.get())
        self.flyer_smoother = PoseSmoother(self.smoothing_window_var.get())

        # UI State Variables
        self.roi_tracking_enabled = tk.BooleanVar(value=False)
        self.track_base = tk.BooleanVar(value=True)
        self.track_flyer = tk.BooleanVar(value=True)
        self.show_stats = tk.BooleanVar(value=False)

        # ROI State
        self.base_roi =  {"x": 30,  "y": 110, "w": 150, "h": 180, "name": "base"}
        self.flyer_roi = {"x": 30, "y": 15,  "w": 150, "h": 180, "name": "flyer"}
        self.active_roi = None
        self.drag_info = {}
        self.handle_size = 8

        # Tracking State: base/flyer roles are bound to ByteTrack IDs and stay
        # sticky until a track dies or the user reassigns them
        self.track_state = self._new_track_state()
        self._inversion_count = 0
        self.role_hint_var = tk.StringVar(value="")

        # Stats State
        self.last_com_base = None
        self.last_com_flyer = None
        self.last_frame_time = None
        self.base_com_history = deque(maxlen=15)
        self.flyer_com_history = deque(maxlen=15)

        # Stats UI Variables
        self.base_velocity_var = tk.StringVar(value="Base Vel: N/A")
        self.flyer_velocity_var = tk.StringVar(value="Flyer Vel: N/A")
        self.base_wobble_var = tk.StringVar(value="Base Wobble: N/A")
        self.flyer_wobble_var = tk.StringVar(value="Flyer Wobble: N/A")
        self.alignment_var = tk.StringVar(value="Alignment: N/A")
        self.plumb_line_var = tk.StringVar(value="Plumb Line: N/A")
        self.stunt_score_var = tk.StringVar(value="Stunt Score: N/A")

        self.create_widgets()

    def create_widgets(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=5, pady=5)

        self.video_frame = tk.Frame(main_frame)
        self.video_frame.pack(side=tk.LEFT)

        self.canvas_left = tk.Canvas(self.video_frame, width=self.display_width, height=self.display_height, bg='black')
        self.canvas_left.pack(side=tk.LEFT, padx=5, pady=5)
        self.canvas_middle = tk.Canvas(self.video_frame, width=self.display_width, height=self.display_height, bg='black')
        self.canvas_middle.pack(side=tk.LEFT, padx=5, pady=5)
        self.canvas_right = tk.Canvas(self.video_frame, width=self.display_width, height=self.display_height, bg='black')
        self.canvas_right.pack(side=tk.LEFT, padx=5, pady=5)

        # Stats Panel
        self.stats_frame = tk.Frame(main_frame, bd=2, relief=tk.SUNKEN)

        tk.Label(self.stats_frame, text="Live Stats", font=("Arial", 12, "bold")).pack(pady=5, padx=10)
        tk.Label(self.stats_frame, textvariable=self.base_velocity_var, font=("Arial", 10)).pack(pady=2, padx=10, anchor="w")
        tk.Label(self.stats_frame, textvariable=self.flyer_velocity_var, font=("Arial", 10)).pack(pady=2, padx=10, anchor="w")
        tk.Label(self.stats_frame, textvariable=self.base_wobble_var, font=("Arial", 10)).pack(pady=2, padx=10, anchor="w")
        tk.Label(self.stats_frame, textvariable=self.flyer_wobble_var, font=("Arial", 10)).pack(pady=2, padx=10, anchor="w")
        tk.Label(self.stats_frame, textvariable=self.alignment_var, font=("Arial", 10)).pack(pady=2, padx=10, anchor="w")
        tk.Label(self.stats_frame, textvariable=self.plumb_line_var, font=("Arial", 10)).pack(pady=2, padx=10, anchor="w")
        tk.Label(self.stats_frame, textvariable=self.stunt_score_var, font=("Arial", 14, "bold"), fg="#0077c2").pack(pady=10, padx=10, anchor="center")

        self.canvas_middle.bind("<Button-1>", self.on_roi_press)
        self.canvas_middle.bind("<B1-Motion>", self.on_roi_drag)
        self.canvas_middle.bind("<ButtonRelease-1>", self.on_roi_release)
        self.canvas_middle.bind("<Button-3>", self.on_canvas_right_click)
        self.root.bind("<Configure>", self.on_window_resize)

        self.slider = tk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_slider_move)
        self.slider.pack(fill=tk.X, padx=10, pady=5)

        self.controls_frame = tk.Frame(self.root)
        self.controls_frame.pack(pady=5)

        self.btn_open = tk.Button(self.controls_frame, text="Open Video", command=self.open_video)
        self.btn_open.pack(side=tk.LEFT, padx=5)
        self.btn_play = tk.Button(self.controls_frame, text="Play/Pause", command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=5)
        self.btn_save = tk.Button(self.controls_frame, text="Save Video", command=self.save_video)
        self.btn_save.pack(side=tk.LEFT, padx=5)
        self.btn_save_csv = tk.Button(self.controls_frame, text="Save CSV", command=self.save_csv_data)
        self.btn_save_csv.pack(side=tk.LEFT, padx=5)
        self.btn_save_viz = tk.Button(self.controls_frame, text="Save CSV + Viz", command=self.save_csv_and_viz)
        self.btn_save_viz.pack(side=tk.LEFT, padx=5)
        self.btn_swap = tk.Button(self.controls_frame, text="Swap Base/Flyer", command=self.swap_roles)
        self.btn_swap.pack(side=tk.LEFT, padx=5)

        self.chk_roi = tk.Checkbutton(self.controls_frame, text="Enable ROI Tracking", var=self.roi_tracking_enabled, command=self.on_roi_toggle)
        self.chk_roi.pack(side=tk.LEFT, padx=10)
        self.chk_base = tk.Checkbutton(self.controls_frame, text="Track Base", var=self.track_base, command=self.on_visibility_toggle)
        self.chk_base.pack(side=tk.LEFT, padx=5)
        self.chk_flyer = tk.Checkbutton(self.controls_frame, text="Track Flyer", var=self.track_flyer, command=self.on_visibility_toggle)
        self.chk_flyer.pack(side=tk.LEFT, padx=5)
        self.chk_stats = tk.Checkbutton(self.controls_frame, text="Show Stats", var=self.show_stats, command=self.on_visibility_toggle)
        self.chk_stats.pack(side=tk.LEFT, padx=10)

        # Passive role-inversion warning — roles are never auto-swapped
        self.role_hint_label = tk.Label(self.root, textvariable=self.role_hint_var, fg="#cc6600", font=("Arial", 10, "bold"))
        self.role_hint_label.pack()

        self.adv_controls_frame = tk.LabelFrame(self.root, text="Tracking Controls", padx=10, pady=5)
        self.adv_controls_frame.pack(padx=10, pady=5, fill=tk.X)

        row1 = tk.Frame(self.adv_controls_frame)
        row1.pack(fill=tk.X)
        tk.Label(row1, text="Smoothing:").pack(side=tk.LEFT, padx=(0, 5))
        self.smoothing_slider = tk.Scale(row1, from_=1, to=30, orient=tk.HORIZONTAL, variable=self.smoothing_window_var, command=self.on_smoothing_update)
        self.smoothing_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(row1, text="Confidence:").pack(side=tk.LEFT, padx=(15, 5))
        tk.Scale(row1, from_=0.05, to=0.95, resolution=0.05, orient=tk.HORIZONTAL, variable=self.conf_threshold_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        row2 = tk.Frame(self.adv_controls_frame)
        row2.pack(fill=tk.X)
        tk.Label(row2, text="Keypoint Vis:").pack(side=tk.LEFT, padx=(0, 5))
        tk.Scale(row2, from_=0.05, to=0.95, resolution=0.05, orient=tk.HORIZONTAL, variable=self.keypoint_vis_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.crop_frame = tk.LabelFrame(self.root, text="Crop Controls", padx=10, pady=5)
        self.crop_frame.pack(padx=10, pady=5, fill=tk.X)
        tk.Checkbutton(self.crop_frame, text="Crop Detection Area", variable=self.detection_crop_enabled, command=self.on_crop_toggle).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(self.crop_frame, text="Crop Output Video",   variable=self.output_crop_enabled,    command=self.on_crop_toggle).pack(side=tk.LEFT, padx=5)
        tk.Button(self.crop_frame, text="Reset Crop", command=self.reset_crop).pack(side=tk.LEFT, padx=10)
        tk.Label(self.crop_frame, text="← drag yellow box on video to reposition", fg="gray").pack(side=tk.LEFT, padx=5)

    def on_window_resize(self, event):
        if event.widget is not self.root or not self.video_path:
            return
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(100, self._apply_resize)

    def _apply_resize(self):
        self._resize_job = None
        bottom_h = (self.slider.winfo_height() +
                    self.controls_frame.winfo_height() +
                    self.adv_controls_frame.winfo_height() + 30)
        avail_w = self.root.winfo_width() - 20
        avail_h = self.root.winfo_height() - bottom_h - 20

        if self.show_stats.get():
            avail_w -= self.stats_frame.winfo_width() + 10

        canvas_w = (avail_w - 30) // 3
        canvas_h = avail_h

        aspect = self.video_width / self.video_height
        if canvas_w / max(canvas_h, 1) > aspect:
            canvas_w = int(canvas_h * aspect)
        else:
            canvas_h = int(canvas_w / aspect)

        if canvas_w < 100 or canvas_h < 100:
            return
        if canvas_w == self.display_width and canvas_h == self.display_height:
            return

        self.display_width = canvas_w
        self.display_height = canvas_h
        for canvas in [self.canvas_left, self.canvas_middle, self.canvas_right]:
            canvas.config(width=self.display_width, height=self.display_height)

        if not self.playing and self.current_frame_data is not None:
            self.process_and_display_frame()

    def open_video(self):
        videos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw_videos')
        os.makedirs(videos_dir, exist_ok=True)

        self.video_path = filedialog.askopenfilename(
            initialdir=videos_dir,
            title="Select a video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.MOV")]
        )
        if not self.video_path: return

        self.playing = False
        self.base_smoother.clear(); self.flyer_smoother.clear()
        self.track_state = self._new_track_state()
        self._inversion_count = 0
        self.role_hint_var.set("")
        self._reset_tracker()
        self.cap = cv2.VideoCapture(self.video_path)
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.config(to=self.total_frames - 1)

        self.display_width = int(self.video_width * (self.display_height / self.video_height))
        for canvas in [self.canvas_left, self.canvas_middle, self.canvas_right]:
            canvas.config(width=self.display_width, height=self.display_height)
        self.detection_crop = {"x": 0, "y": 0, "w": self.display_width, "h": self.display_height, "name": "crop"}

        self.playing = True
        self.video_loop()

    def video_loop(self):
        if self.playing and self.cap and self.cap.isOpened():
            current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if self.last_frame_time is None:
                self.last_frame_time = current_time

            ret, frame = self.cap.read()
            if ret:
                self.current_frame_data = frame
                time_delta = current_time - self.last_frame_time
                self.process_and_display_frame(time_delta)
                self.last_frame_time = current_time
                self.root.after(15, self.video_loop)
            else:
                self.playing = False
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if ret: self.current_frame_data = frame
                self.process_and_display_frame(0)

    def process_and_display_frame(self, time_delta=0):
        if self.current_frame_data is None: return
        frame = self.current_frame_data

        original_frame = frame.copy()
        overlay_frame = frame.copy()
        pose_only_frame = np.zeros_like(frame)

        if self.roi_tracking_enabled.get():
            base_results, flyer_results = self.find_poses_by_roi(frame)
        else:
            base_results, flyer_results = self.find_poses_auto(frame)

        smoothed_base = self.base_smoother.smooth(base_results)
        smoothed_flyer = self.flyer_smoother.smooth(flyer_results)

        if self.show_stats.get():
            self.update_stats_panel(smoothed_base, smoothed_flyer, time_delta)

        all_poses = [] if self.roi_tracking_enabled.get() else self._last_detected_poses
        self.draw_classified_poses(overlay_frame, smoothed_base, smoothed_flyer, all_poses)
        self.draw_classified_poses(pose_only_frame, smoothed_base, smoothed_flyer, all_poses)

        self.display_all_frames(original_frame, overlay_frame, pose_only_frame)
        if self.cap: self.slider.set(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))

    def display_all_frames(self, original, overlay, pose_only):
        for key, frame_data in [("left", original), ("middle", overlay), ("right", pose_only)]:
            canvas = getattr(self, f"canvas_{key}")
            img_resized = cv2.resize(frame_data, (self.display_width, self.display_height))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
            setattr(self, f"photo_{key}", photo)
            canvas.create_image(0, 0, image=photo, anchor=tk.NW)

        if self.roi_tracking_enabled.get() or self.detection_crop_enabled.get() or self.output_crop_enabled.get():
            self.draw_rois_on_canvas(self.canvas_middle)

    def draw_rois_on_canvas(self, canvas):
        canvas.delete("roi")
        if self.roi_tracking_enabled.get():
            for roi, color in [(self.base_roi, "red"), (self.flyer_roi, "blue")]:
                if (roi["name"] == "base" and self.track_base.get()) or (roi["name"] == "flyer" and self.track_flyer.get()):
                    x1, y1, x2, y2 = roi["x"], roi["y"], roi["x"] + roi["w"], roi["y"] + roi["h"]
                    canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags="roi")
                    s = self.handle_size // 2
                    for h_pos in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                        canvas.create_rectangle(h_pos[0]-s, h_pos[1]-s, h_pos[0]+s, h_pos[1]+s, fill=color, outline=color, tags="roi")
        if self.detection_crop_enabled.get() or self.output_crop_enabled.get():
            c = self.detection_crop
            x1, y1, x2, y2 = c["x"], c["y"], c["x"] + c["w"], c["y"] + c["h"]
            canvas.create_rectangle(x1, y1, x2, y2, outline="#FFD700", width=2, dash=(8, 4), tags="roi")
            s = self.handle_size // 2
            for pos in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                canvas.create_rectangle(pos[0]-s, pos[1]-s, pos[0]+s, pos[1]+s, fill="#FFD700", outline="#FFD700", tags="roi")
            label = "Detect+Output" if (self.detection_crop_enabled.get() and self.output_crop_enabled.get()) else ("Detect" if self.detection_crop_enabled.get() else "Output")
            canvas.create_text(x1 + 4, y1 + 4, text=f"[{label}]", fill="#FFD700", anchor=tk.NW, tags="roi")

    def find_poses_by_roi(self, frame):
        base_results, flyer_results = None, None
        if self.track_base.get():
            base_results = self.process_single_roi(frame, self.base_roi)
        if self.track_flyer.get():
            flyer_results = self.process_single_roi(frame, self.flyer_roi)
        return base_results, flyer_results

    def process_single_roi(self, frame, roi):
        fh, fw = frame.shape[:2]
        rx, ry, rw, rh = self.get_scaled_roi(roi, fw, fh)
        rx2, ry2 = min(rx + rw, fw), min(ry + rh, fh)
        crop = frame[ry:ry2, rx:rx2]
        if crop.size == 0:
            return None
        if self.yolo_roi is None:
            self.yolo_roi = YOLO('yolov8m-pose.pt')
        yolo_out = self.yolo_roi(crop, verbose=False)[0]
        if yolo_out.keypoints is None or len(yolo_out.boxes) == 0:
            return None
        best_i = int(yolo_out.boxes.conf.argmax())
        if float(yolo_out.boxes.conf[best_i]) < self.conf_threshold_var.get():
            return None
        kp_xyn = yolo_out.keypoints.xyn[best_i].cpu().numpy()
        kp_conf = yolo_out.keypoints.conf[best_i].cpu().numpy()
        lm_list = self._yolo_to_landmarks(kp_xyn, kp_conf)
        self.translate_landmarks(lm_list, rx, ry, rx2 - rx, ry2 - ry, fw, fh)
        return SmoothedResults(lm_list)

    def _yolo_to_landmarks(self, kp_xyn, kp_conf):
        """Convert YOLO normalized keypoints to a LandmarkList."""
        return LandmarkList([
            Landmark(x=float(kp_xyn[i][0]), y=float(kp_xyn[i][1]), visibility=float(kp_conf[i]))
            for i in range(len(kp_xyn))
        ])

    def _pose_bottom_y(self, pose):
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

    def _pose_avg_x(self, pose):
        visible = [lm for lm in pose.landmark if lm.visibility > 0.3]
        if not visible:
            return 0.5
        return sum(lm.x for lm in visible) / len(visible)

    def _best_stack_pair(self, ids, by_id):
        """Pick (flyer_id, base_id): flyer = person whose lowest point is highest
        in frame; base = person most horizontally aligned directly below them."""
        ordered = sorted(ids, key=lambda t: self._pose_bottom_y(by_id[t]))
        flyer_id = ordered[0]
        flyer_x = self._pose_avg_x(by_id[flyer_id])
        flyer_bottom = self._pose_bottom_y(by_id[flyer_id])
        below = [t for t in ordered[1:] if self._pose_bottom_y(by_id[t]) > flyer_bottom]
        if not below:
            below = ordered[1:]
        base_id = min(below, key=lambda t: abs(self._pose_avg_x(by_id[t]) - flyer_x))
        return flyer_id, base_id

    def _new_track_state(self):
        return {
            'base_id': None, 'flyer_id': None,   # ByteTrack IDs bound to each role
            'base_lm': None, 'flyer_lm': None,   # last known landmarks per role
            'base_miss': 0, 'flyer_miss': 0,     # consecutive frames the role's track was absent
            'initialized': False,                # roles picked (heuristic or user)
        }

    def _reset_tracker(self):
        """Clear ByteTrack state so a new video / export pass starts fresh."""
        predictor = getattr(self.yolo, 'predictor', None)
        for tracker in getattr(predictor, 'trackers', None) or []:
            if hasattr(tracker, 'reset'):
                tracker.reset()

    def _invalidate_role_ids(self):
        """Track IDs restart after a tracker reset: drop the stale bindings and
        let the overlap re-bind recover each role from its last known box."""
        s = self.track_state
        s['base_id'] = s['flyer_id'] = None
        s['base_miss'] = s['flyer_miss'] = 5  # eligible for immediate re-bind

    def _rebind_role(self, state, role, by_id, w, h):
        """A role's track died and ByteTrack couldn't recover it: re-bind the
        role to the unassigned track that best overlaps its last known box."""
        last_lm = state[f'{role}_lm']
        if last_lm is None:
            return None
        other_id = state['flyer_id'] if role == 'base' else state['base_id']
        ref_box = self.get_bounding_box(last_lm.landmark, w, h)
        best_id, best_iou = None, 0.25
        for tid, pose in by_id.items():
            if tid == other_id:
                continue
            iou = self.calculate_iou(ref_box, self.get_bounding_box(pose.landmark, w, h))
            if iou > best_iou:
                best_iou, best_id = iou, tid
        return best_id

    def _detect_poses_auto(self, frame, state):
        """Single-pass multi-person detection with ByteTrack persistent IDs.
        Roles are sticky: base/flyer are bound to track IDs once (heuristic on
        the first frame with 2+ people, or by the user) and follow those IDs —
        they are never silently reassigned by per-frame geometry.
        Mutates `state`; returns (base_obj, flyer_obj)."""
        h, w = frame.shape[:2]

        detect_frame, dc_x, dc_y = frame, 0, 0
        if self.detection_crop_enabled.get():
            rx, ry, rw, rh = self.get_scaled_roi(self.detection_crop, w, h)
            rx2, ry2 = min(rx + rw, w), min(ry + rh, h)
            if rx2 > rx and ry2 > ry:
                detect_frame = frame[ry:ry2, rx:rx2]
                dc_x, dc_y = rx, ry

        # conf=0.1 feeds low-confidence boxes to ByteTrack, whose second
        # association stage uses them to hold tracks through occlusion —
        # critical when the flyer covers the base. iou=0.7 keeps NMS from
        # suppressing the stacked pair down to one detection. The UI
        # confidence slider is applied below, but never to the boxes carrying
        # the pair's track IDs.
        yolo_out = self.yolo.track(detect_frame, persist=True, verbose=False, conf=0.1, iou=0.7)[0]

        conf = self.conf_threshold_var.get()
        role_ids = {state['base_id'], state['flyer_id']}
        poses, ids = [], []
        if yolo_out.keypoints is not None and yolo_out.boxes.id is not None:
            for i in range(len(yolo_out.boxes)):
                tid = int(yolo_out.boxes.id[i])
                if float(yolo_out.boxes.conf[i]) < conf and tid not in role_ids:
                    continue
                kp_xyn = yolo_out.keypoints.xyn[i].cpu().numpy()
                kp_conf = yolo_out.keypoints.conf[i].cpu().numpy()
                lm_list = self._yolo_to_landmarks(kp_xyn, kp_conf)
                if dc_x or dc_y:
                    dh, dw = detect_frame.shape[:2]
                    self.translate_landmarks(lm_list, dc_x, dc_y, dw, dh, w, h)
                poses.append(lm_list)
                ids.append(tid)
        self._last_detected_poses = poses
        self._last_detected_ids = ids
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
                if state[f'{role}_miss'] >= 5:
                    new_id = self._rebind_role(state, role, by_id, w, h)
                    if new_id is not None:
                        state[f'{role}_id'] = new_id
                        current = by_id[new_id]
            if current is not None:
                state[f'{role}_miss'] = 0
                state[f'{role}_lm'] = current

        current_base = by_id.get(state['base_id'])
        current_flyer = by_id.get(state['flyer_id'])

        # Raw pair identity so draw_classified_poses can single out spotters
        self._raw_base_pose  = current_base
        self._raw_flyer_pose = current_flyer

        base_obj  = SmoothedResults(current_base)  if current_base  else None
        flyer_obj = SmoothedResults(current_flyer) if current_flyer else None
        return base_obj, flyer_obj

    def find_poses_auto(self, frame):
        base_results, flyer_results = self._detect_poses_auto(frame, self.track_state)
        self._update_role_hint(base_results, flyer_results)
        return base_results, flyer_results

    def _update_role_hint(self, base_results, flyer_results):
        """Passive warning when the flyer has sat below the base for a while —
        roles are never auto-swapped, the user decides."""
        if base_results and flyer_results:
            flyer_bottom = self._pose_bottom_y(flyer_results.pose_landmarks)
            base_bottom  = self._pose_bottom_y(base_results.pose_landmarks)
            if flyer_bottom > base_bottom + 0.05:
                self._inversion_count += 1
            else:
                self._inversion_count = 0
        else:
            self._inversion_count = 0
        if self._inversion_count >= 15:
            self.role_hint_var.set("⚠ Flyer is below base — roles may be swapped (use Swap Base/Flyer, or right-click a person)")
        elif self._inversion_count == 0:
            self.role_hint_var.set("")

    def draw_classified_poses(self, frame, base_results, flyer_results, all_poses=None):
        fh, fw = frame.shape[:2]
        vis_thresh = self.keypoint_vis_var.get()

        # Draw all non-pair people as thin grey lines first (renders under the colored pair)
        if all_poses:
            for pose in all_poses:
                if pose is self._raw_base_pose or pose is self._raw_flyer_pose:
                    continue
                lm = pose.landmark
                for a, b in YOLO_CONNECTIONS:
                    if (a < len(lm) and b < len(lm)
                            and lm[a].visibility > vis_thresh and lm[b].visibility > vis_thresh):
                        p1 = (int(lm[a].x * fw), int(lm[a].y * fh))
                        p2 = (int(lm[b].x * fw), int(lm[b].y * fh))
                        cv2.line(frame, p1, p2, (90, 90, 90), 1, cv2.LINE_AA)

        # Draw base (red) and flyer (blue) in full color on top
        for results, color, enabled in [
            (base_results,  (0, 0, 255), self.track_base.get()),
            (flyer_results, (255, 0, 0), self.track_flyer.get()),
        ]:
            if not enabled or not results or not results.pose_landmarks:
                continue
            lm = results.pose_landmarks.landmark
            for a, b in YOLO_CONNECTIONS:
                if (a < len(lm) and b < len(lm)
                        and lm[a].visibility > vis_thresh and lm[b].visibility > vis_thresh):
                    p1 = (int(lm[a].x * fw), int(lm[a].y * fh))
                    p2 = (int(lm[b].x * fw), int(lm[b].y * fh))
                    cv2.line(frame, p1, p2, color, 3, cv2.LINE_AA)

    def on_canvas_right_click(self, event):
        if not self._last_detected_poses:
            return
        # Find the detected person whose centroid is closest to the click
        nx, ny = event.x / self.display_width, event.y / self.display_height
        best_idx, best_dist = -1, float('inf')
        for i, pose in enumerate(self._last_detected_poses):
            visible = [lm for lm in pose.landmark if lm.visibility > 0.3]
            if not visible:
                continue
            cx = sum(lm.x for lm in visible) / len(visible)
            cy = sum(lm.y for lm in visible) / len(visible)
            dist = np.hypot(cx - nx, cy - ny)
            if dist < best_dist:
                best_dist, best_idx = dist, i
        if best_idx == -1:
            return
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Set as Base",  command=lambda: self._assign_role(best_idx, 'base'))
        menu.add_command(label="Set as Flyer", command=lambda: self._assign_role(best_idx, 'flyer'))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _assign_role(self, pose_idx, role):
        if pose_idx >= len(self._last_detected_ids):
            return
        tid = self._last_detected_ids[pose_idx]
        state = self.track_state
        other = 'flyer' if role == 'base' else 'base'
        if state[f'{other}_id'] == tid:
            # User reassigned the other role's person: give the displaced role
            # this role's old track (net effect: a swap)
            state[f'{other}_id'] = state[f'{role}_id']
            state[f'{other}_lm'] = state[f'{role}_lm']
        state[f'{role}_id'] = tid
        state[f'{role}_lm'] = self._last_detected_poses[pose_idx]
        state[f'{role}_miss'] = 0
        state['initialized'] = True
        self._inversion_count = 0
        self.role_hint_var.set("")
        self.base_smoother.clear()
        self.flyer_smoother.clear()
        if not self.playing and self.current_frame_data is not None:
            self.process_and_display_frame()

    def swap_roles(self):
        s = self.track_state
        s['base_id'], s['flyer_id'] = s['flyer_id'], s['base_id']
        s['base_lm'], s['flyer_lm'] = s['flyer_lm'], s['base_lm']
        s['base_miss'], s['flyer_miss'] = s['flyer_miss'], s['base_miss']
        self._inversion_count = 0
        self.role_hint_var.set("")
        self.base_smoother.clear()
        self.flyer_smoother.clear()
        if self.current_frame_data is not None:
            self.process_and_display_frame()

    def on_slider_move(self, value):
        if self.cap and abs(self.cap.get(cv2.CAP_PROP_POS_FRAMES) - int(value)) > 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(value))
            if not self.playing:
                ret, frame = self.cap.read()
                if ret: self.current_frame_data = frame
                self.process_and_display_frame()

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing: self.video_loop()

    def on_roi_toggle(self):
        if self.roi_tracking_enabled.get():
            self.track_base.set(True); self.track_flyer.set(True)
        if self.current_frame_data is not None:
            self.process_and_display_frame()

    def on_crop_toggle(self):
        if self.current_frame_data is not None:
            self.process_and_display_frame()

    def reset_crop(self):
        self.detection_crop.update({"x": 0, "y": 0, "w": self.display_width, "h": self.display_height})
        if self.current_frame_data is not None:
            self.process_and_display_frame()

    def on_visibility_toggle(self):
        if self.show_stats.get():
            self.stats_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)
        else:
            self.stats_frame.pack_forget()
        if self.current_frame_data is not None:
            self.process_and_display_frame()

    def get_handle_at(self, x, y):
        s = self.handle_size // 2
        rois = []
        if self.roi_tracking_enabled.get():
            for roi in [self.flyer_roi, self.base_roi]:
                if (roi["name"] == "base" and not self.track_base.get()) or (roi["name"] == "flyer" and not self.track_flyer.get()): continue
                rois.append(roi)
        if self.detection_crop_enabled.get() or self.output_crop_enabled.get():
            rois.append(self.detection_crop)
        for roi in rois:
            x1, y1, x2, y2 = roi["x"], roi["y"], roi["x"] + roi["w"], roi["y"] + roi["h"]
            handles = {"tl": (x1, y1), "tr": (x2, y1), "bl": (x1, y2), "br": (x2, y2)}
            for name, pos in handles.items():
                if pos[0]-s <= x <= pos[0]+s and pos[1]-s <= y <= pos[1]+s: return roi, name
        return None, None

    def on_roi_press(self, event):
        if not self.roi_tracking_enabled.get() and not self.detection_crop_enabled.get() and not self.output_crop_enabled.get(): return
        roi, handle = self.get_handle_at(event.x, event.y)
        if handle:
            self.active_roi = roi
            self.drag_info = {"type": "resize", "handle": handle, "orig_x": event.x, "orig_y": event.y, "roi_x": roi["x"], "roi_y": roi["y"], "roi_w": roi["w"], "roi_h": roi["h"]}
        else:
            candidates = []
            if self.roi_tracking_enabled.get():
                candidates.extend([self.flyer_roi, self.base_roi])
            if self.detection_crop_enabled.get() or self.output_crop_enabled.get():
                candidates.append(self.detection_crop)
            for r in candidates:
                if r["x"] <= event.x <= r["x"] + r["w"] and r["y"] <= event.y <= r["y"] + r["h"]:
                    self.active_roi = r
                    self.drag_info = {"type": "move", "orig_x": event.x, "orig_y": event.y, "roi_x": r["x"], "roi_y": r["y"]}; break

    def on_roi_drag(self, event):
        if not self.active_roi: return
        dx, dy = event.x - self.drag_info["orig_x"], event.y - self.drag_info["orig_y"]
        if self.drag_info["type"] == "move":
            self.active_roi["x"] = self.drag_info["roi_x"] + dx
            self.active_roi["y"] = self.drag_info["roi_y"] + dy
        else:
            x, y, w, h = self.drag_info["roi_x"], self.drag_info["roi_y"], self.drag_info["roi_w"], self.drag_info["roi_h"]
            if self.drag_info["handle"] in ["br", "tr"]: self.active_roi["w"] = max(20, w + dx)
            if self.drag_info["handle"] in ["br", "bl"]: self.active_roi["h"] = max(20, h + dy)
            if self.drag_info["handle"] in ["tl", "bl"]: self.active_roi["x"] = x + dx; self.active_roi["w"] = max(20, w - dx)
            if self.drag_info["handle"] in ["tl", "tr"]: self.active_roi["y"] = y + dy; self.active_roi["h"] = max(20, h - dy)
        if not self.playing: self.process_and_display_frame()

    def on_roi_release(self, event):
        if not self.playing and self.active_roi: self.process_and_display_frame()
        self.active_roi = None; self.drag_info = {}

    def on_smoothing_update(self, val):
        n = int(val)
        self.base_smoother.set_window(n)
        self.flyer_smoother.set_window(n)

    def update_stats_panel(self, base_results, flyer_results, time_delta):
        h, w = self.video_height, self.video_width

        base_lm  = base_results.pose_landmarks  if base_results  else None
        flyer_lm = flyer_results.pose_landmarks if flyer_results else None

        base_com   = self.calculate_center_of_mass(base_lm, w, h)
        flyer_com  = self.calculate_center_of_mass(flyer_lm, w, h)
        base_torso = self.get_torso_length(base_lm, w, h)
        flyer_torso = self.get_torso_length(flyer_lm, w, h)
        ref_torso  = base_torso or flyer_torso

        # Velocity in torso lengths / second — camera-distance invariant
        if base_com and self.last_com_base and time_delta > 0 and ref_torso:
            vel = np.linalg.norm(np.array(base_com) - np.array(self.last_com_base)) / time_delta / ref_torso
            self.base_velocity_var.set(f"Base Vel: {vel:.2f} TL/s")
        else:
            self.base_velocity_var.set("Base Vel: N/A")
        self.last_com_base = base_com

        if flyer_com and self.last_com_flyer and time_delta > 0 and ref_torso:
            vel = np.linalg.norm(np.array(flyer_com) - np.array(self.last_com_flyer)) / time_delta / ref_torso
            self.flyer_velocity_var.set(f"Flyer Vel: {vel:.2f} TL/s")
        else:
            self.flyer_velocity_var.set("Flyer Vel: N/A")
        self.last_com_flyer = flyer_com

        # 2D wobble: std dev across both axes, normalized by torso length
        if base_com: self.base_com_history.append(base_com)
        if flyer_com: self.flyer_com_history.append(flyer_com)

        if len(self.base_com_history) > 5 and ref_torso:
            pts = np.array(list(self.base_com_history))
            wobble = np.mean(np.std(pts, axis=0)) / ref_torso
            self.base_wobble_var.set(f"Base Wobble: {wobble:.3f} TL")
        else:
            self.base_wobble_var.set("Base Wobble: N/A")

        if len(self.flyer_com_history) > 5 and ref_torso:
            pts = np.array(list(self.flyer_com_history))
            wobble = np.mean(np.std(pts, axis=0)) / ref_torso
            self.flyer_wobble_var.set(f"Flyer Wobble: {wobble:.3f} TL")
        else:
            self.flyer_wobble_var.set("Flyer Wobble: N/A")

        # Joint Alignment (Base): shoulder/hip/ankle vertical stack deviation, normalized
        if base_lm and base_torso:
            lm = base_lm.landmark
            shoulder_x = (lm[KP_L_SHOULDER].x + lm[KP_R_SHOULDER].x) / 2
            hip_x      = (lm[KP_L_HIP].x      + lm[KP_R_HIP].x)      / 2
            ankle_x    = (lm[KP_L_ANKLE].x     + lm[KP_R_ANKLE].x)    / 2
            alignment_px = (abs(shoulder_x - hip_x) + abs(hip_x - ankle_x)) * w
            self.alignment_var.set(f"Alignment: {alignment_px / base_torso:.3f} TL")
        else:
            self.alignment_var.set("Alignment: N/A")

        # Plumb Line: horizontal CoM offset between base and flyer, normalized
        if base_com and flyer_com and ref_torso:
            plumb = abs(base_com[0] - flyer_com[0]) / ref_torso
            self.plumb_line_var.set(f"Plumb Line: {plumb:.3f} TL")
        else:
            self.plumb_line_var.set("Plumb Line: N/A")

        # Stunt Score: all components normalized so it's camera-invariant
        if base_com and flyer_com and len(self.flyer_com_history) > 5 and ref_torso:
            flyer_height_score = (1 - flyer_com[1] / h) * 100
            pts = np.array(list(self.flyer_com_history))
            flyer_wobble_norm  = np.mean(np.std(pts, axis=0)) / ref_torso
            flyer_wobble_score = max(0, 100 - flyer_wobble_norm * 500)
            plumb_norm  = abs(base_com[0] - flyer_com[0]) / ref_torso
            plumb_score = max(0, 100 - plumb_norm * 100)
            score = (flyer_height_score * 0.4) + (flyer_wobble_score * 0.3) + (plumb_score * 0.3)
            self.stunt_score_var.set(f"Stunt Score: {score:.1f}")
        else:
            self.stunt_score_var.set("Stunt Score: N/A")

    def get_bounding_box(self, landmarks, w, h):
        # Filter out undetected keypoints (YOLO returns 0,0 with conf~0)
        pts = [(lm.x * w, lm.y * h) for lm in landmarks if lm.visibility > 0.3]
        if not pts:
            pts = [(lm.x * w, lm.y * h) for lm in landmarks]
        x_coords, y_coords = zip(*pts)
        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

    def calculate_iou(self, box1, box2):
        x1_inter, y1_inter = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2_inter, y2_inter = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter_area / (box1_area + box2_area + 1e-6)

    def calculate_center_of_mass(self, landmarks, w, h):
        if not landmarks: return None
        com_indices = {
            'torso_center': (KP_L_SHOULDER, KP_R_SHOULDER, KP_L_HIP, KP_R_HIP),
            'legs':         (KP_L_KNEE, KP_R_KNEE, KP_L_ANKLE, KP_R_ANKLE),
            'arms':         (KP_L_ELBOW, KP_R_ELBOW, KP_L_WRIST, KP_R_WRIST),
        }
        total_weight = 0
        com_x, com_y = 0, 0
        for indices in com_indices.values():
            for idx in indices:
                if idx < len(landmarks.landmark):
                    lm = landmarks.landmark[idx]
                    if lm.visibility > 0.5:
                        com_x += lm.x * w
                        com_y += lm.y * h
                        total_weight += 1
        if total_weight == 0: return None
        return (com_x / total_weight, com_y / total_weight)

    def get_torso_length(self, landmarks, w, h):
        """Pixel distance from avg shoulder to avg hip — used to normalize all stats."""
        if not landmarks: return None
        lm = landmarks.landmark
        sx = (lm[KP_L_SHOULDER].x + lm[KP_R_SHOULDER].x) / 2 * w
        sy = (lm[KP_L_SHOULDER].y + lm[KP_R_SHOULDER].y) / 2 * h
        hx = (lm[KP_L_HIP].x      + lm[KP_R_HIP].x)      / 2 * w
        hy = (lm[KP_L_HIP].y      + lm[KP_R_HIP].y)      / 2 * h
        length = np.hypot(sx - hx, sy - hy)
        return length if length > 0 else None

    def get_scaled_roi(self, roi, frame_w, frame_h):
        scale_x, scale_y = frame_w / self.display_width, frame_h / self.display_height
        return int(roi["x"]*scale_x), int(roi["y"]*scale_y), int(roi["w"]*scale_x), int(roi["h"]*scale_y)

    def translate_landmarks(self, lm_list, crop_x, crop_y, crop_w, crop_h, frame_w, frame_h):
        for lm in lm_list.landmark:
            lm.x = (lm.x * crop_w + crop_x) / frame_w
            lm.y = (lm.y * crop_h + crop_y) / frame_h

    def save_video(self):
        if not self.video_path: messagebox.showwarning("No Video", "Please open a video file first."); return
        dialog = tk.Toplevel(self.root); dialog.title("Save Options"); dialog.geometry("300x100"); dialog.resizable(False, False)
        dialog.geometry(f"+{self.root.winfo_x()+150}+{self.root.winfo_y()+150}")
        tk.Label(dialog, text="Choose what to save:").pack(pady=10)
        btn_frame = tk.Frame(dialog); btn_frame.pack()
        def save_and_close(mocap_only): dialog.destroy(); self._execute_save(mocap_only)
        tk.Button(btn_frame, text="Video with Mocap", command=lambda: save_and_close(False)).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Mocap Only", command=lambda: save_and_close(True)).pack(side=tk.LEFT, padx=10)

    def _execute_save(self, mocap_only):
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'edited_videos')
        os.makedirs(save_dir, exist_ok=True)
        save_path = filedialog.asksaveasfilename(initialdir=save_dir, defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")], title="Save Video As")
        if not save_path: return
        cap = cv2.VideoCapture(self.video_path)
        width, height, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)

        out_w, out_h, crop_x, crop_y = width, height, 0, 0
        if self.output_crop_enabled.get():
            rx, ry, rw, rh = self.get_scaled_roi(self.detection_crop, width, height)
            cx1 = max(0, min(rx, width));  cy1 = max(0, min(ry, height))
            cx2 = max(0, min(rx + rw, width)); cy2 = max(0, min(ry + rh, height))
            if cx2 > cx1 and cy2 > cy1:
                out_w, out_h, crop_x, crop_y = cx2 - cx1, cy2 - cy1, cx1, cy1

        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
        save_base_sm  = PoseSmoother(self.smoothing_window_var.get())
        save_flyer_sm = PoseSmoother(self.smoothing_window_var.get())
        save_state = self._new_track_state()
        self._reset_tracker()  # export re-runs the video from frame 0 with its own tracking state
        roi_enabled = self.roi_tracking_enabled.get()
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_dialog = tk.Toplevel(self.root); progress_dialog.title("Saving...")
        progress_label = tk.Label(progress_dialog, text=f"Processing frame 0/{num_frames}"); progress_label.pack(padx=20, pady=10)
        progress_dialog.geometry(f"+{self.root.winfo_x()+150}+{self.root.winfo_y()+150}"); self.root.update_idletasks()
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret: break
            output_frame = np.zeros_like(frame) if mocap_only else frame.copy()
            if roi_enabled:
                base_results, flyer_results = self.find_poses_by_roi(frame)
            else:
                base_results, flyer_results = self._detect_poses_auto(frame, save_state)
            smoothed_base, smoothed_flyer = save_base_sm.smooth(base_results), save_flyer_sm.smooth(flyer_results)
            save_all_poses = [] if roi_enabled else self._last_detected_poses
            self.draw_classified_poses(output_frame, smoothed_base, smoothed_flyer, save_all_poses)
            if out_w < width or out_h < height:
                output_frame = output_frame[crop_y:crop_y + out_h, crop_x:crop_x + out_w]
            out.write(output_frame)
            progress_label.config(text=f"Processing frame {i+1}/{num_frames}"); self.root.update_idletasks()
        cap.release(); out.release(); progress_dialog.destroy()
        # The export polluted the shared tracker: reset it and let live playback
        # re-bind its roles from their last known boxes
        self._reset_tracker()
        self._invalidate_role_ids()
        messagebox.showinfo("Save Complete", f"Video saved to {save_path}")

    def save_csv_data(self):
        if not self.video_path:
            messagebox.showwarning("No Video", "Please open a video file first.")
            return
        self._execute_csv_save()

    def _execute_csv_save(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Save CSV As")
        if not save_path:
            return

        cap = cv2.VideoCapture(self.video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        progress_dialog = tk.Toplevel(self.root)
        progress_dialog.title("Saving CSV...")
        progress_label = tk.Label(progress_dialog, text=f"Processing frame 0/{num_frames}")
        progress_label.pack(padx=20, pady=10)
        progress_dialog.geometry(f"+{self.root.winfo_x()+150}+{self.root.winfo_y()+150}")
        self.root.update_idletasks()

        save_base_sm  = PoseSmoother(self.smoothing_window_var.get())
        save_flyer_sm = PoseSmoother(self.smoothing_window_var.get())
        save_state = self._new_track_state()
        self._reset_tracker()  # export re-runs the video from frame 0 with its own tracking state
        roi_enabled = self.roi_tracking_enabled.get()

        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'person_id', 'landmark', 'x', 'y', 'visibility'])

            for i in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                if roi_enabled:
                    base_results, flyer_results = self.find_poses_by_roi(frame)
                else:
                    base_results, flyer_results = self._detect_poses_auto(frame, save_state)

                smoothed_base  = save_base_sm.smooth(base_results)
                smoothed_flyer = save_flyer_sm.smooth(flyer_results)

                for person_id, results in [('base', smoothed_base), ('flyer', smoothed_flyer)]:
                    if results and results.pose_landmarks:
                        for j, lm in enumerate(results.pose_landmarks.landmark):
                            writer.writerow([i, person_id, j, lm.x, lm.y, lm.visibility])

                progress_label.config(text=f"Processing frame {i+1}/{num_frames}")
                self.root.update_idletasks()

        cap.release()
        progress_dialog.destroy()
        # The export polluted the shared tracker: reset it and let live playback
        # re-bind its roles from their last known boxes
        self._reset_tracker()
        self._invalidate_role_ids()
        messagebox.showinfo("Save Complete", f"CSV data saved to {save_path}")
        return save_path

    def save_csv_and_viz(self):
        if not self.video_path:
            messagebox.showwarning("No Video", "Please open a video file first.")
            return
        csv_path = self._execute_csv_save()
        if csv_path:
            self._generate_visualization(csv_path)

    def _generate_visualization(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                messagebox.showwarning("Empty Data", "The CSV file is empty, cannot generate visualization.")
                return

            com_indices = {
                'torso_center': (KP_L_SHOULDER, KP_R_SHOULDER, KP_L_HIP, KP_R_HIP),
                'legs':         (KP_L_KNEE, KP_R_KNEE, KP_L_ANKLE, KP_R_ANKLE),
                'arms':         (KP_L_ELBOW, KP_R_ELBOW, KP_L_WRIST, KP_R_WRIST),
            }
            all_indices = [idx for indices in com_indices.values() for idx in indices]

            com_data = []
            for frame_num, frame_df in df.groupby('frame'):
                for person_id, person_df in frame_df.groupby('person_id'):
                    total_weight, com_x, com_y = 0, 0, 0
                    for idx in all_indices:
                        lm = person_df[person_df['landmark'] == idx]
                        if not lm.empty and lm['visibility'].iloc[0] > 0.5:
                            com_x += lm['x'].iloc[0]
                            com_y += lm['y'].iloc[0]
                            total_weight += 1
                    if total_weight > 0:
                        com_data.append({
                            'frame': frame_num,
                            'person_id': person_id,
                            'com_y': 1 - (com_y / total_weight)
                        })

            if not com_data:
                messagebox.showwarning("No Data", "Could not calculate Center of Mass from the data.")
                return

            viz_df = pd.DataFrame(com_data)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            viz_df['y_velocity']     = viz_df.groupby('person_id')['com_y'].diff().fillna(0) * fps
            viz_df['y_acceleration'] = viz_df.groupby('person_id')['y_velocity'].diff().fillna(0) * fps

            color_map = {'base': 'red', 'flyer': 'blue'}
            fig_height = px.line(viz_df, x='frame', y='com_y',      color='person_id', title='Vertical Center of Mass Over Time',  labels={'frame': 'Frame', 'com_y': 'Vertical Position (Normalized)'}, color_discrete_map=color_map)
            fig_vel    = px.line(viz_df, x='frame', y='y_velocity',  color='person_id', title='Vertical Velocity Over Time',         labels={'frame': 'Frame', 'y_velocity': 'Velocity (normalized/s)'},   color_discrete_map=color_map)
            fig_accel  = px.line(viz_df, x='frame', y='y_acceleration', color='person_id', title='Vertical Acceleration Over Time', labels={'frame': 'Frame', 'y_acceleration': 'Acceleration (normalized/s²)'}, color_discrete_map=color_map)
            for fig in [fig_height, fig_vel, fig_accel]:
                fig.update_layout(legend_title_text='Performer')

            html_path = os.path.splitext(csv_path)[0] + '_visualization.html'
            with open(html_path, 'w') as f:
                f.write("<html><head><title>Stunt Analysis</title></head><body>\n")
                f.write("<h1 style='text-align: center;'>Stunt Performance Analysis</h1>\n")
                f.write(fig_height.to_html(full_html=False, include_plotlyjs='cdn'))
                f.write(fig_vel.to_html(full_html=False, include_plotlyjs=False))
                f.write(fig_accel.to_html(full_html=False, include_plotlyjs=False))
                f.write("</body></html>")

            messagebox.showinfo("Visualization Saved", f"Interactive visualization saved to {html_path}")

        except Exception as e:
            messagebox.showerror("Visualization Error", f"An error occurred while creating the visualization: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = StuntCVApp(root)
    root.mainloop()
