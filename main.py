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

        # YOLOv8 Pose model — downloads yolov8n-pose.pt automatically on first run
        self.yolo = YOLO('yolov8n-pose.pt')

        # UI-Controlled Tracking Parameters
        self.smoothing_window_var = tk.IntVar(value=10)

        # Smoothing Deques
        self.base_history = deque(maxlen=self.smoothing_window_var.get())
        self.flyer_history = deque(maxlen=self.smoothing_window_var.get())

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

        # Tracking State
        self.last_base_results = None
        self.last_flyer_results = None

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

        self.adv_controls_frame = tk.LabelFrame(self.root, text="Tracking Controls", padx=10, pady=10)
        self.adv_controls_frame.pack(padx=10, pady=5, fill=tk.X)

        tk.Label(self.adv_controls_frame, text="Smoothing:").pack(side=tk.LEFT, padx=(0, 5))
        self.smoothing_slider = tk.Scale(self.adv_controls_frame, from_=1, to=30, orient=tk.HORIZONTAL, variable=self.smoothing_window_var, command=self.on_smoothing_update)
        self.smoothing_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

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
        self.base_history.clear(); self.flyer_history.clear()
        self.last_base_results, self.last_flyer_results = None, None
        self.cap = cv2.VideoCapture(self.video_path)
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.config(to=self.total_frames - 1)

        self.display_width = int(self.video_width * (self.display_height / self.video_height))
        for canvas in [self.canvas_left, self.canvas_middle, self.canvas_right]:
            canvas.config(width=self.display_width, height=self.display_height)

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

        smoothed_base = self.smooth_pose(base_results, self.base_history)
        smoothed_flyer = self.smooth_pose(flyer_results, self.flyer_history)

        if self.show_stats.get():
            self.update_stats_panel(smoothed_base, smoothed_flyer, time_delta)

        self.draw_classified_poses(overlay_frame, smoothed_base, smoothed_flyer)
        self.draw_classified_poses(pose_only_frame, smoothed_base, smoothed_flyer)

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

        if self.roi_tracking_enabled.get():
            self.draw_rois_on_canvas(self.canvas_middle)

    def draw_rois_on_canvas(self, canvas):
        canvas.delete("roi")
        for roi, color in [(self.base_roi, "red"), (self.flyer_roi, "blue")]:
            if (roi["name"] == "base" and self.track_base.get()) or (roi["name"] == "flyer" and self.track_flyer.get()):
                x1, y1, x2, y2 = roi["x"], roi["y"], roi["x"] + roi["w"], roi["y"] + roi["h"]
                canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags="roi")
                s = self.handle_size // 2
                for h_pos in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                    canvas.create_rectangle(h_pos[0]-s, h_pos[1]-s, h_pos[0]+s, h_pos[1]+s, fill=color, outline=color, tags="roi")

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
        yolo_out = self.yolo(crop, verbose=False)[0]
        if yolo_out.keypoints is None or len(yolo_out.boxes) == 0:
            return None
        best_i = int(yolo_out.boxes.conf.argmax())
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

    def _pose_avg_y(self, pose):
        """Mean y of visible landmarks. Lower value = higher in frame = more likely the flyer."""
        visible = [lm for lm in pose.landmark if lm.visibility > 0.3]
        if not visible:
            return 0.5
        return sum(lm.y for lm in visible) / len(visible)

    def _best_match(self, ref_lm, candidates, exclude, w, h):
        """Returns index of best matching pose using IoU with centroid-distance fallback."""
        ref_box = self.get_bounding_box(ref_lm.landmark, w, h)
        ref_cx = (ref_box[0] + ref_box[2]) / 2
        ref_cy = (ref_box[1] + ref_box[3]) / 2
        ref_h = max(ref_box[3] - ref_box[1], 1)

        best_iou_idx, best_iou = -1, 0.0
        best_cent_idx, best_cent_dist = -1, float('inf')

        for i, pose in enumerate(candidates):
            if i in exclude:
                continue
            box = self.get_bounding_box(pose.landmark, w, h)
            iou = self.calculate_iou(ref_box, box)
            if iou > best_iou:
                best_iou, best_iou_idx = iou, i
            dist = np.hypot((box[0] + box[2]) / 2 - ref_cx, (box[1] + box[3]) / 2 - ref_cy)
            if dist < best_cent_dist:
                best_cent_dist, best_cent_idx = dist, i

        if best_iou >= 0.3:
            return best_iou_idx
        # Centroid fallback: handles fast toss/catch where boxes stop overlapping between frames
        if best_cent_dist < ref_h * 1.5:
            return best_cent_idx
        return -1

    def _detect_poses_auto(self, frame, last_base_lm, last_flyer_lm):
        """Single-pass multi-person detection via YOLOv8.
        Returns (base_obj, flyer_obj, raw_base_lm, raw_flyer_lm)."""
        h, w = frame.shape[:2]
        yolo_out = self.yolo(frame, verbose=False)[0]

        detected_poses = []
        if yolo_out.keypoints is not None:
            for i in range(len(yolo_out.boxes)):
                if float(yolo_out.boxes.conf[i]) < 0.3:
                    continue
                kp_xyn = yolo_out.keypoints.xyn[i].cpu().numpy()
                kp_conf = yolo_out.keypoints.conf[i].cpu().numpy()
                detected_poses.append(self._yolo_to_landmarks(kp_xyn, kp_conf))
        self._last_detected_poses = detected_poses

        current_base, current_flyer = None, None

        if last_base_lm or last_flyer_lm:
            matched_indices = set()

            if last_base_lm:
                idx = self._best_match(last_base_lm, detected_poses, matched_indices, w, h)
                if idx != -1:
                    current_base = detected_poses[idx]
                    matched_indices.add(idx)

            if last_flyer_lm:
                idx = self._best_match(last_flyer_lm, detected_poses, matched_indices, w, h)
                if idx != -1:
                    current_flyer = detected_poses[idx]
                    matched_indices.add(idx)

            # Assign any unmatched poses using height bias rather than arrival order:
            # the highest person in frame goes to the flyer role, lowest to base.
            unmatched = [detected_poses[i] for i in range(len(detected_poses)) if i not in matched_indices]
            if unmatched:
                by_height = sorted(unmatched, key=self._pose_avg_y)
                if not current_flyer:
                    current_flyer = by_height[0]    # highest in frame → flyer
                    by_height = by_height[1:]
                if not current_base and by_height:
                    current_base = by_height[-1]    # lowest remaining → base
        else:
            # First frame: assign by vertical extremes so spotters in the middle are ignored.
            # Use mean y (not raw sum) so landmark count doesn't bias the sort.
            if len(detected_poses) == 1:
                current_base = detected_poses[0]
            elif len(detected_poses) >= 2:
                by_y = sorted(detected_poses, key=self._pose_avg_y)
                current_flyer = by_y[0]   # smallest mean y = highest in frame
                current_base  = by_y[-1]  # largest mean y  = lowest in frame

        # Preserve last known position on tracking loss so next frame can re-acquire
        new_base_lm  = current_base  if current_base  else last_base_lm
        new_flyer_lm = current_flyer if current_flyer else last_flyer_lm

        base_obj  = SmoothedResults(current_base)  if current_base  else None
        flyer_obj = SmoothedResults(current_flyer) if current_flyer else None
        return base_obj, flyer_obj, new_base_lm, new_flyer_lm

    def find_poses_auto(self, frame):
        base_results, flyer_results, self.last_base_results, self.last_flyer_results = self._detect_poses_auto(
            frame, self.last_base_results, self.last_flyer_results
        )
        return base_results, flyer_results

    def draw_classified_poses(self, frame, base_results, flyer_results):
        fh, fw = frame.shape[:2]
        for results, color, enabled in [
            (base_results,  (0, 0, 255), self.track_base.get()),
            (flyer_results, (255, 0, 0), self.track_flyer.get()),
        ]:
            if not enabled or not results or not results.pose_landmarks:
                continue
            lm = results.pose_landmarks.landmark
            for a, b in YOLO_CONNECTIONS:
                if (a < len(lm) and b < len(lm)
                        and lm[a].visibility > 0.5 and lm[b].visibility > 0.5):
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
        if pose_idx >= len(self._last_detected_poses):
            return
        pose = self._last_detected_poses[pose_idx]
        if role == 'base':
            self.last_base_results = pose
        else:
            self.last_flyer_results = pose
        # Clear smoothing history so the new assignment doesn't blend with the old person
        self.base_history.clear()
        self.flyer_history.clear()
        if not self.playing and self.current_frame_data is not None:
            self.process_and_display_frame()

    def swap_roles(self):
        self.last_base_results, self.last_flyer_results = self.last_flyer_results, self.last_base_results
        self.base_history.clear()
        self.flyer_history.clear()
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

    def on_visibility_toggle(self):
        if self.show_stats.get():
            self.stats_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)
        else:
            self.stats_frame.pack_forget()
        if self.current_frame_data is not None:
            self.process_and_display_frame()

    def get_handle_at(self, x, y):
        s = self.handle_size // 2
        for roi in [self.flyer_roi, self.base_roi]:
            if (roi["name"] == "base" and not self.track_base.get()) or (roi["name"] == "flyer" and not self.track_flyer.get()): continue
            x1, y1, x2, y2 = roi["x"], roi["y"], roi["x"] + roi["w"], roi["y"] + roi["h"]
            handles = {"tl": (x1, y1), "tr": (x2, y1), "bl": (x1, y2), "br": (x2, y2)}
            for name, pos in handles.items():
                if pos[0]-s <= x <= pos[0]+s and pos[1]-s <= y <= pos[1]+s: return roi, name
        return None, None

    def on_roi_press(self, event):
        if not self.roi_tracking_enabled.get(): return
        roi, handle = self.get_handle_at(event.x, event.y)
        if handle:
            self.active_roi = roi
            self.drag_info = {"type": "resize", "handle": handle, "orig_x": event.x, "orig_y": event.y, "roi_x": roi["x"], "roi_y": roi["y"], "roi_w": roi["w"], "roi_h": roi["h"]}
        else:
            for r in [self.flyer_roi, self.base_roi]:
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
        new_window_size = int(val)
        if not hasattr(self, 'base_history') or new_window_size != self.base_history.maxlen:
            self.base_history = deque(maxlen=new_window_size)
            self.flyer_history = deque(maxlen=new_window_size)

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

    def smooth_pose(self, results, history):
        if not results or not results.pose_landmarks:
            history.clear()
            return None
        history.append(results.pose_landmarks.landmark)
        n = len(history[0])
        smoothed = [
            Landmark(
                x=sum(f[i].x for f in history) / len(history),
                y=sum(f[i].y for f in history) / len(history),
                visibility=sum(f[i].visibility for f in history) / len(history),
            )
            for i in range(n)
        ]
        return SmoothedResults(LandmarkList(smoothed))

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
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        save_base_hist, save_flyer_hist = deque(maxlen=self.smoothing_window_var.get()), deque(maxlen=self.smoothing_window_var.get())
        last_base_results, last_flyer_results = None, None
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
                base_results, flyer_results, last_base_results, last_flyer_results = self._find_poses_auto_for_save(frame, last_base_results, last_flyer_results)
            smoothed_base, smoothed_flyer = self.smooth_pose(base_results, save_base_hist), self.smooth_pose(flyer_results, save_flyer_hist)
            self.draw_classified_poses(output_frame, smoothed_base, smoothed_flyer)
            out.write(output_frame)
            progress_label.config(text=f"Processing frame {i+1}/{num_frames}"); self.root.update_idletasks()
        cap.release(); out.release(); progress_dialog.destroy()
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

        save_base_hist, save_flyer_hist = deque(maxlen=self.smoothing_window_var.get()), deque(maxlen=self.smoothing_window_var.get())
        last_base_results, last_flyer_results = None, None
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
                    base_results, flyer_results, last_base_results, last_flyer_results = self._find_poses_auto_for_save(frame, last_base_results, last_flyer_results)

                smoothed_base  = self.smooth_pose(base_results, save_base_hist)
                smoothed_flyer = self.smooth_pose(flyer_results, save_flyer_hist)

                for person_id, results in [('base', smoothed_base), ('flyer', smoothed_flyer)]:
                    if results and results.pose_landmarks:
                        for j, lm in enumerate(results.pose_landmarks.landmark):
                            writer.writerow([i, person_id, j, lm.x, lm.y, lm.visibility])

                progress_label.config(text=f"Processing frame {i+1}/{num_frames}")
                self.root.update_idletasks()

        cap.release()
        progress_dialog.destroy()
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

    def _find_poses_auto_for_save(self, frame, last_base, last_flyer):
        return self._detect_poses_auto(frame, last_base, last_flyer)


if __name__ == "__main__":
    root = tk.Tk()
    app = StuntCVApp(root)
    root.mainloop()
