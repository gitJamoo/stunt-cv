"""Stunt CV desktop app: Tkinter UI, video playback, and orchestration.

Logic lives in focused modules — pose_data (types + geometry), tracking
(YOLO + ByteTrack role tracking), stats (live panel metrics), insights
(post-run analysis, Plotly report, DeepSeek chat client). This file should
only contain UI concerns and the glue between them.
"""
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import numpy as np
import csv
import threading
import webbrowser

import pandas as pd
import plotly.express as px
import os

import insights
from pose_data import (YOLO_CONNECTIONS, KP_L_SHOULDER, KP_R_SHOULDER, KP_L_ELBOW,
                       KP_R_ELBOW, KP_L_WRIST, KP_R_WRIST, KP_L_HIP, KP_R_HIP,
                       KP_L_KNEE, KP_R_KNEE, KP_L_ANKLE, KP_R_ANKLE,
                       PoseSmoother, pose_bottom_y, pose_centroid,
                       center_of_mass, torso_length, base_alignment_tl)
from tracking import PoseTracker
from stats import LiveStats


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

        # Detection / tracking / stats engines
        self.tracker = PoseTracker('yolov8m-pose.pt')
        self.live_stats = LiveStats()

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
        self.track_state = PoseTracker.new_state()
        self._inversion_count = 0
        self.role_hint_var = tk.StringVar(value="")

        # Analysis / AI Chat State
        self.analysis_df = None
        self.analysis_fps = None
        self.analysis_insights = ""
        self.chat_window = None
        self.chat_messages = []

        # Playback clock for live stats
        self.last_frame_time = None

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
        self.btn_analyze = tk.Button(self.controls_frame, text="Analyze", command=self.run_analysis)
        self.btn_analyze.pack(side=tk.LEFT, padx=5)
        self.btn_chat = tk.Button(self.controls_frame, text="AI Chat", command=self.open_ai_chat)
        self.btn_chat.pack(side=tk.LEFT, padx=5)

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

    # ------------------------------------------------------------------
    # Layout / resize

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

    # ------------------------------------------------------------------
    # Playback

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
        self.track_state = PoseTracker.new_state()
        self._inversion_count = 0
        self.role_hint_var.set("")
        self.tracker.reset()
        self.live_stats.reset()
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

        all_poses = [] if self.roi_tracking_enabled.get() else self.tracker.last_poses
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

    # ------------------------------------------------------------------
    # Detection (delegates to PoseTracker)

    def find_poses_by_roi(self, frame):
        base_results, flyer_results = None, None
        fh, fw = frame.shape[:2]
        conf = self.conf_threshold_var.get()
        if self.track_base.get():
            base_results = self.tracker.detect_in_roi(frame, self._roi_rect(self.base_roi, fw, fh), conf)
        if self.track_flyer.get():
            flyer_results = self.tracker.detect_in_roi(frame, self._roi_rect(self.flyer_roi, fw, fh), conf)
        return base_results, flyer_results

    def find_poses_auto(self, frame):
        h, w = frame.shape[:2]
        crop_rect = self._roi_rect(self.detection_crop, w, h) if self.detection_crop_enabled.get() else None
        base_results, flyer_results = self.tracker.detect_auto(
            frame, self.track_state, self.conf_threshold_var.get(), crop_rect)
        self._update_role_hint(base_results, flyer_results)
        return base_results, flyer_results

    def _roi_rect(self, roi, frame_w, frame_h):
        """Display-space ROI dict -> clamped (x1, y1, x2, y2) in frame pixels."""
        rx, ry, rw, rh = self.get_scaled_roi(roi, frame_w, frame_h)
        return (max(0, rx), max(0, ry), min(rx + rw, frame_w), min(ry + rh, frame_h))

    def get_scaled_roi(self, roi, frame_w, frame_h):
        scale_x, scale_y = frame_w / self.display_width, frame_h / self.display_height
        return int(roi["x"]*scale_x), int(roi["y"]*scale_y), int(roi["w"]*scale_x), int(roi["h"]*scale_y)

    def _update_role_hint(self, base_results, flyer_results):
        """Passive warning when the flyer has sat below the base for a while —
        roles are never auto-swapped, the user decides."""
        if base_results and flyer_results:
            flyer_bottom = pose_bottom_y(flyer_results.pose_landmarks)
            base_bottom  = pose_bottom_y(base_results.pose_landmarks)
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

    # ------------------------------------------------------------------
    # Drawing

    def draw_classified_poses(self, frame, base_results, flyer_results, all_poses=None):
        fh, fw = frame.shape[:2]
        vis_thresh = self.keypoint_vis_var.get()

        # Draw all non-pair people as thin grey lines first (renders under the colored pair)
        if all_poses:
            for pose in all_poses:
                if pose is self.tracker.raw_base or pose is self.tracker.raw_flyer:
                    continue
                self._draw_skeleton(frame, pose.landmark, (90, 90, 90), 1, vis_thresh, fw, fh)

        # Draw base (red) and flyer (blue) in full color on top
        for results, color, enabled in [
            (base_results,  (0, 0, 255), self.track_base.get()),
            (flyer_results, (255, 0, 0), self.track_flyer.get()),
        ]:
            if not enabled or not results or not results.pose_landmarks:
                continue
            self._draw_skeleton(frame, results.pose_landmarks.landmark, color, 3, vis_thresh, fw, fh)

    @staticmethod
    def _draw_skeleton(frame, lm, color, thickness, vis_thresh, fw, fh):
        for a, b in YOLO_CONNECTIONS:
            if (a < len(lm) and b < len(lm)
                    and lm[a].visibility > vis_thresh and lm[b].visibility > vis_thresh):
                p1 = (int(lm[a].x * fw), int(lm[a].y * fh))
                p2 = (int(lm[b].x * fw), int(lm[b].y * fh))
                cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Role correction

    def on_canvas_right_click(self, event):
        if not self.tracker.last_poses:
            return
        # Find the detected person whose centroid is closest to the click
        nx, ny = event.x / self.display_width, event.y / self.display_height
        best_idx, best_dist = -1, float('inf')
        for i, pose in enumerate(self.tracker.last_poses):
            centroid = pose_centroid(pose)
            if centroid is None:
                continue
            dist = np.hypot(centroid[0] - nx, centroid[1] - ny)
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
        if pose_idx >= len(self.tracker.last_ids):
            return
        tid = self.tracker.last_ids[pose_idx]
        state = self.track_state
        other = 'flyer' if role == 'base' else 'base'
        if state[f'{other}_id'] == tid:
            # User reassigned the other role's person: give the displaced role
            # this role's old track (net effect: a swap)
            state[f'{other}_id'] = state[f'{role}_id']
            state[f'{other}_lm'] = state[f'{role}_lm']
        state[f'{role}_id'] = tid
        state[f'{role}_lm'] = self.tracker.last_poses[pose_idx]
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

    # ------------------------------------------------------------------
    # UI event handlers

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

    # ------------------------------------------------------------------
    # Live stats

    def update_stats_panel(self, base_results, flyer_results, time_delta):
        s = self.live_stats.update(base_results, flyer_results,
                                   self.video_width, self.video_height, time_delta)
        self.base_velocity_var.set(s['base_vel'])
        self.flyer_velocity_var.set(s['flyer_vel'])
        self.base_wobble_var.set(s['base_wobble'])
        self.flyer_wobble_var.set(s['flyer_wobble'])
        self.alignment_var.set(s['alignment'])
        self.plumb_line_var.set(s['plumb'])
        self.stunt_score_var.set(s['score'])

    # ------------------------------------------------------------------
    # Analysis + AI chat

    def run_analysis(self):
        """Re-processes the whole video, builds a metrics DataFrame, opens an
        interactive Plotly report, and feeds the results to the AI chat."""
        if not self.video_path:
            messagebox.showwarning("No Video", "Please open a video file first.")
            return
        self.playing = False

        cap = cv2.VideoCapture(self.video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        progress_dialog, progress_label = self._make_progress_dialog("Analyzing...", num_frames)

        ana_base_sm  = PoseSmoother(self.smoothing_window_var.get())
        ana_flyer_sm = PoseSmoother(self.smoothing_window_var.get())
        ana_state = PoseTracker.new_state()
        self.tracker.reset()  # analysis re-runs the video from frame 0
        roi_enabled = self.roi_tracking_enabled.get()

        rows = []
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if roi_enabled:
                base_results, flyer_results = self.find_poses_by_roi(frame)
            else:
                base_results, flyer_results = self.tracker.detect_auto(frame, ana_state, self.conf_threshold_var.get())
            smoothed_base  = ana_base_sm.smooth(base_results)
            smoothed_flyer = ana_flyer_sm.smooth(flyer_results)

            base_lm  = smoothed_base.pose_landmarks  if smoothed_base  else None
            flyer_lm = smoothed_flyer.pose_landmarks if smoothed_flyer else None
            base_com   = center_of_mass(base_lm, w, h)
            flyer_com  = center_of_mass(flyer_lm, w, h)
            base_torso = torso_length(base_lm, w, h)
            ref_torso  = base_torso or torso_length(flyer_lm, w, h)

            rows.append({
                'frame': i,
                'base_com_x':  base_com[0]  if base_com  else None,
                'base_com_y':  base_com[1]  if base_com  else None,
                'flyer_com_x': flyer_com[0] if flyer_com else None,
                'flyer_com_y': flyer_com[1] if flyer_com else None,
                'ref_torso': ref_torso,
                'alignment_tl': base_alignment_tl(base_lm, w, base_torso),
            })
            progress_label.config(text=f"Processing frame {i+1}/{num_frames}")
            self.root.update_idletasks()

        cap.release()
        progress_dialog.destroy()
        self._recover_live_tracking()

        self.analysis_df = insights.compute_metrics(rows, w, h, fps)
        self.analysis_fps = fps
        self.analysis_insights = insights.generate_insights(self.analysis_df, fps)

        report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'edited_videos')
        os.makedirs(report_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(self.video_path))[0]
        html_path = os.path.join(report_dir, f"{stem}_analysis.html")
        insights.build_analysis_html(self.analysis_df, html_path, self.analysis_insights)
        webbrowser.open(os.path.abspath(html_path))

        chat_was_open = self.chat_window is not None and self.chat_window.winfo_exists()
        self.open_ai_chat()
        if chat_was_open:
            self._append_chat("Insights", self.analysis_insights)

    def open_ai_chat(self):
        if self.chat_window is not None and self.chat_window.winfo_exists():
            self.chat_window.lift()
            return
        win = tk.Toplevel(self.root)
        win.title("AI Coach (DeepSeek)")
        win.geometry("560x620")
        self.chat_window = win

        top = tk.Frame(win)
        top.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(top, text="API Key:").pack(side=tk.LEFT)
        self.api_key_entry = tk.Entry(top, show="*", width=22)
        self.api_key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.api_key_entry.insert(0, os.environ.get("DEEPSEEK_API_KEY", ""))
        tk.Label(top, text="Model:").pack(side=tk.LEFT)
        self.model_entry = tk.Entry(top, width=16)
        self.model_entry.pack(side=tk.LEFT, padx=4)
        self.model_entry.insert(0, "deepseek-chat")

        self.chat_log = scrolledtext.ScrolledText(win, wrap=tk.WORD, state=tk.DISABLED, font=("Arial", 10))
        self.chat_log.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        bottom = tk.Frame(win)
        bottom.pack(fill=tk.X, padx=8, pady=6)
        self.chat_entry = tk.Entry(bottom)
        self.chat_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.chat_entry.bind("<Return>", lambda e: self.send_chat_message())
        self.chat_send_btn = tk.Button(bottom, text="Send", command=self.send_chat_message)
        self.chat_send_btn.pack(side=tk.LEFT, padx=4)

        self.chat_messages = []
        if self.analysis_insights:
            self._append_chat("Insights", self.analysis_insights)
        else:
            self._append_chat("System", "Tip: run Analyze first so the AI can see your stunt's metrics.")

    def _append_chat(self, sender, text):
        self.chat_log.config(state=tk.NORMAL)
        self.chat_log.insert(tk.END, f"{sender}: {text}\n\n")
        self.chat_log.config(state=tk.DISABLED)
        self.chat_log.see(tk.END)

    def send_chat_message(self):
        text = self.chat_entry.get().strip()
        if not text:
            return
        api_key = self.api_key_entry.get().strip()
        if not api_key:
            self._append_chat("System", "Enter your DeepSeek API key first (or set DEEPSEEK_API_KEY).")
            return
        model = self.model_entry.get().strip() or "deepseek-chat"
        self.chat_entry.delete(0, tk.END)
        self._append_chat("You", text)
        self.chat_messages.append({"role": "user", "content": text})

        if self.analysis_df is not None and not self.analysis_df.empty:
            context = insights.summarize_for_llm(self.analysis_df, self.analysis_fps, self.analysis_insights)
        else:
            context = "No analysis has been run yet — answer general stunt questions."
        system_prompt = (
            "You are an experienced cheerleading/acro stunt coach. The user analyzed a stunt video "
            "with pose tracking (base and flyer). Answer their questions using the data below when "
            "relevant. Be concrete and concise.\n\n" + context
        )
        # Keep the last 12 turns so the prompt stays small
        messages = [{"role": "system", "content": system_prompt}] + self.chat_messages[-12:]

        self.chat_send_btn.config(state=tk.DISABLED)
        self._append_chat("AI", "(thinking...)")

        def worker():
            reply, error = None, None
            try:
                reply = insights.DeepSeekClient(api_key, model).chat(messages)
            except Exception as e:
                error = str(e)
            self.root.after(0, lambda: self._on_chat_reply(reply, error))

        threading.Thread(target=worker, daemon=True).start()

    def _on_chat_reply(self, reply, error):
        if self.chat_window is None or not self.chat_window.winfo_exists():
            return
        # Remove the "(thinking...)" placeholder line
        self.chat_log.config(state=tk.NORMAL)
        self.chat_log.delete("end-3l", tk.END)
        self.chat_log.insert(tk.END, "\n")
        self.chat_log.config(state=tk.DISABLED)
        if error:
            self._append_chat("System", f"Request failed: {error}")
        else:
            self.chat_messages.append({"role": "assistant", "content": reply})
            self._append_chat("AI", reply)
        self.chat_send_btn.config(state=tk.NORMAL)

    # ------------------------------------------------------------------
    # Exports

    def _make_progress_dialog(self, title, num_frames):
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        label = tk.Label(dialog, text=f"Processing frame 0/{num_frames}")
        label.pack(padx=20, pady=10)
        dialog.geometry(f"+{self.root.winfo_x()+150}+{self.root.winfo_y()+150}")
        self.root.update_idletasks()
        return dialog, label

    def _recover_live_tracking(self):
        """An export/analysis pass polluted the shared tracker: reset it and let
        live playback re-bind its roles from their last known boxes."""
        self.tracker.reset()
        PoseTracker.invalidate_role_ids(self.track_state)

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
            cx1, cy1, cx2, cy2 = self._roi_rect(self.detection_crop, width, height)
            if cx2 > cx1 and cy2 > cy1:
                out_w, out_h, crop_x, crop_y = cx2 - cx1, cy2 - cy1, cx1, cy1

        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
        save_base_sm  = PoseSmoother(self.smoothing_window_var.get())
        save_flyer_sm = PoseSmoother(self.smoothing_window_var.get())
        save_state = PoseTracker.new_state()
        self.tracker.reset()  # export re-runs the video from frame 0 with its own tracking state
        roi_enabled = self.roi_tracking_enabled.get()
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_dialog, progress_label = self._make_progress_dialog("Saving...", num_frames)
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret: break
            output_frame = np.zeros_like(frame) if mocap_only else frame.copy()
            if roi_enabled:
                base_results, flyer_results = self.find_poses_by_roi(frame)
            else:
                base_results, flyer_results = self.tracker.detect_auto(frame, save_state, self.conf_threshold_var.get())
            smoothed_base, smoothed_flyer = save_base_sm.smooth(base_results), save_flyer_sm.smooth(flyer_results)
            save_all_poses = [] if roi_enabled else self.tracker.last_poses
            self.draw_classified_poses(output_frame, smoothed_base, smoothed_flyer, save_all_poses)
            if out_w < width or out_h < height:
                output_frame = output_frame[crop_y:crop_y + out_h, crop_x:crop_x + out_w]
            out.write(output_frame)
            progress_label.config(text=f"Processing frame {i+1}/{num_frames}"); self.root.update_idletasks()
        cap.release(); out.release(); progress_dialog.destroy()
        self._recover_live_tracking()
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
        progress_dialog, progress_label = self._make_progress_dialog("Saving CSV...", num_frames)

        save_base_sm  = PoseSmoother(self.smoothing_window_var.get())
        save_flyer_sm = PoseSmoother(self.smoothing_window_var.get())
        save_state = PoseTracker.new_state()
        self.tracker.reset()  # export re-runs the video from frame 0 with its own tracking state
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
                    base_results, flyer_results = self.tracker.detect_auto(frame, save_state, self.conf_threshold_var.get())

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
        self._recover_live_tracking()
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
