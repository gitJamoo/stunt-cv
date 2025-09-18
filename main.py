import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import csv
from collections import deque
from mediapipe.framework.formats import landmark_pb2

import pandas as pd
import plotly.express as px
import os

class SmoothedResults:
    def __init__(self, landmarks):
        self.pose_landmarks = landmarks

class StuntCVApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stunt CV - ROI Tracking")

        # Core App State
        self.video_path = None
        self.cap = None
        self.playing = False
        self.video_width, self.video_height, self.total_frames = 0, 0, 0
        self.display_width, self.display_height = 480, 360
        self.current_frame_data = None

        # MediaPipe Pose Setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

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
        self.base_com_history = deque(maxlen=15) # For wobbliness calculation
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
        # Main content frame
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
        # Don't pack it yet, will be controlled by checkbox

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
        
        self.chk_roi = tk.Checkbutton(self.controls_frame, text="Enable ROI Tracking", var=self.roi_tracking_enabled, command=self.on_roi_toggle)
        self.chk_roi.pack(side=tk.LEFT, padx=10)
        self.chk_base = tk.Checkbutton(self.controls_frame, text="Track Base", var=self.track_base, command=self.on_visibility_toggle)
        self.chk_base.pack(side=tk.LEFT, padx=5)
        self.chk_flyer = tk.Checkbutton(self.controls_frame, text="Track Flyer", var=self.track_flyer, command=self.on_visibility_toggle)
        self.chk_flyer.pack(side=tk.LEFT, padx=5)
        self.chk_stats = tk.Checkbutton(self.controls_frame, text="Show Stats", var=self.show_stats, command=self.on_visibility_toggle)
        self.chk_stats.pack(side=tk.LEFT, padx=10)

        # Advanced Controls
        self.adv_controls_frame = tk.LabelFrame(self.root, text="Tracking Controls", padx=10, pady=10)
        self.adv_controls_frame.pack(padx=10, pady=5, fill=tk.X)

        tk.Label(self.adv_controls_frame, text="Smoothing:").pack(side=tk.LEFT, padx=(0, 5))
        self.smoothing_slider = tk.Scale(self.adv_controls_frame, from_=1, to=30, orient=tk.HORIZONTAL, variable=self.smoothing_window_var, command=self.on_smoothing_update)
        self.smoothing_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def open_video(self):
        # Define the target directory for videos, create it if it doesn't exist
        videos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw_videos')
        os.makedirs(videos_dir, exist_ok=True)

        self.video_path = filedialog.askopenfilename(
            initialdir=videos_dir,
            title="Select a video file",
            filetypes=[("Video files", "*.mp4 *.avi")]
        )
        if not self.video_path: return

        self.playing = False
        self.base_history.clear(); self.flyer_history.clear()
        self.last_base_results, self.last_flyer_results = None, None # Reset trackers
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
        h, w, _ = frame.shape
        rx, ry, rw, rh = self.get_scaled_roi(roi, w, h)
        if rx < w and ry < h:
            crop = frame[ry:ry+rh, rx:rx+rw]
            if crop.size > 0:
                results = self.pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                if results.pose_landmarks:
                    self.translate_landmarks(results.pose_landmarks, rx, ry, rw, rh, w, h)
                    return results
        return None

    def find_poses_auto(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        all_results = self.pose.process(frame_rgb).pose_landmarks

        # This is a simplified stand-in for a multi-pose detection model.
        # We process the frame once, then blank out the first person and process again.
        detected_poses = []
        if all_results:
            detected_poses.append(all_results)
            frame_rgb_copy = np.copy(frame_rgb)
            box1 = self.get_bounding_box(all_results.landmark, w, h)
            cv2.rectangle(frame_rgb_copy, (int(box1[0])-10, int(box1[1])-10), (int(box1[2])+10, int(box1[3])+10), (0,0,0), -1)
            
            second_results = self.pose.process(frame_rgb_copy).pose_landmarks
            if second_results:
                box2 = self.get_bounding_box(second_results.landmark, w, h)
                # Simple check to ensure the second person is reasonably distinct
                if self.calculate_iou(box1, box2) < 0.1:
                    detected_poses.append(second_results)

        current_base, current_flyer = None, None

        # If we have previous tracking data, use it to match
        if self.last_base_results or self.last_flyer_results:
            matched_indices = set()

            # Match for Base
            if self.last_base_results:
                last_box = self.get_bounding_box(self.last_base_results.landmark, w, h)
                best_match_idx, best_iou = -1, 0
                for i, pose in enumerate(detected_poses):
                    current_box = self.get_bounding_box(pose.landmark, w, h)
                    iou = self.calculate_iou(last_box, current_box)
                    if i not in matched_indices and iou > best_iou:
                        best_iou = iou
                        best_match_idx = i
                if best_match_idx != -1 and best_iou > 0.1: # Threshold for a valid match
                    current_base = detected_poses[best_match_idx]
                    matched_indices.add(best_match_idx)

            # Match for Flyer
            if self.last_flyer_results:
                last_box = self.get_bounding_box(self.last_flyer_results.landmark, w, h)
                best_match_idx, best_iou = -1, 0
                for i, pose in enumerate(detected_poses):
                    if i in matched_indices: continue
                    current_box = self.get_bounding_box(pose.landmark, w, h)
                    iou = self.calculate_iou(last_box, current_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_match_idx = i
                if best_match_idx != -1 and best_iou > 0.1:
                    current_flyer = detected_poses[best_match_idx]
                    matched_indices.add(best_match_idx)
            
            # Assign any remaining poses
            for i, pose in enumerate(detected_poses):
                if i not in matched_indices:
                    if not current_base:
                        current_base = pose
                    elif not current_flyer:
                        current_flyer = pose

        # If no tracking data, use vertical position as a fallback for the first frame
        else:
            if len(detected_poses) == 1:
                current_base = detected_poses[0]
            elif len(detected_poses) == 2:
                avg_y1 = sum(lm.y for lm in detected_poses[0].landmark)
                avg_y2 = sum(lm.y for lm in detected_poses[1].landmark)
                if avg_y1 > avg_y2:
                    current_base, current_flyer = detected_poses[0], detected_poses[1]
                else:
                    current_base, current_flyer = detected_poses[1], detected_poses[0]

        # Wrap results in the expected class structure and update state
        base_results_obj, flyer_results_obj = None, None
        if current_base:
            base_results_obj = SmoothedResults(current_base)
            self.last_base_results = current_base
        else:
            self.last_base_results = None

        if current_flyer:
            flyer_results_obj = SmoothedResults(current_flyer)
            self.last_flyer_results = current_flyer
        else:
            self.last_flyer_results = None

        return base_results_obj, flyer_results_obj

    def draw_classified_poses(self, frame, base_results, flyer_results):
        if self.track_base.get() and base_results and base_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, base_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2), connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,0,255), thickness=2))
        if self.track_flyer.get() and flyer_results and flyer_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, flyer_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2), connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=2))

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
        # Re-initialize deques only if the size has actually changed
        if not hasattr(self, 'base_history') or new_window_size != self.base_history.maxlen:
            self.base_history = deque(maxlen=new_window_size)
            self.flyer_history = deque(maxlen=new_window_size)

    def update_stats_panel(self, base_results, flyer_results, time_delta):
        h, w = self.video_height, self.video_width
        
        base_com = self.calculate_center_of_mass(base_results.pose_landmarks if base_results else None, w, h)
        flyer_com = self.calculate_center_of_mass(flyer_results.pose_landmarks if flyer_results else None, w, h)

        # Velocity
        if base_com and self.last_com_base and time_delta > 0:
            self.base_velocity_var.set(f"Base Vel: {np.linalg.norm(np.array(base_com) - np.array(self.last_com_base)) / time_delta:.2f} pps")
        else: self.base_velocity_var.set("Base Vel: N/A")
        self.last_com_base = base_com

        if flyer_com and self.last_com_flyer and time_delta > 0:
            self.flyer_velocity_var.set(f"Flyer Vel: {np.linalg.norm(np.array(flyer_com) - np.array(self.last_com_flyer)) / time_delta:.2f} pps")
        else: self.flyer_velocity_var.set("Flyer Vel: N/A")
        self.last_com_flyer = flyer_com

        # Wobbliness
        if base_com: self.base_com_history.append(base_com)
        if flyer_com: self.flyer_com_history.append(flyer_com)
        if len(self.base_com_history) > 5: # Need a few frames to calculate wobble
            wobble = np.std([c[0] for c in self.base_com_history]) # Horizontal wobble
            self.base_wobble_var.set(f"Base Wobble: {wobble:.2f}")
        else: self.base_wobble_var.set("Base Wobble: N/A")
        if len(self.flyer_com_history) > 5:
            wobble = np.std([c[0] for c in self.flyer_com_history])
            self.flyer_wobble_var.set(f"Flyer Wobble: {wobble:.2f}")
        else: self.flyer_wobble_var.set("Flyer Wobble: N/A")

        # Joint Alignment (Base)
        if base_results and base_results.pose_landmarks:
            lm = base_results.pose_landmarks.landmark
            shoulder_x = (lm[11].x + lm[12].x) / 2
            hip_x = (lm[23].x + lm[24].x) / 2
            ankle_x = (lm[27].x + lm[28].x) / 2
            alignment = (abs(shoulder_x - hip_x) + abs(hip_x - ankle_x)) * w
            self.alignment_var.set(f"Alignment: {alignment:.2f} px")
        else: self.alignment_var.set("Alignment: N/A")

        # Plumb Line
        if base_com and flyer_com:
            plumb_line = abs(base_com[0] - flyer_com[0])
            self.plumb_line_var.set(f"Plumb Line: {plumb_line:.2f} px")
        else: self.plumb_line_var.set("Plumb Line: N/A")

        # Stunt Score
        if base_com and flyer_com and len(self.flyer_com_history) > 5:
            flyer_height_score = (1 - flyer_com[1] / h) * 100 # Higher is better
            flyer_wobble_score = max(0, 100 - np.std([c[0] for c in self.flyer_com_history]))
            plumb_score = max(0, 100 - abs(base_com[0] - flyer_com[0]))
            # Simple weighted average
            score = (flyer_height_score * 0.4) + (flyer_wobble_score * 0.3) + (plumb_score * 0.3)
            self.stunt_score_var.set(f"Stunt Score: {score:.1f}")
        else: self.stunt_score_var.set("Stunt Score: N/A")

    def get_bounding_box(self, landmarks, w, h):
        x_coords = [lm.x * w for lm in landmarks]; y_coords = [lm.y * h for lm in landmarks]
        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

    def calculate_iou(self, box1, box2):
        x1_inter, y1_inter = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2_inter, y2_inter = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1]); box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter_area / (box1_area + box2_area + 1e-6)

    def calculate_center_of_mass(self, landmarks, w, h):
        if not landmarks: return None
        # A simple approximation of CoM using key body landmarks
        # Weights can be adjusted for better accuracy
        com_indices = {
            'torso_center': (11, 12, 23, 24), # Shoulders and Hips
            'legs': (25, 26, 27, 28),
            'arms': (13, 14, 15, 16)
        }
        total_weight = 0
        com_x, com_y = 0, 0
        
        for part, indices in com_indices.items():
            weight = 1.0 # Simple weighting
            for idx in indices:
                if idx < len(landmarks.landmark):
                    lm = landmarks.landmark[idx]
                    if lm.visibility > 0.5: # Only include visible landmarks
                        com_x += lm.x * w * weight
                        com_y += lm.y * h * weight
                        total_weight += weight
        
        if total_weight == 0: return None
        return (com_x / total_weight, com_y / total_weight)

    def get_scaled_roi(self, roi, frame_w, frame_h):
        scale_x, scale_y = frame_w / self.display_width, frame_h / self.display_height
        return int(roi["x"]*scale_x), int(roi["y"]*scale_y), int(roi["w"]*scale_x), int(roi["h"]*scale_y)

    def translate_landmarks(self, landmarks, crop_x, crop_y, crop_w, crop_h, frame_w, frame_h):
        for lm in landmarks.landmark: lm.x = (lm.x * crop_w + crop_x) / frame_w; lm.y = (lm.y * crop_h + crop_y) / frame_h

    def smooth_pose(self, results, history):
        if not results or not results.pose_landmarks: history.clear(); return None
        history.append(results.pose_landmarks.landmark)
        smoothed_list = landmark_pb2.NormalizedLandmarkList()
        for i in range(len(history[0])):
            avg_x = sum(frame[i].x for frame in history) / len(history); avg_y = sum(frame[i].y for frame in history) / len(history)
            avg_z = sum(frame[i].z for frame in history) / len(history); avg_vis = sum(frame[i].visibility for frame in history) / len(history)
            lm = smoothed_list.landmark.add(); lm.x, lm.y, lm.z, lm.visibility = avg_x, avg_y, avg_z, avg_vis
        return SmoothedResults(smoothed_list)

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
        last_base_results, last_flyer_results = None, None # Local tracker state for saving
        roi_enabled = self.roi_tracking_enabled.get()
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_dialog = tk.Toplevel(self.root); progress_dialog.title("Saving...")
        progress_label = tk.Label(progress_dialog, text=f"Processing frame 0/{num_frames}"); progress_label.pack(padx=20, pady=10)
        progress_dialog.geometry(f"+{self.root.winfo_x()+150}+{self.root.winfo_y()+150}"); self.root.update_idletasks()
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret: break
            output_frame = np.zeros_like(frame) if mocap_only else frame.copy()
            
            # Use a local tracking state for the save process
            if roi_enabled:
                base_results, flyer_results = self.find_poses_by_roi(frame)
            else:
                # We need to replicate the tracking logic here for the save process
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
        last_base_results, last_flyer_results = None, None # Local tracker state for saving
        roi_enabled = self.roi_tracking_enabled.get()

        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['frame', 'person_id', 'landmark', 'x', 'y', 'z', 'visibility']
            writer.writerow(header)

            for i in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                if roi_enabled:
                    base_results, flyer_results = self.find_poses_by_roi(frame)
                else:
                    base_results, flyer_results, last_base_results, last_flyer_results = self._find_poses_auto_for_save(frame, last_base_results, last_flyer_results)

                smoothed_base = self.smooth_pose(base_results, save_base_hist)
                smoothed_flyer = self.smooth_pose(flyer_results, save_flyer_hist)

                for person_id, results in [('base', smoothed_base), ('flyer', smoothed_flyer)]:
                    if results and results.pose_landmarks:
                        for j, lm in enumerate(results.pose_landmarks.landmark):
                            writer.writerow([i, person_id, j, lm.x, lm.y, lm.z, lm.visibility])
                
                progress_label.config(text=f"Processing frame {i+1}/{num_frames}")
                self.root.update_idletasks()

        cap.release()
        progress_dialog.destroy()
        messagebox.showinfo("Save Complete", f"CSV data saved to {save_path}")
        return save_path # Return path on success

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

            # Calculate CoM for each frame from the raw landmark data
            com_data = []
            for frame_num, frame_df in df.groupby('frame'):
                for person_id, person_df in frame_df.groupby('person_id'):
                    # Use the same CoM logic as the live stats
                    com_indices = {
                        'torso_center': (11, 12, 23, 24), # Shoulders and Hips
                        'legs': (25, 26, 27, 28),
                        'arms': (13, 14, 15, 16)
                    }
                    total_weight = 0
                    com_x, com_y = 0, 0
                    for part, indices in com_indices.items():
                        weight = 1.0
                        for idx in indices:
                            lm = person_df[person_df['landmark'] == idx]
                            if not lm.empty and lm['visibility'].iloc[0] > 0.5:
                                com_x += lm['x'].iloc[0] * weight
                                com_y += lm['y'].iloc[0] * weight
                                total_weight += weight
                    
                    if total_weight > 0:
                        com_data.append({
                            'frame': frame_num,
                            'person_id': person_id,
                            'com_y': 1 - (com_y / total_weight) # Invert Y-axis for intuitive plotting (0 at bottom)
                        })
            
            if not com_data:
                messagebox.showwarning("No Data", "Could not calculate Center of Mass from the data.")
                return

            viz_df = pd.DataFrame(com_data)

            # Calculate Velocity and Acceleration
            viz_df['y_velocity'] = viz_df.groupby('person_id')['com_y'].diff().fillna(0) / (1/self.cap.get(cv2.CAP_PROP_FPS))
            viz_df['y_acceleration'] = viz_df.groupby('person_id')['y_velocity'].diff().fillna(0) / (1/self.cap.get(cv2.CAP_PROP_FPS))


            fig_height = px.line(viz_df, x='frame', y='com_y', color='person_id',
                          title='Vertical Center of Mass Over Time',
                          labels={'frame': 'Frame Number', 'com_y': 'Vertical Position (Normalized)', 'person_id': 'Performer'},
                          color_discrete_map={'base': 'red', 'flyer': 'blue'})
            fig_height.update_layout(legend_title_text='Performer')

            fig_vel = px.line(viz_df, x='frame', y='y_velocity', color='person_id',
                          title='Vertical Velocity Over Time',
                          labels={'frame': 'Frame Number', 'y_velocity': 'Vertical Velocity (pixels/sec)', 'person_id': 'Performer'},
                          color_discrete_map={'base': 'red', 'flyer': 'blue'})
            fig_vel.update_layout(legend_title_text='Performer')

            fig_accel = px.line(viz_df, x='frame', y='y_acceleration', color='person_id',
                          title='Vertical Acceleration Over Time',
                          labels={'frame': 'Frame Number', 'y_acceleration': 'Vertical Acceleration (pixels/sec^2)', 'person_id': 'Performer'},
                          color_discrete_map={'base': 'red', 'flyer': 'blue'})
            fig_accel.update_layout(legend_title_text='Performer')

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
        # This is a non-state-updating version of find_poses_auto for file saving
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        all_results = self.pose.process(frame_rgb).pose_landmarks

        detected_poses = []
        if all_results:
            detected_poses.append(all_results)
            frame_rgb_copy = np.copy(frame_rgb)
            box1 = self.get_bounding_box(all_results.landmark, w, h)
            cv2.rectangle(frame_rgb_copy, (int(box1[0])-10, int(box1[1])-10), (int(box1[2])+10, int(box1[3])+10), (0,0,0), -1)
            second_results = self.pose.process(frame_rgb_copy).pose_landmarks
            if second_results:
                box2 = self.get_bounding_box(second_results.landmark, w, h)
                if self.calculate_iou(box1, box2) < 0.1:
                    detected_poses.append(second_results)

        current_base, current_flyer = None, None

        if last_base or last_flyer:
            matched_indices = set()
            if last_base:
                last_box = self.get_bounding_box(last_base.landmark, w, h)
                best_match_idx, best_iou = -1, 0
                for i, pose in enumerate(detected_poses):
                    current_box = self.get_bounding_box(pose.landmark, w, h)
                    iou = self.calculate_iou(last_box, current_box)
                    if i not in matched_indices and iou > best_iou:
                        best_iou = iou
                        best_match_idx = i
                if best_match_idx != -1 and best_iou > 0.1:
                    current_base = detected_poses[best_match_idx]
                    matched_indices.add(best_match_idx)

            if last_flyer:
                last_box = self.get_bounding_box(last_flyer.landmark, w, h)
                best_match_idx, best_iou = -1, 0
                for i, pose in enumerate(detected_poses):
                    if i in matched_indices: continue
                    current_box = self.get_bounding_box(pose.landmark, w, h)
                    iou = self.calculate_iou(last_box, current_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_match_idx = i
                if best_match_idx != -1 and best_iou > 0.1:
                    current_flyer = detected_poses[best_match_idx]
                    matched_indices.add(best_match_idx)
            
            for i, pose in enumerate(detected_poses):
                if i not in matched_indices:
                    if not current_base: current_base = pose
                    elif not current_flyer: current_flyer = pose
        else:
            if len(detected_poses) == 1:
                current_base = detected_poses[0]
            elif len(detected_poses) == 2:
                avg_y1 = sum(lm.y for lm in detected_poses[0].landmark)
                avg_y2 = sum(lm.y for lm in detected_poses[1].landmark)
                if avg_y1 > avg_y2:
                    current_base, current_flyer = detected_poses[0], detected_poses[1]
                else:
                    current_base, current_flyer = detected_poses[1], detected_poses[0]

        base_results_obj = SmoothedResults(current_base) if current_base else None
        flyer_results_obj = SmoothedResults(current_flyer) if current_flyer else None
        
        # Return the new state to be used in the next iteration of the save loop
        return base_results_obj, flyer_results_obj, current_base, current_flyer

if __name__ == "__main__":
    root = tk.Tk()
    app = StuntCVApp(root)
    root.mainloop()