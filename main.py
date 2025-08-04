
import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from collections import deque
from mediapipe.framework.formats import landmark_pb2

class SmoothedResults:
    def __init__(self, landmarks):
        self.pose_landmarks = landmarks

class StuntCVApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stunt CV - ROI Tracking")

        # Video and Pose Processing Setup
        self.video_path = None
        self.cap = None
        self.playing = False
        self.video_width, self.video_height, self.total_frames = 0, 0, 0
        self.display_width, self.display_height = 480, 360

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        # Smoothing Deques
        self.smoothing_window = 5
        self.base_history = deque(maxlen=self.smoothing_window)
        self.flyer_history = deque(maxlen=self.smoothing_window)

        # ROI Tracking State
        self.roi_tracking_enabled = tk.BooleanVar(value=False)
        self.base_roi = {"x": 50, "y": 150, "w": 150, "h": 200}
        self.flyer_roi = {"x": 250, "y": 50, "w": 150, "h": 200}
        self.active_roi = None
        self.drag_start_pos = None

        self.create_widgets()

    def create_widgets(self):
        self.video_frame = tk.Frame(self.root)
        self.video_frame.pack()

        self.canvas_left = tk.Canvas(self.video_frame, width=self.display_width, height=self.display_height, bg='black')
        self.canvas_left.pack(side=tk.LEFT, padx=5, pady=5)
        self.canvas_middle = tk.Canvas(self.video_frame, width=self.display_width, height=self.display_height, bg='black')
        self.canvas_middle.pack(side=tk.LEFT, padx=5, pady=5)
        self.canvas_right = tk.Canvas(self.video_frame, width=self.display_width, height=self.display_height, bg='black')
        self.canvas_right.pack(side=tk.LEFT, padx=5, pady=5)

        # Bind mouse events for ROI manipulation
        self.canvas_middle.bind("<Button-1>", self.on_roi_move_press)
        self.canvas_middle.bind("<B1-Motion>", self.on_roi_move_drag)
        self.canvas_middle.bind("<ButtonRelease-1>", self.on_roi_move_release)
        self.canvas_middle.bind("<Button-3>", self.on_roi_resize_press)
        self.canvas_middle.bind("<B3-Motion>", self.on_roi_resize_drag)
        self.canvas_middle.bind("<ButtonRelease-3>", self.on_roi_resize_release)

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
        
        self.chk_roi = tk.Checkbutton(self.controls_frame, text="Enable ROI Tracking", var=self.roi_tracking_enabled, command=self.toggle_roi_tracking)
        self.chk_roi.pack(side=tk.LEFT, padx=10)

    def open_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if self.video_path:
            self.playing = False
            self.base_history.clear()
            self.flyer_history.clear()
            self.cap = cv2.VideoCapture(self.video_path)
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.config(to=self.total_frames)
            
            self.display_width = int(self.video_width * (self.display_height / self.video_height))
            for canvas in [self.canvas_left, self.canvas_middle, self.canvas_right]:
                canvas.config(width=self.display_width, height=self.display_height)
            
            self.playing = True
            self.update_frame()

    def update_frame(self):
        if self.playing and self.cap and self.cap.isOpened():
            current_frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.slider.set(current_frame_pos)

            ret, frame = self.cap.read()
            if ret:
                original_frame, overlay_frame, pose_only_frame = frame.copy(), frame.copy(), np.zeros_like(frame)

                if self.roi_tracking_enabled.get():
                    base_results, flyer_results = self.find_poses_by_roi(frame)
                else:
                    base_results, flyer_results = self.find_poses_auto(frame)

                smoothed_base = self.smooth_pose(base_results, self.base_history)
                smoothed_flyer = self.smooth_pose(flyer_results, self.flyer_history)

                self.draw_classified_poses(overlay_frame, smoothed_base, smoothed_flyer)
                self.draw_classified_poses(pose_only_frame, smoothed_base, smoothed_flyer)

                self.display_all_frames(original_frame, overlay_frame, pose_only_frame)
                self.root.after(15, self.update_frame)
            else:
                self.playing = False
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def find_poses_by_roi(self, frame):
        base_results, flyer_results = None, None
        h, w, _ = frame.shape

        for roi_dict, role in [(self.base_roi, "base"), (self.flyer_roi, "flyer")]:
            rx, ry, rw, rh = self.get_scaled_roi(roi_dict, w, h)
            if rx < w and ry < h:
                crop = frame[ry:ry+rh, rx:rx+rw]
                if crop.size > 0:
                    results = self.pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    if results.pose_landmarks:
                        self.translate_landmarks(results.pose_landmarks, rx, ry, w, h)
                        if role == "base": base_results = results
                        else: flyer_results = results
        return base_results, flyer_results

    def find_poses_auto(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        results1 = self.pose.process(frame_rgb)
        results2 = None

        if results1.pose_landmarks:
            frame_rgb_copy = np.copy(frame_rgb)
            box1 = self.get_bounding_box(results1.pose_landmarks.landmark, w, h)
            cv2.rectangle(frame_rgb_copy, (int(box1[0])-10, int(box1[1])-10), (int(box1[2])+10, int(box1[3])+10), (0,0,0), -1)
            results2 = self.pose.process(frame_rgb_copy)

            if results2.pose_landmarks:
                box2 = self.get_bounding_box(results2.pose_landmarks.landmark, w, h)
                if self.calculate_iou(box1, box2) > 0.1: results2 = None
        
        if results1 and results2 and results1.pose_landmarks and results2.pose_landmarks:
            avg_y1 = sum(lm.y for lm in results1.pose_landmarks.landmark)
            avg_y2 = sum(lm.y for lm in results2.pose_landmarks.landmark)
            return (results1, results2) if avg_y1 > avg_y2 else (results2, results1)
        
        return results1, results2

    def draw_classified_poses(self, frame, base_results, flyer_results):
        base_color, flyer_color = (0, 0, 255), (255, 0, 0)
        if base_results and base_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, base_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=base_color, thickness=2, circle_radius=2), connection_drawing_spec=self.mp_drawing.DrawingSpec(color=base_color, thickness=2))
        if flyer_results and flyer_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, flyer_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=flyer_color, thickness=2, circle_radius=2), connection_drawing_spec=self.mp_drawing.DrawingSpec(color=flyer_color, thickness=2))

    def display_all_frames(self, original, overlay, pose_only):
        for key, frame in [("left", original), ("middle", overlay), ("right", pose_only)]:
            canvas = getattr(self, f"canvas_{key}")
            img_resized = cv2.resize(frame, (self.display_width, self.display_height))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
            setattr(self, f"photo_{key}", photo)
            canvas.create_image(0, 0, image=photo, anchor=tk.NW)

        if self.roi_tracking_enabled.get():
            self.draw_rois_on_canvas(self.canvas_middle)

    # --- ROI and Utility Methods ---
    def toggle_roi_tracking(self):
        if not self.playing:
            self.update_frame() # Redraw to show/hide ROIs

    def on_slider_move(self, value):
        if self.cap and abs(self.cap.get(cv2.CAP_PROP_POS_FRAMES) - int(value)) > 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(value))
            if not self.playing: self.update_frame()

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing: self.update_frame()

    def on_roi_move_press(self, event):
        if not self.roi_tracking_enabled.get(): return
        for roi in [self.base_roi, self.flyer_roi]:
            if roi["x"] < event.x < roi["x"] + roi["w"] and roi["y"] < event.y < roi["y"] + roi["h"]:
                self.active_roi = roi
                self.drag_start_pos = (event.x, event.y)
                break

    def on_roi_move_drag(self, event):
        if self.active_roi and self.drag_start_pos and self.active_roi.get("is_resizing") is not True:
            dx = event.x - self.drag_start_pos[0]
            dy = event.y - self.drag_start_pos[1]
            self.active_roi["x"] += dx
            self.active_roi["y"] += dy
            self.drag_start_pos = (event.x, event.y)

    def on_roi_move_release(self, event):
        if self.active_roi: self.active_roi["is_resizing"] = False
        self.active_roi = None
        self.drag_start_pos = None

    def on_roi_resize_press(self, event):
        if not self.roi_tracking_enabled.get(): return
        for roi in [self.base_roi, self.flyer_roi]:
            if roi["x"] < event.x < roi["x"] + roi["w"] and roi["y"] < event.y < roi["y"] + roi["h"]:
                self.active_roi = roi
                self.active_roi["is_resizing"] = True
                self.drag_start_pos = (event.x, event.y)
                break

    def on_roi_resize_drag(self, event):
        if self.active_roi and self.drag_start_pos and self.active_roi.get("is_resizing") is True:
            dx = event.x - self.drag_start_pos[0]
            dy = event.y - self.drag_start_pos[1]
            self.active_roi["w"] += dx
            self.active_roi["h"] += dy
            if self.active_roi["w"] < 20: self.active_roi["w"] = 20
            if self.active_roi["h"] < 20: self.active_roi["h"] = 20
            self.drag_start_pos = (event.x, event.y)

    def on_roi_resize_release(self, event):
        if self.active_roi: self.active_roi["is_resizing"] = False
        self.active_roi = None
        self.drag_start_pos = None

    def draw_rois_on_canvas(self, canvas):
        canvas.delete("roi")
        for roi, color in [(self.base_roi, "red"), (self.flyer_roi, "blue")]:
            canvas.create_rectangle(roi["x"], roi["y"], roi["x"] + roi["w"], roi["y"] + roi["h"], outline=color, width=2, tags="roi")

    def get_scaled_roi(self, roi, frame_w, frame_h):
        scale_x = frame_w / self.display_width
        scale_y = frame_h / self.display_height
        return int(roi["x"]*scale_x), int(roi["y"]*scale_y), int(roi["w"]*scale_x), int(roi["h"]*scale_y)

    def translate_landmarks(self, landmarks, crop_x, crop_y, frame_w, frame_h):
        for lm in landmarks.landmark:
            lm.x = (lm.x * (self.get_scaled_roi(self.base_roi, frame_w, frame_h)[2]))/frame_w + crop_x/frame_w
            lm.y = (lm.y * (self.get_scaled_roi(self.base_roi, frame_w, frame_h)[3]))/frame_h + crop_y/frame_h

    def get_bounding_box(self, landmarks, w, h):
        x_coords = [lm.x * w for lm in landmarks]
        y_coords = [lm.y * h for lm in landmarks]
        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

    def calculate_iou(self, box1, box2):
        x1_inter, y1_inter = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2_inter, y2_inter = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter_area / (box1_area + box2_area - inter_area + 1e-6)

    def smooth_pose(self, results, history):
        if not results or not results.pose_landmarks: 
            history.clear()
            return None
        history.append(results.pose_landmarks.landmark)
        smoothed_list = landmark_pb2.NormalizedLandmarkList()
        for i in range(len(history[0])):
            avg_x = sum(frame[i].x for frame in history) / len(history)
            avg_y = sum(frame[i].y for frame in history) / len(history)
            avg_z = sum(frame[i].z for frame in history) / len(history)
            avg_vis = sum(frame[i].visibility for frame in history) / len(history)
            lm = smoothed_list.landmark.add()
            lm.x, lm.y, lm.z, lm.visibility = avg_x, avg_y, avg_z, avg_vis
        return SmoothedResults(smoothed_list)

    def save_video(self):
        print("Save function needs update for ROI mode.")

if __name__ == "__main__":
    root = tk.Tk()
    app = StuntCVApp(root)
    root.mainloop()
