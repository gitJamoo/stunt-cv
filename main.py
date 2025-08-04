
import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog, messagebox
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

        # Smoothing Deques
        self.smoothing_window = 5
        self.base_history = deque(maxlen=self.smoothing_window)
        self.flyer_history = deque(maxlen=self.smoothing_window)

        # UI State Variables
        self.roi_tracking_enabled = tk.BooleanVar(value=False)
        self.track_base = tk.BooleanVar(value=True)
        self.track_flyer = tk.BooleanVar(value=True)

        # ROI State
        self.base_roi =  {"x": 30,  "y": 110, "w": 150, "h": 180, "name": "base"}
        self.flyer_roi = {"x": 30, "y": 15,  "w": 150, "h": 180, "name": "flyer"}
        self.active_roi = None
        self.drag_info = {}
        self.handle_size = 8

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
        
        self.chk_roi = tk.Checkbutton(self.controls_frame, text="Enable ROI Tracking", var=self.roi_tracking_enabled, command=self.on_roi_toggle)
        self.chk_roi.pack(side=tk.LEFT, padx=10)
        self.chk_base = tk.Checkbutton(self.controls_frame, text="Track Base", var=self.track_base, command=self.on_visibility_toggle)
        self.chk_base.pack(side=tk.LEFT, padx=5)
        self.chk_flyer = tk.Checkbutton(self.controls_frame, text="Track Flyer", var=self.track_flyer, command=self.on_visibility_toggle)
        self.chk_flyer.pack(side=tk.LEFT, padx=5)

    def open_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if not self.video_path: return

        self.playing = False
        self.base_history.clear(); self.flyer_history.clear()
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
            ret, frame = self.cap.read()
            if ret:
                self.current_frame_data = frame
                self.process_and_display_frame()
                self.root.after(15, self.video_loop)
            else:
                self.playing = False
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if ret: self.current_frame_data = frame
                self.process_and_display_frame()

    def process_and_display_frame(self):
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

    def get_bounding_box(self, landmarks, w, h):
        x_coords = [lm.x * w for lm in landmarks]; y_coords = [lm.y * h for lm in landmarks]
        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

    def calculate_iou(self, box1, box2):
        x1_inter, y1_inter = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2_inter, y2_inter = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1]); box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter_area / (box1_area + box2_area - inter_area + 1e-6)

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
        save_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")], title="Save Video As")
        if not save_path: return
        cap = cv2.VideoCapture(self.video_path)
        width, height, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        save_base_hist, save_flyer_hist = deque(maxlen=self.smoothing_window), deque(maxlen=self.smoothing_window)
        roi_enabled = self.roi_tracking_enabled.get()
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_dialog = tk.Toplevel(self.root); progress_dialog.title("Saving...")
        progress_label = tk.Label(progress_dialog, text=f"Processing frame 0/{num_frames}"); progress_label.pack(padx=20, pady=10)
        progress_dialog.geometry(f"+{self.root.winfo_x()+150}+{self.root.winfo_y()+150}"); self.root.update_idletasks()
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret: break
            output_frame = np.zeros_like(frame) if mocap_only else frame.copy()
            if roi_enabled: base_results, flyer_results = self.find_poses_by_roi(frame)
            else: base_results, flyer_results = self.find_poses_auto(frame)
            smoothed_base, smoothed_flyer = self.smooth_pose(base_results, save_base_hist), self.smooth_pose(flyer_results, save_flyer_hist)
            self.draw_classified_poses(output_frame, smoothed_base, smoothed_flyer)
            out.write(output_frame)
            progress_label.config(text=f"Processing frame {i+1}/{num_frames}"); self.root.update_idletasks()
        cap.release(); out.release(); progress_dialog.destroy()
        messagebox.showinfo("Save Complete", f"Video saved to {save_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = StuntCVApp(root)
    root.mainloop()
