
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
        self.root.title("Stunt CV - Interactive Tracking")

        self.video_path = None
        self.cap = None
        self.playing = False
        self.video_width = 0
        self.video_height = 0
        self.total_frames = 0

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        self.smoothing_window = 5
        self.pose1_history = deque(maxlen=self.smoothing_window)
        self.pose2_history = deque(maxlen=self.smoothing_window)

        self.create_widgets()

    def create_widgets(self):
        self.video_frame = tk.Frame(self.root)
        self.video_frame.pack()

        self.canvas_left = tk.Canvas(self.video_frame, width=480, height=270, bg='black')
        self.canvas_left.pack(side=tk.LEFT, padx=5, pady=5)
        self.canvas_middle = tk.Canvas(self.video_frame, width=480, height=270, bg='black')
        self.canvas_middle.pack(side=tk.LEFT, padx=5, pady=5)
        self.canvas_right = tk.Canvas(self.video_frame, width=480, height=270, bg='black')
        self.canvas_right.pack(side=tk.LEFT, padx=5, pady=5)

        # --- Slider --- 
        self.slider_frame = tk.Frame(self.root)
        self.slider_frame.pack(fill=tk.X, padx=10, pady=5)
        self.slider = tk.Scale(self.slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_slider_move)
        self.slider.pack(fill=tk.X)

        # --- Buttons and Checkboxes --- 
        self.controls_frame = tk.Frame(self.root)
        self.controls_frame.pack(pady=5)

        self.btn_open = tk.Button(self.controls_frame, text="Open Video", command=self.open_video)
        self.btn_open.pack(side=tk.LEFT, padx=5)
        self.btn_play = tk.Button(self.controls_frame, text="Play/Pause", command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=5)
        self.btn_save = tk.Button(self.controls_frame, text="Save Video", command=self.save_video)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        self.track_base = tk.BooleanVar(value=True)
        self.track_flyer = tk.BooleanVar(value=True)
        self.chk_base = tk.Checkbutton(self.controls_frame, text="Track Base", var=self.track_base)
        self.chk_base.pack(side=tk.LEFT, padx=5)
        self.chk_flyer = tk.Checkbutton(self.controls_frame, text="Track Flyer", var=self.track_flyer)
        self.chk_flyer.pack(side=tk.LEFT, padx=5)

    def on_slider_move(self, value):
        if self.cap and abs(self.cap.get(cv2.CAP_PROP_POS_FRAMES) - int(value)) > 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(value))
            if not self.playing:
                self.update_frame()

    def open_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if self.video_path:
            self.playing = False
            self.pose1_history.clear()
            self.pose2_history.clear()
            self.cap = cv2.VideoCapture(self.video_path)
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.config(to=self.total_frames)
            
            display_height = 360
            display_width = int(self.video_width * (display_height / self.video_height))
            for canvas in [self.canvas_left, self.canvas_middle, self.canvas_right]:
                canvas.config(width=display_width, height=display_height)
            
            self.playing = True
            self.update_frame()

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.update_frame()

    def get_bounding_box(self, landmarks, width, height):
        x_coords = [lm.x * width for lm in landmarks]
        y_coords = [lm.y * height for lm in landmarks]
        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

    def calculate_iou(self, box1, box2):
        x1_inter, y1_inter = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2_inter, y2_inter = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter_area / (box1_area + box2_area - inter_area + 1e-6)

    def find_poses(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        results1 = self.pose.process(frame_rgb)
        results2 = None

        if results1.pose_landmarks:
            frame_rgb_copy = np.copy(frame_rgb)
            box1 = self.get_bounding_box(results1.pose_landmarks.landmark, w, h)
            padding = 10
            cv2.rectangle(frame_rgb_copy, (int(box1[0]-padding), int(box1[1]-padding)), (int(box1[2]+padding), int(box1[3]+padding)), (0,0,0), -1)
            results2 = self.pose.process(frame_rgb_copy)

            if results2.pose_landmarks:
                box2 = self.get_bounding_box(results2.pose_landmarks.landmark, w, h)
                if self.calculate_iou(box1, box2) > 0.1:
                    results2 = None
        return results1, results2

    def smooth_pose(self, results, history):
        if not results or not results.pose_landmarks:
            history.clear()
            return None
        history.append(results.pose_landmarks.landmark)
        smoothed_landmark_list = landmark_pb2.NormalizedLandmarkList()
        for i in range(len(history[0])):
            avg_x = sum(frame[i].x for frame in history) / len(history)
            avg_y = sum(frame[i].y for frame in history) / len(history)
            avg_z = sum(frame[i].z for frame in history) / len(history)
            avg_vis = sum(frame[i].visibility for frame in history) / len(history)
            landmark = smoothed_landmark_list.landmark.add()
            landmark.x, landmark.y, landmark.z, landmark.visibility = avg_x, avg_y, avg_z, avg_vis
        return SmoothedResults(smoothed_landmark_list)

    def classify_and_draw_poses(self, frame, r1, r2, track_base, track_flyer):
        poses = []
        if r1 and r1.pose_landmarks: poses.append(r1.pose_landmarks)
        if r2 and r2.pose_landmarks: poses.append(r2.pose_landmarks)
        if not poses: return

        base_color, flyer_color = (0, 0, 255), (255, 0, 0)

        if len(poses) == 1:
            if track_base or track_flyer: # Draw if either is toggled on
                self.mp_drawing.draw_landmarks(frame, poses[0], self.mp_pose.POSE_CONNECTIONS)
        else:
            avg_y1 = sum(lm.y for lm in poses[0].landmark) / len(poses[0].landmark)
            avg_y2 = sum(lm.y for lm in poses[1].landmark) / len(poses[1].landmark)
            base, flyer = (poses[0], poses[1]) if avg_y1 > avg_y2 else (poses[1], poses[0])

            if track_base:
                self.mp_drawing.draw_landmarks(frame, base, self.mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=base_color, thickness=2, circle_radius=2), connection_drawing_spec=self.mp_drawing.DrawingSpec(color=base_color, thickness=2))
            if track_flyer:
                self.mp_drawing.draw_landmarks(frame, flyer, self.mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=flyer_color, thickness=2, circle_radius=2), connection_drawing_spec=self.mp_drawing.DrawingSpec(color=flyer_color, thickness=2))

    def update_frame(self):
        if self.playing and self.cap.isOpened():
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.slider.set(current_frame)

            ret, frame = self.cap.read()
            if ret:
                original_frame, overlay_frame, pose_only_frame = frame.copy(), frame.copy(), np.zeros_like(frame)

                results1, results2 = self.find_poses(frame)
                smoothed_r1 = self.smooth_pose(results1, self.pose1_history)
                smoothed_r2 = self.smooth_pose(results2, self.pose2_history)

                self.classify_and_draw_poses(overlay_frame, smoothed_r1, smoothed_r2, self.track_base.get(), self.track_flyer.get())
                self.classify_and_draw_poses(pose_only_frame, smoothed_r1, smoothed_r2, self.track_base.get(), self.track_flyer.get())

                display_height = 360
                display_width = int(self.video_width * (display_height / self.video_height))

                for canvas, img_data, photo_attr in [
                    (self.canvas_left, original_frame, "photo_left"),
                    (self.canvas_middle, overlay_frame, "photo_middle"),
                    (self.canvas_right, pose_only_frame, "photo_right")]:
                    img_resized = cv2.resize(img_data, (display_width, display_height))
                    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                    photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
                    setattr(self, photo_attr, photo)
                    canvas.create_image(0, 0, image=photo, anchor=tk.NW)

                self.root.after(15, self.update_frame)
            else:
                self.playing = False
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def save_video(self):
        if self.video_path:
            save_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
            if save_path:
                cap = cv2.VideoCapture(self.video_path)
                width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

                save_p1_hist, save_p2_hist = deque(maxlen=self.smoothing_window), deque(maxlen=self.smoothing_window)
                track_base_on_save, track_flyer_on_save = self.track_base.get(), self.track_flyer.get()

                for _ in range(self.total_frames):
                    ret, frame = cap.read()
                    if not ret: break
                    overlay_frame = frame.copy()
                    r1, r2 = self.find_poses(frame)
                    smoothed_r1, smoothed_r2 = self.smooth_pose(r1, save_p1_hist), self.smooth_pose(r2, save_p2_hist)
                    self.classify_and_draw_poses(overlay_frame, smoothed_r1, smoothed_r2, track_base_on_save, track_flyer_on_save)
                    out.write(overlay_frame)

                cap.release()
                out.release()
                print(f"Video saved to {save_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = StuntCVApp(root)
    root.mainloop()
