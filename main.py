import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

class StuntCVApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stunt CV - Triple View")

        self.video_path = None
        self.cap = None
        self.playing = False
        self.video_width = 0
        self.video_height = 0

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
        self.mp_drawing = mp.solutions.drawing_utils

        self.create_widgets()

    def create_widgets(self):
        # Main frame for the video canvases
        self.video_frame = tk.Frame(self.root)
        self.video_frame.pack()

        # Canvas for original video (left)
        self.canvas_left = tk.Canvas(self.video_frame, width=480, height=270, bg='black')
        self.canvas_left.pack(side=tk.LEFT, padx=5, pady=5)

        # Canvas for video with pose overlay (middle)
        self.canvas_middle = tk.Canvas(self.video_frame, width=480, height=270, bg='black')
        self.canvas_middle.pack(side=tk.LEFT, padx=5, pady=5)

        # Canvas for pose only (right)
        self.canvas_right = tk.Canvas(self.video_frame, width=480, height=270, bg='black')
        self.canvas_right.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_frame = tk.Frame(self.root)
        self.btn_frame.pack(pady=10)

        self.btn_open = tk.Button(self.btn_frame, text="Open Video", command=self.open_video)
        self.btn_open.pack(side=tk.LEFT, padx=5)

        self.btn_play = tk.Button(self.btn_frame, text="Play/Pause", command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=5)

        self.btn_save = tk.Button(self.btn_frame, text="Save Video", command=self.save_video)
        self.btn_save.pack(side=tk.LEFT, padx=5)

    def open_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Adjust canvas sizes to fit video aspect ratio
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

    def classify_and_draw_base(self, frame, results1, results2):
        poses = []
        if results1 and results1.pose_landmarks:
            poses.append(results1.pose_landmarks)
        if results2 and results2.pose_landmarks:
            poses.append(results2.pose_landmarks)

        if not poses:
            return

        if len(poses) == 1:
            base_landmarks = poses[0]
        else:
            avg_y = []
            for pose_landmarks in poses:
                y_coords = [lm.y for lm in pose_landmarks.landmark]
                avg_y.append(sum(y_coords) / len(y_coords))

            if avg_y[0] > avg_y[1]:
                base_landmarks = poses[0]
            else:
                base_landmarks = poses[1]

        base_color = (255, 0, 0)   # Blue in BGR
        self.mp_drawing.draw_landmarks(
            frame, base_landmarks, self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=base_color, thickness=2, circle_radius=2),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=base_color, thickness=2, circle_radius=2)
        )

    def update_frame(self):
        if self.playing and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 1. Original Video (Left)
                original_frame = frame.copy()

                # 2. Pose Overlay Video (Middle)
                overlay_frame = frame.copy()
                
                # 3. Pose-only Video (Right)
                pose_only_frame = np.zeros_like(frame)

                # Process frame for pose
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results1 = self.pose.process(frame_rgb)
                results2 = None

                if results1.pose_landmarks:
                    frame_rgb_copy = np.copy(frame_rgb)
                    h, w, _ = frame_rgb_copy.shape
                    landmarks = results1.pose_landmarks.landmark
                    x_min = min([lm.x for lm in landmarks]) * w
                    x_max = max([lm.x for lm in landmarks]) * w
                    y_min = min([lm.y for lm in landmarks]) * h
                    y_max = max([lm.y for lm in landmarks]) * h
                    cv2.rectangle(frame_rgb_copy, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 0), -1)
                    results2 = self.pose.process(frame_rgb_copy)

                # Draw poses on overlay and pose-only frames
                self.classify_and_draw_base(overlay_frame, results1, results2)
                self.classify_and_draw_base(pose_only_frame, results1, results2)

                # --- Display Frames ---
                display_height = 360
                display_width = int(self.video_width * (display_height / self.video_height))

                # Left canvas
                img_left = cv2.resize(original_frame, (display_width, display_height))
                img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
                self.photo_left = ImageTk.PhotoImage(image=Image.fromarray(img_left))
                self.canvas_left.create_image(0, 0, image=self.photo_left, anchor=tk.NW)

                # Middle canvas
                img_middle = cv2.resize(overlay_frame, (display_width, display_height))
                img_middle = cv2.cvtColor(img_middle, cv2.COLOR_BGR2RGB)
                self.photo_middle = ImageTk.PhotoImage(image=Image.fromarray(img_middle))
                self.canvas_middle.create_image(0, 0, image=self.photo_middle, anchor=tk.NW)

                # Right canvas
                img_right = cv2.resize(pose_only_frame, (display_width, display_height))
                img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
                self.photo_right = ImageTk.PhotoImage(image=Image.fromarray(img_right))
                self.canvas_right.create_image(0, 0, image=self.photo_right, anchor=tk.NW)

                self.root.after(10, self.update_frame)
            else:
                self.cap.release()
                self.playing = False

    def save_video(self):
        if self.video_path:
            save_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
            if save_path:
                # For now, saving the middle view (overlay). 
                # This could be changed to save all three or let the user choose.
                cap = cv2.VideoCapture(self.video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    overlay_frame = frame.copy()
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results1 = self.pose.process(frame_rgb)
                    results2 = None

                    if results1.pose_landmarks:
                        frame_rgb_copy = np.copy(frame_rgb)
                        h, w, _ = frame_rgb_copy.shape
                        landmarks = results1.pose_landmarks.landmark
                        x_min = min([lm.x for lm in landmarks]) * w
                        x_max = max([lm.x for lm in landmarks]) * w
                        y_min = min([lm.y for lm in landmarks]) * h
                        y_max = max([lm.y for lm in landmarks]) * h
                        cv2.rectangle(frame_rgb_copy, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 0), -1)
                        results2 = self.pose.process(frame_rgb_copy)

                    self.classify_and_draw_base(overlay_frame, results1, results2)
                    out.write(overlay_frame)

                cap.release()
                out.release()
                print(f"Video saved to {save_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = StuntCVApp(root)
    root.mainloop()