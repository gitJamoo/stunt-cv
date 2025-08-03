
import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

class StuntCVApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stunt CV")

        self.video_path = None
        self.cap = None
        self.playing = False

        # Increase model complexity for better accuracy
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        self.create_widgets()

    def create_widgets(self):
        self.canvas = tk.Canvas(self.root, width=1280, height=720)
        self.canvas.pack()

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
            self.playing = True
            self.update_frame()

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.update_frame()

    def draw_base_pose(self, frame, results):
        # This function now only draws the primary detected person (the base).
        # The flyer logic is commented out for future use.
        if results and results.pose_landmarks:
            base_color = (255, 0, 0)  # Blue in BGR
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=base_color, thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=base_color, thickness=2, circle_radius=2)
            )

    # def classify_and_draw_poses(self, frame, results1, results2):
    #     poses = []
    #     if results1 and results1.pose_landmarks:
    #         poses.append(results1.pose_landmarks)
    #     if results2 and results2.pose_landmarks:
    #         poses.append(results2.pose_landmarks)
    #
    #     if len(poses) < 2:
    #         for pose_landmarks in poses:
    #             self.mp_drawing.draw_landmarks(frame, pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
    #         return
    #
    #     avg_y = []
    #     for pose_landmarks in poses:
    #         y_coords = [lm.y for lm in pose_landmarks.landmark]
    #         avg_y.append(sum(y_coords) / len(y_coords))
    #
    #     if avg_y[0] < avg_y[1]:
    #         flyer_landmarks = poses[0]
    #         base_landmarks = poses[1]
    #     else:
    #         flyer_landmarks = poses[1]
    #         base_landmarks = poses[0]
    #
    #     flyer_color = (0, 0, 255)  # Red in BGR
    #     base_color = (255, 0, 0)   # Blue in BGR
    #
    #     self.mp_drawing.draw_landmarks(
    #         frame, flyer_landmarks, self.mp_pose.POSE_CONNECTIONS,
    #         landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=flyer_color, thickness=2, circle_radius=2),
    #         connection_drawing_spec=self.mp_drawing.DrawingSpec(color=flyer_color, thickness=2, circle_radius=2)
    #     )
    #     self.mp_drawing.draw_landmarks(
    #         frame, base_landmarks, self.mp_pose.POSE_CONNECTIONS,
    #         landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=base_color, thickness=2, circle_radius=2),
    #         connection_drawing_spec=self.mp_drawing.DrawingSpec(color=base_color, thickness=2, circle_radius=2)
    #     )

    def update_frame(self):
        if self.playing and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)

                # Simplified drawing for just the base
                self.draw_base_pose(frame, results)

                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                self.root.after(10, self.update_frame)
            else:
                self.cap.release()
                self.playing = False

    def save_video(self):
        if self.video_path:
            save_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
            if save_path:
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

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.pose.process(frame_rgb)

                    # Simplified drawing for just the base
                    self.draw_base_pose(frame, results)
                    out.write(frame)

                cap.release()
                out.release()
                print(f"Video saved to {save_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = StuntCVApp(root)
    root.mainloop()
