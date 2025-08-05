# Stunt CV - Acrobatic Motion Analysis

**Stunt CV** is a desktop application for analyzing acrobatic performances in videos. It uses computer vision to track the movements of performers (a "base" and a "flyer"), providing visual feedback and data export capabilities for detailed analysis.

## Features

- **Dual Person Tracking:** Automatically identifies and tracks two main performers in a video.
- **ROI-Based Tracking:** Manually define regions of interest (ROIs) for precise tracking of the base and flyer.
- **Pose Estimation:** Utilizes MediaPipe Pose to detect and visualize 33 key body landmarks for each performer.
- **Movement Smoothing:** Applies a smoothing filter to the pose data to reduce jitter and provide a clearer representation of the motion.
- **Multi-View Display:** 
    - **Left Pane:** Original, unprocessed video.
    - **Middle Pane:** Video with pose overlays and interactive ROI controls.
    - **Right Pane:** Mocap-only view, showing just the pose skeletons on a black background.
- **Video Export:** Save the processed video with pose overlays or as a mocap-only animation.
- **CSV Data Export:** Export detailed frame-by-frame pose data (x, y, z coordinates and visibility) for both performers to a CSV file for external analysis (e.g., in Excel, Python with Pandas).

## How It Works

The application processes a video file frame by frame. For each frame, it performs the following steps:

1.  **Pose Detection:** It uses one of two methods to find the performers:
    *   **Automatic:** The system scans the frame to find two distinct individuals, automatically classifying them as the base (lower person) and flyer (upper person).
    *   **ROI-Based:** The user can draw and position two boxes (ROIs) on the screen. The system will only search for a person within each designated box.
2.  **Landmark Identification:** Once a person is detected, MediaPipe Pose identifies 33 key body landmarks.
3.  **Data Smoothing:** To stabilize the tracking, the landmark positions are averaged over a small window of recent frames.
4.  **Visualization:** The application displays the original video, the video with the pose skeletons overlaid, and a mocap-only view. The base is colored red, and the flyer is colored blue.

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
- Tkinter (usually included with Python)
- NumPy
- Pillow (PIL)

Install the required packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

1.  **Run the application:**
    ```bash
    python main.py
    ```
2.  **Open Video:** Click the "Open Video" button to load a video file (*.mp4, *.avi).
3.  **Playback:** Use the "Play/Pause" button and the slider to navigate the video.
4.  **Tracking Mode:**
    *   **Automatic (Default):** The application will attempt to find the base and flyer automatically.
    *   **ROI Tracking:** Check the "Enable ROI Tracking" box. Two boxes (red for base, blue for flyer) will appear. 
        *   Click and drag the boxes to position them over the performers.
        *   Click and drag the corners to resize the boxes.
        *   Use the "Track Base" and "Track Flyer" checkboxes to toggle the visibility and tracking for each ROI.
5.  **Save Video:** Click "Save Video" to export the processed footage. You will be prompted to choose between:
    *   **Video with Mocap:** Saves the video with the pose skeletons drawn on top.
    *   **Mocap Only:** Saves a video of just the skeletons on a black background.
6.  **Save CSV:** Click "Save CSV" to export the pose landmark data. The resulting file will contain the frame number, person ID ('base' or 'flyer'), landmark index (0-32), and the x, y, z, and visibility values for each landmark.

## Future Development

- Add more detailed analysis metrics (e.g., joint angles, velocity, acceleration).
- Implement a keyframe system for highlighting specific moments.
- Improve automatic tracking to handle more complex scenarios (e.g., occlusions, fast movements).