# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
pip install -r requirements.txt
python main.py
```

There are no tests or linting configured. The only dependency file is `requirements.txt`.

## Architecture

The entire application lives in a single file: `main.py`. It is a Tkinter desktop GUI app with two classes:

- **`SmoothedResults`** — a thin wrapper that normalizes the return type from both tracking modes, so callers can treat them identically (`.pose_landmarks` attribute).
- **`StuntCVApp`** — the main class. All UI layout, video playback, pose detection, smoothing, stats, and export logic lives here.

### Core Data Flow (per frame)

```
video_loop() → process_and_display_frame()
    → find_poses_by_roi() OR find_poses_auto()   # detect raw poses
    → smooth_pose()                               # temporal averaging via deque
    → draw_classified_poses()                     # render skeletons (base=red, flyer=blue)
    → update_stats_panel()                        # velocity, wobble, alignment, score
    → display_all_frames()                        # render to 3 canvases
```

### Tracking Modes

**Auto mode** (`find_poses_auto`): Runs MediaPipe Pose twice per frame — once on the full frame, then again on a copy with the first person blacked out. Uses IoU matching against the previous frame's bounding boxes to maintain base/flyer identity across frames. Falls back to vertical position (lower = base) on the first frame.

**ROI mode** (`find_poses_by_roi`): Crops to two user-defined bounding boxes, runs MediaPipe on each crop independently, then translates landmarks back to full-frame coordinates via `translate_landmarks()`.

### Save Logic

Saving video or CSV re-processes the entire video from scratch (not from cached frame data). `_find_poses_auto_for_save` is a near-duplicate of `find_poses_auto` that takes/returns explicit state instead of mutating `self`, so the save loop can maintain its own tracking state without corrupting live playback state.

### Stats Calculations

- **Velocity**: Euclidean distance of center-of-mass between consecutive frames, divided by `time_delta`.
- **Wobble**: Standard deviation of horizontal CoM positions over a 15-frame history window.
- **CoM** (`calculate_center_of_mass`): Simple unweighted average of key landmark groups (shoulders/hips, legs, arms). Visibility threshold of 0.5.
- **Stunt Score**: Weighted average of flyer height (40%), flyer horizontal stability (30%), and base/flyer horizontal alignment (30%). All in raw pixel space — not normalized to person size or camera distance.
- **Alignment** (Base): Sum of horizontal deviations between shoulders, hips, and ankles.
- **Plumb Line**: Absolute horizontal distance between base and flyer CoMs.

### Output Directories

- `raw_videos/` — default open dialog directory
- `edited_videos/` — default save dialog directory for video export
- CSV and HTML visualization are saved to a user-chosen path; the HTML file is placed alongside the CSV.
