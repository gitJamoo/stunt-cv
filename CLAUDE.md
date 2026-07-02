# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
pip install -r requirements.txt
python main.py
```

There are no tests or linting configured. The only dependency file is `requirements.txt`. The YOLOv8 pose model weights (`yolov8m-pose.pt`) download automatically on first run via ultralytics.

## Architecture

The application is two files: `main.py` (all UI, playback, detection, tracking, export) and `insights.py` (post-run metrics DataFrame, rule-based coaching insights, Plotly report, and the DeepSeek chat client — everything headless/testable without Tk). It is a Tkinter desktop GUI app for analyzing cheer/acro stunts — it tracks two performers (a "base" and a "flyer") in video using YOLOv8-pose.

Classes:

- **`Landmark` / `LandmarkList` / `SmoothedResults`** — thin wrappers that mimic the old MediaPipe result shape (`.pose_landmarks.landmark[i].x/.y/.visibility`, normalized 0–1 coordinates). All YOLO output is converted into these via `_yolo_to_landmarks()`, so downstream code (smoothing, drawing, stats, export) is detector-agnostic.
- **`PoseSmoother`** — boxcar-averages landmarks over a rolling window and holds the last smoothed pose through short detection gaps (`max_gap` frames) so brief occlusions don't snap the skeleton.
- **`StuntCVApp`** — everything else: UI layout, video playback, detection, tracking, smoothing, stats, and export.

### Pose Model

YOLOv8-pose (ultralytics) with **COCO 17 keypoints** — not MediaPipe's 33. Keypoint indices are named constants at the top of `main.py` (`KP_L_SHOULDER = 5`, etc.); `YOLO_CONNECTIONS` defines the drawn skeleton. The model file is hardcoded in `__init__` (`yolov8m-pose.pt`; swap to `yolov8n-pose.pt` for speed). There are **two model instances**: `self.yolo` runs auto mode through `.track()` (ByteTrack), and `self.yolo_roi` (lazy) does plain detection for ROI mode — they must stay separate because `.track()` registers tracker callbacks on the model's predictor, and per-crop detections would corrupt tracker state.

### Core Data Flow (per frame)

```
video_loop() → process_and_display_frame()
    → find_poses_by_roi() OR find_poses_auto()   # detect + classify base/flyer
    → smooth_pose()                               # temporal averaging via deque
    → update_stats_panel()                        # velocity, wobble, alignment, score
    → draw_classified_poses()                     # base=red, flyer=blue, spotters=grey
    → display_all_frames()                        # render to 3 canvases
```

### Tracking Modes

**Auto mode** (`find_poses_auto` → `_detect_poses_auto`): One `yolo.track(persist=True)` pass per frame gives every person a persistent ByteTrack ID. **Roles are sticky to track IDs**: on the first frame with 2+ people, `_best_stack_pair` picks flyer (person whose lowest point is highest in frame) and base (most horizontally aligned below); after that, roles just follow their IDs and are never silently reassigned by geometry. The track call uses `conf=0.1` (ByteTrack's second association stage needs low-confidence boxes to hold tracks through occlusion) and `iou=0.7` (so NMS doesn't merge the stacked pair); the UI confidence slider filters everyone *except* the boxes carrying the pair's track IDs. If a role's track dies for ≥5 frames, `_rebind_role` re-binds it to the unassigned track that best overlaps its last known box. "Who's higher" comparisons use `_pose_bottom_y` (ankles/hips max-y), not mean keypoint y — an occluded base with only shoulders visible would otherwise look like the highest person. Role inversion (flyer below base for 15 frames) only sets a passive UI warning (`role_hint_var`); it never auto-swaps.

`_detect_poses_auto` takes a `state` dict (`_new_track_state()`: role→track-ID bindings, last known landmarks, miss counters) and mutates it, so the live path (`self.track_state`) and the save/export loops each keep their own. It also writes `_last_detected_poses` / `_last_detected_ids` / `_raw_base_pose` / `_raw_flyer_pose` on `self` for drawing and click-to-assign. ByteTrack state lives *inside* `self.yolo`, so exports call `_reset_tracker()` before re-processing and `_reset_tracker()` + `_invalidate_role_ids()` after — live playback then recovers its roles via the overlap re-bind.

**ROI mode** (`find_poses_by_roi`): Crops to two user-defined boxes, runs plain detection (`self.yolo_roi`) on each crop (best detection only), then translates landmarks back to full-frame coordinates via `translate_landmarks()`.

**Manual role correction**: "Swap Base/Flyer" button, and right-click on a person in the middle canvas → context menu to bind their track ID to a role (assigning the other role's person swaps the pair). Both clear the smoothers.

### Crop Controls

A single draggable yellow box (`detection_crop`) serves two independent toggles: **Crop Detection Area** (YOLO only runs inside the box) and **Crop Output Video** (saved video is cropped to the box). ROI/crop boxes are stored in display-canvas coordinates and scaled to frame coordinates via `get_scaled_roi()` — relevant when touching the resize logic.

### Save Logic

Saving video or CSV re-processes the entire video from scratch (not from cached frame data), with its own local `_new_track_state()` and `PoseSmoother`s, bracketed by tracker resets (see Auto mode above). Manual role corrections made during live playback are **not** carried into exports — the export re-runs the initialization heuristic. "Save CSV + Viz" additionally generates a Plotly HTML (CoM height, velocity, acceleration over time) next to the CSV.

### Stats Calculations

All stats are normalized by **torso length** (`get_torso_length`, shoulder-to-hip pixel distance, unit "TL") to be camera-distance invariant.

- **Velocity**: CoM displacement between frames / time_delta / torso length (TL/s).
- **Wobble**: Mean std dev of CoM over a 15-frame history, / torso length.
- **CoM** (`calculate_center_of_mass`): Unweighted average of torso/leg/arm keypoints with visibility > 0.5.
- **Alignment** (base): Horizontal deviation across shoulder/hip/ankle stack.
- **Plumb Line**: Horizontal offset between base and flyer CoMs.
- **Stunt Score**: Weighted: flyer height 40%, flyer wobble 30%, plumb line 30%.

### Analysis & AI Chat

The **Analyze** button re-processes the video (same pattern as exports: own track state + smoothers, tracker resets around the pass), collects per-frame CoM/torso rows, and hands them to `insights.compute_metrics()` → a DataFrame of normalized metrics (height, velocity, wobble, plumb, stunt score per frame). It then writes an interactive Plotly report to `edited_videos/<video>_analysis.html` (auto-opened in the browser), generates rule-based coaching notes (`insights.generate_insights`), and opens the chat window.

The **AI Chat** button opens a Toplevel chat box backed by `insights.DeepSeekClient` (plain `requests` against DeepSeek's OpenAI-compatible `/chat/completions`). The API key comes from the window's entry field or `DEEPSEEK_API_KEY`; the model name is editable (default `deepseek-chat`). Each send builds a system prompt from `insights.summarize_for_llm()` — the coaching notes plus a downsampled metrics CSV — and the API call runs on a worker thread, posting back via `root.after`.

### Output Directories

- `raw_videos/` — default open dialog directory
- `edited_videos/` — default save dialog directory for video export
- CSV and HTML visualization are saved to a user-chosen path; the HTML file is placed alongside the CSV.
