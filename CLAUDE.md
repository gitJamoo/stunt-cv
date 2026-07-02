# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
pip install -r requirements.txt
python main.py                      # Tkinter desktop app
python -m uvicorn server:app        # local web UI (POC) at http://127.0.0.1:8000
```

Linting is not configured. The only dependency file is `requirements.txt`. The YOLOv8 pose model weights (`yolov8m-pose.pt`) download automatically on first run via ultralytics.

**Test:** `python tests/smoke_test.py [video]` — headless (no Tk) end-to-end check of tracking, role rebinding, smoothing, and the insights pipeline against a real video (defaults to the first .mp4 in `raw_videos/`). Run it after touching tracking or analysis code.

## Architecture

A Tkinter desktop app for analyzing cheer/acro stunts — it tracks two performers (a "base" and a "flyer") in video using YOLOv8-pose. Everything except `main.py` is headless (no Tkinter) and importable in tests:

- **`pose_data.py`** — keypoint constants (`KP_*`, `YOLO_CONNECTIONS`), data types, and pure geometry: `Landmark`/`LandmarkList`/`SmoothedResults` (thin wrappers that mimic the old MediaPipe result shape — `.pose_landmarks.landmark[i].x/.y/.visibility`, normalized 0–1 coordinates — so downstream code is detector-agnostic), `PoseSmoother` (boxcar smoothing that holds the last pose through short detection gaps), CoM, torso length, IoU, bounding boxes, `pose_bottom_y`, `base_alignment_tl`.
- **`tracking.py`** — `PoseTracker`: owns both YOLO model handles and all detection/role logic (`detect_auto`, `detect_in_roi`, `reset`, `new_state`, `invalidate_role_ids`). Role state is a plain dict owned by the *caller*, so live playback and export passes keep independent states. Exposes `last_poses`/`last_ids`/`raw_base`/`raw_flyer` for drawing and click-to-assign.
- **`stats.py`** — `LiveStats`: per-frame stats panel metrics; keeps the rolling CoM histories and returns formatted strings for the panel.
- **`insights.py`** — post-run metrics DataFrame, rule-based coaching insights, Plotly report, LLM context summary, and `DeepSeekClient`.
- **`main.py`** — `StuntCVApp`: Tkinter UI, playback loop, ROI/crop dragging, export dialogs, analysis orchestration, and the AI chat window. Should contain only UI concerns and glue — put new logic in the headless modules.
- **`server.py` + `web/index.html`** — FastAPI local web UI (the planned successor to the Tk app — see `TODO.md`). Key design: the YOLO pass produces **role-agnostic `tracks`** (every person, every frame, raw keypoints + ByteTrack ID); base/flyer is a separate editable **`role_map`** (segments of frame → {role: track_id}). Role corrections (`/api/reassign`, `/api/swap_roles`) relabel + recompute metrics from the cached tracks via `_rebuild_analysis` — inference never re-runs. Role names are plain strings in `ROLES` so group stunts extend the list, not the schema. Other endpoints: `/api/videos`, `/api/upload`, `/api/analyze` (background job, one at a time — `PoseTracker` is not thread-safe; cancellable), `/api/jobs/{id}`, `/api/analysis/{video}` (cache from `analyses/<video>.json`, rejected if `version != ANALYSIS_VERSION`), `/api/frame/{video}/{i}` (JPEG frames; skeletons drawn client-side — the JS `CONN` list must match `pose_data.YOLO_CONNECTIONS`), `/api/chat` (DeepSeek proxy; browser key or `DEEPSEEK_API_KEY` fallback). Binds localhost, no auth.

### Pose Model

YOLOv8-pose (ultralytics) with **COCO 17 keypoints** — not MediaPipe's 33. Keypoint indices are named constants in `pose_data.py` (`KP_L_SHOULDER = 5`, etc.); `YOLO_CONNECTIONS` defines the drawn skeleton. The model path is set where `PoseTracker` is constructed in `main.py` (`yolov8m-pose.pt`; swap to `yolov8n-pose.pt` for speed). `PoseTracker` keeps **two model instances**: the tracking model runs auto mode through `.track()` (ByteTrack), and a lazy second instance does plain detection for ROI mode — they must stay separate because `.track()` registers tracker callbacks on the model's predictor, and per-crop detections would corrupt tracker state.

### Core Data Flow (per frame)

```
video_loop() → process_and_display_frame()
    → find_poses_by_roi() OR find_poses_auto()   # → PoseTracker.detect_in_roi / detect_auto
    → PoseSmoother.smooth()                       # temporal averaging + gap hold
    → update_stats_panel()                        # → LiveStats.update
    → draw_classified_poses()                     # base=red, flyer=blue, spotters=grey
    → display_all_frames()                        # render to 3 canvases
```

### Tracking Modes

**Auto mode** (`PoseTracker.detect_auto`): One `yolo.track(persist=True)` pass per frame gives every person a persistent ByteTrack ID. **Roles are sticky to track IDs**: on the first frame with 2+ people, `_best_stack_pair` picks flyer (person whose lowest point is highest in frame) and base (most horizontally aligned below); after that, roles just follow their IDs and are never silently reassigned by geometry. The track call uses `conf=0.1` (ByteTrack's second association stage needs low-confidence boxes to hold tracks through occlusion) and `iou=0.7` (so NMS doesn't merge the stacked pair); the UI confidence threshold filters everyone *except* the boxes carrying the pair's track IDs. If a role's track dies for ≥5 frames, `_rebind_role` re-binds it to the unassigned track that best overlaps its last known box. "Who's higher" comparisons use `pose_bottom_y` (ankles/hips max-y), not mean keypoint y — an occluded base with only shoulders visible would otherwise look like the highest person. Role inversion (flyer below base for 15 frames) only sets a passive UI warning (`role_hint_var`, in `main.py`); it never auto-swaps.

ByteTrack state lives *inside* the YOLO model, so export/analysis passes call `tracker.reset()` before re-processing and `_recover_live_tracking()` (reset + `invalidate_role_ids`) after — live playback then recovers its roles via the overlap re-bind.

**ROI mode** (`PoseTracker.detect_in_roi`): Crops to two user-defined boxes, runs plain detection on each crop (best detection only), then translates landmarks back to full-frame coordinates.

**Manual role correction**: "Swap Base/Flyer" button, and right-click on a person in the middle canvas → context menu to bind their track ID to a role (assigning the other role's person swaps the pair). Both clear the smoothers.

### Crop Controls

A single draggable yellow box (`detection_crop`) serves two independent toggles: **Crop Detection Area** (YOLO only runs inside the box) and **Crop Output Video** (saved video is cropped to the box). ROI/crop boxes are stored in display-canvas coordinates and scaled to frame coordinates via `get_scaled_roi()` — relevant when touching the resize logic.

### Save Logic

Saving video or CSV re-processes the entire video from scratch (not from cached frame data), with its own local `PoseTracker.new_state()` and `PoseSmoother`s, bracketed by tracker resets (see Auto mode above). Manual role corrections made during live playback are **not** carried into exports — the export re-runs the initialization heuristic. "Save CSV + Viz" additionally generates a Plotly HTML (CoM height, velocity, acceleration over time) next to the CSV.

### Stats Calculations

All stats are normalized by **torso length** (`pose_data.torso_length`, shoulder-to-hip pixel distance, unit "TL") to be camera-distance invariant.

- **Velocity**: CoM displacement between frames / time_delta / torso length (TL/s).
- **Wobble**: Mean std dev of CoM over a 15-frame history, / torso length.
- **CoM** (`pose_data.center_of_mass`): Unweighted average of torso/leg/arm keypoints with visibility > 0.5.
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
