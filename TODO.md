# Stunt CV Roadmap

## 1. Better UI + FastAPI web upgrade

Goal: move from the Tkinter desktop app to a polished local web UI.

- [x] Extract headless modules (`tracking.py` / `stats.py` / `insights.py`) — prerequisite
- [x] FastAPI skeleton: `server.py` + `web/index.html` POC (video list → analyze with progress → insights + charts → AI chat)
- [x] Show the video in the browser with pose overlays (frames served as JPEG, skeletons drawn client-side on canvas from per-frame poses)
- [x] Playback controls on web: play/pause, scrub, frame-step (arrow keys), slow motion, overlay toggle, click-a-chart-to-seek
- [x] Upload videos from the browser
- [x] Cancel button for running analyses
- [x] Persist analyses (`analyses/<video>.json`, auto-loaded on video select)
- [x] Browser API key field (localStorage) with server env fallback
- [ ] Role correction on web: click a person to set base/flyer (needs per-frame track IDs in the API + partial re-analysis from the corrected frame)
- [ ] Progress over WebSocket instead of polling
- [ ] Streamed chat responses (DeepSeek SSE)
- [ ] Smoother playback: pre-encoded annotated MP4 as an alternative to per-frame JPEG fetches for long videos
- [ ] Retire or simplify the Tkinter app once the web UI covers its features

Until the web UI replaces it, worthwhile Tk fixes:
- [ ] Run Analyze/Save on a worker thread (UI currently freezes during processing)
- [ ] Keyboard shortcuts: space = play/pause, arrows = frame step; playback speed control
- [ ] BASE/FLYER labels drawn on the skeletons; hint that right-click reassigns roles
- [ ] Persist settings (sliders, model, last directory) to a settings.json

## 2. Group stunt support (design open — ??)

Current architecture assumes exactly one base + one flyer. Open questions:
which group shapes to support first (2-base + flyer? full pyramid with
backspot?), and what "the stunt" means for scoring when there are multiple
supports.

- [ ] Generalize role model: N performers with labeled roles (`base_l`, `base_r`, `backspot`, `flyer`, ...) bound to track IDs — `tracking.py` state dict becomes role→id map
- [ ] Role assignment UI: right-click menu lists all roles; colors per role
- [ ] Stats generalization: flyer measured against the *support group's* combined center instead of a single base; per-performer wobble
- [ ] Group-specific insights: sync between bases (do they dip/extend together?), catch timing, weight distribution drift
- [ ] Multiple flyers / multiple stunt groups in one frame (probably much later)

## Older ideas (still open)

- Joint angle calculation (knee flexion, shoulder angle) for form feedback
- Keyframe marking (load, extension, catch, dismount) to segment the analysis by phase
- Side-by-side comparison of two attempts of the same skill
