"""Local web UI for Stunt CV — FastAPI proof of concept.

Run from the repo root:

    uvicorn server:app --port 8000
    # or: python server.py

Then open http://127.0.0.1:8000. Serves the frontend from web/, exposes the
analysis pipeline (tracking → metrics → insights) and the DeepSeek chat as a
JSON API. The DeepSeek key is read from the DEEPSEEK_API_KEY environment
variable server-side; it is never sent to the browser.

Local tool only: binds to localhost and has no auth — do not expose it to a
network as-is.
"""
import json
import os
import threading
import uuid

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import insights
from pose_data import PoseSmoother, center_of_mass, torso_length, base_alignment_tl
from tracking import PoseTracker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, 'raw_videos')
WEB_DIR = os.path.join(BASE_DIR, 'web')
VIDEO_EXTS = ('.mp4', '.avi', '.mov')

app = FastAPI(title="Stunt CV")

# PoseTracker is not thread-safe and ByteTrack state lives inside the model,
# so exactly one analysis runs at a time.
_tracker = None
_tracker_lock = threading.Lock()
_jobs = {}           # job_id -> {'status', 'progress', 'total', 'error', 'result'}
_last_llm_summary = None  # chat context from the most recent completed analysis


def _get_tracker():
    global _tracker
    if _tracker is None:
        _tracker = PoseTracker()
    return _tracker


class AnalyzeRequest(BaseModel):
    video: str
    max_frames: int | None = None   # for quick test runs
    conf: float = 0.4


class ChatRequest(BaseModel):
    message: str
    history: list = []              # prior [{'role', 'content'}] turns
    model: str = "deepseek-chat"


@app.get("/api/videos")
def list_videos():
    if not os.path.isdir(RAW_DIR):
        return {"videos": []}
    names = sorted(f for f in os.listdir(RAW_DIR) if f.lower().endswith(VIDEO_EXTS))
    return {"videos": names}


@app.post("/api/analyze")
def start_analysis(req: AnalyzeRequest):
    video_path = os.path.join(RAW_DIR, os.path.basename(req.video))
    if not os.path.isfile(video_path):
        raise HTTPException(404, f"video not found: {req.video}")
    if any(j['status'] == 'running' for j in _jobs.values()):
        raise HTTPException(409, "an analysis is already running")

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {'status': 'running', 'progress': 0, 'total': 0,
                     'error': None, 'result': None}
    threading.Thread(target=_run_analysis_job,
                     args=(job_id, video_path, req.max_frames, req.conf),
                     daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


@app.post("/api/chat")
def chat(req: ChatRequest):
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(400, "DEEPSEEK_API_KEY is not set on the server")
    context = _last_llm_summary or "No analysis has been run yet — answer general stunt questions."
    system_prompt = (
        "You are an experienced cheerleading/acro stunt coach. The user analyzed a stunt video "
        "with pose tracking (base and flyer). Answer their questions using the data below when "
        "relevant. Be concrete and concise.\n\n" + context
    )
    messages = ([{"role": "system", "content": system_prompt}]
                + req.history[-12:]
                + [{"role": "user", "content": req.message}])
    try:
        reply = insights.DeepSeekClient(api_key, req.model).chat(messages)
    except Exception as e:
        raise HTTPException(502, f"DeepSeek request failed: {e}")
    return {"reply": reply}


def _run_analysis_job(job_id, video_path, max_frames, conf):
    global _last_llm_summary
    job = _jobs[job_id]
    try:
        rows, w, h, fps = _process_video(job, video_path, max_frames, conf)
        df = insights.compute_metrics(rows, w, h, fps)
        text = insights.generate_insights(df, fps)
        _last_llm_summary = insights.summarize_for_llm(df, fps, text)
        job['result'] = {
            'video': os.path.basename(video_path),
            'fps': fps,
            'insights': text,
            # to_json handles NaN -> null, which raw to_dict does not
            'metrics': json.loads(df.round(4).to_json(orient='records')),
        }
        job['status'] = 'done'
    except Exception as e:
        job['status'] = 'error'
        job['error'] = str(e)


def _process_video(job, video_path, max_frames, conf):
    """Same per-frame pass as the desktop Analyze button, minus the UI."""
    with _tracker_lock:
        tracker = _get_tracker()
        tracker.reset()
        state = PoseTracker.new_state()
        base_sm, flyer_sm = PoseSmoother(8), PoseSmoother(8)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"could not open {video_path}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total = min(total, max_frames)
        job['total'] = total

        rows = []
        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            base, flyer = tracker.detect_auto(frame, state, conf)
            sb, sf = base_sm.smooth(base), flyer_sm.smooth(flyer)

            base_lm = sb.pose_landmarks if sb else None
            flyer_lm = sf.pose_landmarks if sf else None
            base_com = center_of_mass(base_lm, w, h)
            flyer_com = center_of_mass(flyer_lm, w, h)
            base_torso = torso_length(base_lm, w, h)
            rows.append({
                'frame': i,
                'base_com_x':  base_com[0]  if base_com  else None,
                'base_com_y':  base_com[1]  if base_com  else None,
                'flyer_com_x': flyer_com[0] if flyer_com else None,
                'flyer_com_y': flyer_com[1] if flyer_com else None,
                'ref_torso': base_torso or torso_length(flyer_lm, w, h),
                'alignment_tl': base_alignment_tl(base_lm, w, base_torso),
            })
            job['progress'] = i + 1
        cap.release()
        tracker.reset()
    return rows, w, h, fps


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, 'index.html'))


app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
