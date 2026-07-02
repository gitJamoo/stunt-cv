"""Local web UI for Stunt CV — FastAPI backend.

Run from the repo root:

    uvicorn server:app --port 8000
    # or: python server.py

Then open http://127.0.0.1:8000. Serves the frontend from web/, exposes the
analysis pipeline (tracking → metrics → insights) and the DeepSeek chat as a
JSON API, streams raw frames as JPEGs for the canvas player (skeletons are
drawn client-side from the per-frame poses in the analysis result), and
persists finished analyses to analyses/<video>.json so reopening a video
doesn't re-run inference.

The DeepSeek key is read from the DEEPSEEK_API_KEY environment variable
server-side; it is never sent to the browser.

Local tool only: binds to localhost and has no auth — do not expose it to a
network as-is.
"""
import json
import os
import threading
import uuid

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import insights
from pose_data import PoseSmoother, center_of_mass, torso_length, base_alignment_tl
from tracking import PoseTracker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, 'raw_videos')
WEB_DIR = os.path.join(BASE_DIR, 'web')
ANALYSES_DIR = os.path.join(BASE_DIR, 'analyses')
VIDEO_EXTS = ('.mp4', '.avi', '.mov')
FRAME_MAX_W = 960      # frames served to the browser are downscaled to this width
JPEG_QUALITY = 82

app = FastAPI(title="Stunt CV")

# PoseTracker is not thread-safe and ByteTrack state lives inside the model,
# so exactly one analysis runs at a time.
_tracker = None
_tracker_lock = threading.Lock()
_jobs = {}                # job_id -> {'status', 'progress', 'total', 'error', 'result', 'cancel'}
_last_llm_summary = None  # chat context from the most recent completed analysis

# Frame serving keeps one VideoCapture per video; guarded because sequential
# playback requests can interleave with scrubbing
_caps = {}
_caps_lock = threading.Lock()


def _get_tracker():
    global _tracker
    if _tracker is None:
        _tracker = PoseTracker()
    return _tracker


def _video_path(name):
    path = os.path.join(RAW_DIR, os.path.basename(name))
    if not os.path.isfile(path):
        raise HTTPException(404, f"video not found: {name}")
    return path


def _analysis_path(video_name):
    return os.path.join(ANALYSES_DIR, os.path.splitext(os.path.basename(video_name))[0] + '.json')


class AnalyzeRequest(BaseModel):
    video: str
    max_frames: int | None = None   # for quick test runs
    conf: float = 0.4


class ChatRequest(BaseModel):
    message: str
    history: list = []              # prior [{'role', 'content'}] turns
    model: str = "deepseek-chat"
    api_key: str = ""               # optional browser-provided key; falls back to server env


# ----------------------------------------------------------------------
# Videos

@app.get("/api/videos")
def list_videos():
    if not os.path.isdir(RAW_DIR):
        return {"videos": []}
    names = sorted(f for f in os.listdir(RAW_DIR) if f.lower().endswith(VIDEO_EXTS))
    return {"videos": names}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    name = os.path.basename(file.filename or "")
    if not name.lower().endswith(VIDEO_EXTS):
        raise HTTPException(400, f"unsupported file type (want {', '.join(VIDEO_EXTS)})")
    os.makedirs(RAW_DIR, exist_ok=True)
    dest = os.path.join(RAW_DIR, name)
    with open(dest, 'wb') as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    return {"video": name}


@app.get("/api/frame/{video}/{index}")
def get_frame(video: str, index: int):
    """One raw video frame as JPEG, downscaled for the browser player."""
    path = _video_path(video)
    with _caps_lock:
        cap = _caps.get(path)
        if cap is None:
            cap = cv2.VideoCapture(path)
            _caps[path] = cap
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
    if not ret:
        raise HTTPException(404, f"no frame {index}")
    h, w = frame.shape[:2]
    if w > FRAME_MAX_W:
        frame = cv2.resize(frame, (FRAME_MAX_W, int(h * FRAME_MAX_W / w)))
    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise HTTPException(500, "encode failed")
    return Response(content=buf.tobytes(), media_type='image/jpeg')


# ----------------------------------------------------------------------
# Analysis

@app.get("/api/analysis/{video}")
def get_cached_analysis(video: str):
    """Previously computed analysis for this video, if any."""
    path = _analysis_path(video)
    if not os.path.isfile(path):
        raise HTTPException(404, "no cached analysis — run /api/analyze")
    with open(path, encoding='utf-8') as f:
        result = json.load(f)
    global _last_llm_summary
    if result.get('llm_summary'):
        _last_llm_summary = result['llm_summary']
    return result


@app.post("/api/analyze")
def start_analysis(req: AnalyzeRequest):
    video_path = _video_path(req.video)
    if any(j['status'] == 'running' for j in _jobs.values()):
        raise HTTPException(409, "an analysis is already running")

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {'status': 'running', 'progress': 0, 'total': 0,
                     'error': None, 'result': None, 'cancel': False}
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


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    job['cancel'] = True
    return {"status": job['status']}


def _run_analysis_job(job_id, video_path, max_frames, conf):
    global _last_llm_summary
    job = _jobs[job_id]
    try:
        rows, pose_frames, w, h, fps = _process_video(job, video_path, max_frames, conf)
        if job['cancel']:
            job['status'] = 'cancelled'
            return
        df = insights.compute_metrics(rows, w, h, fps)
        text = insights.generate_insights(df, fps)
        llm_summary = insights.summarize_for_llm(df, fps, text)
        _last_llm_summary = llm_summary
        result = {
            'video': os.path.basename(video_path),
            'fps': fps,
            'width': w,
            'height': h,
            'frames': len(rows),
            'insights': text,
            'llm_summary': llm_summary,
            # to_json handles NaN -> null, which raw to_dict does not
            'metrics': json.loads(df.round(4).to_json(orient='records')),
            'poses': pose_frames,
        }
        os.makedirs(ANALYSES_DIR, exist_ok=True)
        with open(_analysis_path(video_path), 'w', encoding='utf-8') as f:
            json.dump(result, f)
        job['result'] = result
        job['status'] = 'done'
    except Exception as e:
        job['status'] = 'error'
        job['error'] = str(e)


def _pose_to_list(lm_list):
    """LandmarkList -> [[x, y, visibility], ...] rounded for compact JSON."""
    return [[round(l.x, 3), round(l.y, 3), round(l.visibility, 2)] for l in lm_list.landmark]


def _process_video(job, video_path, max_frames, conf):
    """Same per-frame pass as the desktop Analyze button, minus the UI.
    Also captures per-frame poses so the browser can draw skeleton overlays:
    each entry is {'b': base_kp|None, 'f': flyer_kp|None, 's': [spotter_kp,...]}
    (base/flyer smoothed to match the desktop rendering, spotters raw)."""
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

        rows, pose_frames = [], []
        for i in range(total):
            if job['cancel']:
                break
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
            pose_frames.append({
                'b': _pose_to_list(base_lm) if base_lm else None,
                'f': _pose_to_list(flyer_lm) if flyer_lm else None,
                's': [_pose_to_list(p) for p in tracker.last_poses
                      if p is not tracker.raw_base and p is not tracker.raw_flyer],
            })
            job['progress'] = i + 1
        cap.release()
        tracker.reset()
    return rows, pose_frames, w, h, fps


# ----------------------------------------------------------------------
# Chat

@app.get("/api/config")
def config():
    """Tells the frontend whether the server already has a DeepSeek key
    (never the key itself)."""
    return {"server_has_key": bool(os.environ.get("DEEPSEEK_API_KEY", "").strip())}


@app.post("/api/chat")
def chat(req: ChatRequest):
    api_key = (req.api_key or os.environ.get("DEEPSEEK_API_KEY", "")).strip()
    if not api_key:
        raise HTTPException(400, "No API key: paste one in the AI Coach panel or set DEEPSEEK_API_KEY on the server")
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


# ----------------------------------------------------------------------
# Frontend

@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, 'index.html'))


app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
