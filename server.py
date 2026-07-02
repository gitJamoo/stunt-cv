"""Local web UI for Stunt CV — FastAPI backend.

Run from the repo root:

    uvicorn server:app --port 8000
    # or: python server.py

Then open http://127.0.0.1:8000.

Analysis architecture: the expensive YOLO/ByteTrack pass produces
role-AGNOSTIC data — every person on every frame with their persistent track
ID and raw keypoints (`tracks`). Who is "base" and who is "flyer" is a
separate, editable mapping (`role_map`: segments of frame → {role: track_id})
recorded from the tracker's automatic assignment and adjustable afterwards
via /api/reassign and /api/swap_roles. Corrections relabel and recompute
metrics/insights from the cached keypoints in milliseconds — inference never
re-runs. Role names are plain strings so the group-stunt generalization
(docs/GROUP_STUNT_PLAN.md) extends ROLES rather than reworking the schema.

Finished analyses persist to analyses/<video>.json (schema ANALYSIS_VERSION;
older caches are rejected and simply re-run).

The DeepSeek key comes from the browser request or the DEEPSEEK_API_KEY
environment variable server-side; it is never sent to the browser.

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
from pose_data import (Landmark, LandmarkList, PoseSmoother, SmoothedResults,
                       base_alignment_tl, center_of_mass, torso_length)
from tracking import PoseTracker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, 'raw_videos')
WEB_DIR = os.path.join(BASE_DIR, 'web')
ANALYSES_DIR = os.path.join(BASE_DIR, 'analyses')
VIDEO_EXTS = ('.mp4', '.avi', '.mov')
FRAME_MAX_W = 960      # frames served to the browser are downscaled to this width
JPEG_QUALITY = 82

ANALYSIS_VERSION = 2
ROLES = ('base', 'flyer')   # group stunts will extend this list
SMOOTH_WINDOW = 8

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


def _load_analysis(video_name):
    path = _analysis_path(video_name)
    if not os.path.isfile(path):
        raise HTTPException(404, "no cached analysis — run /api/analyze")
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if data.get('version') != ANALYSIS_VERSION:
        raise HTTPException(404, "cached analysis uses an old format — re-run /api/analyze")
    return data


class AnalyzeRequest(BaseModel):
    video: str
    max_frames: int | None = None   # for quick test runs
    conf: float = 0.4


class ReassignRequest(BaseModel):
    video: str
    role: str                       # one of ROLES
    track_id: int
    frame: int = 0
    scope: str = 'forward'          # 'forward' (from `frame` on) or 'all'


class SwapRequest(BaseModel):
    video: str


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
    data = _load_analysis(video)
    global _last_llm_summary
    if data.get('llm_summary'):
        _last_llm_summary = data['llm_summary']
    return data


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


@app.post("/api/reassign")
def reassign(req: ReassignRequest):
    """Bind a role to a track ID from a given frame on (or everywhere) and
    recompute metrics/insights from the cached tracks — no re-inference."""
    if req.role not in ROLES:
        raise HTTPException(400, f"unknown role {req.role!r} (want one of {ROLES})")
    data = _load_analysis(req.video)
    if req.scope not in ('forward', 'all'):
        raise HTTPException(400, "scope must be 'forward' or 'all'")
    _apply_reassign(data['role_map'], req.frame, req.role, req.track_id, req.scope)
    return _rebuild_analysis(data)


@app.post("/api/swap_roles")
def swap_roles(req: SwapRequest):
    """Exchange base and flyer along the whole video."""
    data = _load_analysis(req.video)
    for seg in data['role_map']:
        r = seg['roles']
        r['base'], r['flyer'] = r.get('flyer'), r.get('base')
    return _rebuild_analysis(data)


def _apply_reassign(role_map, frame, role, track_id, scope):
    """Mutates role_map. Swap semantics: if the target track already holds
    another role in a segment, the displaced role inherits this role's old
    track (matching the desktop right-click behavior)."""
    def set_role(roles):
        for other, tid in list(roles.items()):
            if other != role and tid == track_id:
                roles[other] = roles.get(role)
        roles[role] = track_id

    if not role_map:
        role_map.append({'start': 0, 'roles': {r: None for r in ROLES}})
    if scope == 'all':
        for seg in role_map:
            set_role(seg['roles'])
        return
    # forward: split the covering segment at `frame`, apply from there on
    idx = 0
    for j, seg in enumerate(role_map):
        if seg['start'] <= frame:
            idx = j
        else:
            break
    if role_map[idx]['start'] < frame:
        role_map.insert(idx + 1, {'start': frame, 'roles': dict(role_map[idx]['roles'])})
        idx += 1
    for seg in role_map[idx:]:
        set_role(seg['roles'])


def _run_analysis_job(job_id, video_path, max_frames, conf):
    job = _jobs[job_id]
    try:
        tracks, role_map, w, h, fps = _process_video(job, video_path, max_frames, conf)
        if job['cancel']:
            job['status'] = 'cancelled'
            return
        data = {
            'version': ANALYSIS_VERSION,
            'video': os.path.basename(video_path),
            'fps': fps, 'width': w, 'height': h, 'frames': len(tracks),
            'roles': list(ROLES),
            'role_map': role_map,
            'tracks': tracks,
        }
        job['result'] = _rebuild_analysis(data)
        job['status'] = 'done'
    except Exception as e:
        job['status'] = 'error'
        job['error'] = str(e)


def _rebuild_analysis(data):
    """Compute metrics/insights from cached tracks + role_map, persist, and
    return the full analysis dict. Single source of truth for both the
    initial analysis and every subsequent role correction."""
    global _last_llm_summary
    df, text, llm_summary = _compute_results(
        data['tracks'], data['role_map'], data['width'], data['height'], data['fps'])
    data['insights'] = text
    data['llm_summary'] = llm_summary
    data['metrics'] = json.loads(df.round(4).to_json(orient='records'))
    _last_llm_summary = llm_summary
    os.makedirs(ANALYSES_DIR, exist_ok=True)
    with open(_analysis_path(data['video']), 'w', encoding='utf-8') as f:
        json.dump(data, f)
    return data


def _pose_to_kp(lm_list):
    """LandmarkList -> [[x, y, visibility], ...] rounded for compact JSON."""
    return [[round(l.x, 4), round(l.y, 4), round(l.visibility, 2)] for l in lm_list.landmark]


def _kp_to_landmarks(kp):
    return LandmarkList([Landmark(x=p[0], y=p[1], visibility=p[2]) for p in kp])


def _roles_at(role_map, seg_idx, frame):
    """Advance the segment cursor to cover `frame`; returns (roles, seg_idx)."""
    while seg_idx + 1 < len(role_map) and role_map[seg_idx + 1]['start'] <= frame:
        seg_idx += 1
    roles = role_map[seg_idx]['roles'] if role_map else {}
    return roles, seg_idx


def _compute_results(tracks, role_map, w, h, fps):
    """Metrics DataFrame + insights from cached role-agnostic tracks."""
    smoothers = {r: PoseSmoother(SMOOTH_WINDOW) for r in ROLES}
    rows = []
    seg_idx = 0
    for i, people in enumerate(tracks):
        roles, seg_idx = _roles_at(role_map, seg_idx, i)
        by_id = {p['id']: p for p in people}
        lm = {}
        for r in ROLES:
            person = by_id.get(roles.get(r))
            raw = SmoothedResults(_kp_to_landmarks(person['kp'])) if person else None
            smoothed = smoothers[r].smooth(raw)
            lm[r] = smoothed.pose_landmarks if smoothed else None

        base_com = center_of_mass(lm['base'], w, h)
        flyer_com = center_of_mass(lm['flyer'], w, h)
        base_torso = torso_length(lm['base'], w, h)
        rows.append({
            'frame': i,
            'base_com_x':  base_com[0]  if base_com  else None,
            'base_com_y':  base_com[1]  if base_com  else None,
            'flyer_com_x': flyer_com[0] if flyer_com else None,
            'flyer_com_y': flyer_com[1] if flyer_com else None,
            'ref_torso': base_torso or torso_length(lm['flyer'], w, h),
            'alignment_tl': base_alignment_tl(lm['base'], w, base_torso),
        })
    df = insights.compute_metrics(rows, w, h, fps)
    text = insights.generate_insights(df, fps)
    llm_summary = insights.summarize_for_llm(df, fps, text)
    return df, text, llm_summary


def _process_video(job, video_path, max_frames, conf):
    """The expensive pass: YOLO + ByteTrack over every frame. Produces
    role-agnostic `tracks` (all people, raw keypoints, persistent IDs) and
    the tracker's automatic `role_map` as the starting role assignment."""
    with _tracker_lock:
        tracker = _get_tracker()
        tracker.reset()
        state = PoseTracker.new_state()

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

        tracks, role_map = [], []
        for i in range(total):
            if job['cancel']:
                break
            ret, frame = cap.read()
            if not ret:
                break
            tracker.detect_auto(frame, state, conf)
            tracks.append([{'id': tid, 'kp': _pose_to_kp(pose)}
                           for tid, pose in zip(tracker.last_ids, tracker.last_poses)])
            current = {r: state[f'{r}_id'] for r in ROLES}
            if not role_map or role_map[-1]['roles'] != current:
                role_map.append({'start': i, 'roles': current})
            job['progress'] = i + 1
        cap.release()
        tracker.reset()
    return tracks, role_map, w, h, fps


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
