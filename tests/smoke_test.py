"""Headless smoke test for the tracking → stats → insights pipeline.
No Tkinter required. Run from the repo root:

    python tests/smoke_test.py [video_path]

Uses the first .mp4 in raw_videos/ by default; needs a clip with a base and
a flyer for the role checks to be meaningful.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

import insights
from pose_data import PoseSmoother, center_of_mass, torso_length, base_alignment_tl
from stats import LiveStats
from tracking import PoseTracker

N_FRAMES = 40
CONF = 0.4


def find_video():
    if len(sys.argv) > 1:
        return sys.argv[1]
    candidates = sorted(glob.glob(os.path.join('raw_videos', '*.mp4')))
    assert candidates, "no .mp4 in raw_videos/ — pass a video path as argument"
    return candidates[0]


def main():
    video = find_video()
    print(f"video: {video}")
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f"could not open {video}"
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    tracker = PoseTracker()
    state = PoseTracker.new_state()
    base_sm, flyer_sm = PoseSmoother(8), PoseSmoother(8)
    live = LiveStats()

    # --- Tracking: roles bind and stay sticky ---
    rows = []
    for i in range(N_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        base, flyer = tracker.detect_auto(frame, state, CONF)
        sb, sf = base_sm.smooth(base), flyer_sm.smooth(flyer)
        panel = live.update(sb, sf, w, h, 1.0 / (fps or 30))

        base_lm = sb.pose_landmarks if sb else None
        flyer_lm = sf.pose_landmarks if sf else None
        base_com = center_of_mass(base_lm, w, h)
        flyer_com = center_of_mass(flyer_lm, w, h)
        base_torso = torso_length(base_lm, w, h)
        rows.append({'frame': i,
                     'base_com_x': base_com[0] if base_com else None,
                     'base_com_y': base_com[1] if base_com else None,
                     'flyer_com_x': flyer_com[0] if flyer_com else None,
                     'flyer_com_y': flyer_com[1] if flyer_com else None,
                     'ref_torso': base_torso or torso_length(flyer_lm, w, h),
                     'alignment_tl': base_alignment_tl(base_lm, w, base_torso)})
    assert state['initialized'], "roles never initialized — video may not show 2 people"
    assert state['base_id'] is not None and state['flyer_id'] != state['base_id']
    print(f"tracking OK: base_id={state['base_id']} flyer_id={state['flyer_id']} "
          f"people_last_frame={len(tracker.last_poses)}")
    print(f"live stats last frame: {panel['score']}, {panel['plumb']}")

    # --- Recovery after a tracker reset (what exports do) ---
    old_pair = (state['base_id'], state['flyer_id'])
    tracker.reset()
    PoseTracker.invalidate_role_ids(state)
    for _ in range(3):
        ret, frame = cap.read()
        assert ret
        base, flyer = tracker.detect_auto(frame, state, CONF)
    assert state['base_id'] is not None and state['flyer_id'] is not None, "rebind failed"
    assert state['base_id'] != state['flyer_id']
    print(f"rebind OK: {old_pair} -> ({state['base_id']}, {state['flyer_id']})")
    cap.release()

    # --- Smoother gap bridging ---
    sm = PoseSmoother(8, max_gap=5)
    assert sm.smooth(base) is not None
    for _ in range(5):
        held = sm.smooth(None)
    assert held is not None, "should hold through a 5-frame gap"
    assert sm.smooth(None) is None, "should drop after max_gap"
    print("smoother OK")

    # --- Insights pipeline ---
    df = insights.compute_metrics(rows, w, h, fps)
    assert not df.empty and 'stunt_score' in df.columns
    text = insights.generate_insights(df, fps)
    assert "Peak flyer height" in text
    html_path = '_test_analysis.html'
    insights.build_analysis_html(df, html_path, text)
    assert os.path.getsize(html_path) > 10000
    os.remove(html_path)
    summary = insights.summarize_for_llm(df, fps, text)
    assert 'METRICS TIMELINE' in summary
    print(f"insights OK ({len(df)} frames, summary {len(summary)} chars)")

    test_server_role_editing()
    print("ALL OK")


def test_server_role_editing():
    """Server-side role map editing + recompute from cached tracks (no HTTP)."""
    import server

    # Two synthetic people: track 1 low in frame (base-like), track 2 high
    def person(tid, y):
        kp = [[0.5, y + j * 0.005, 0.9] for j in range(17)]
        return {'id': tid, 'kp': kp}

    tracks = [[person(1, 0.6), person(2, 0.2)] for _ in range(20)]
    role_map = [{'start': 0, 'roles': {'base': 1, 'flyer': 2}}]

    df, text, summary = server._compute_results(tracks, role_map, 1000, 1000, 30.0)
    assert len(df) == 20
    flyer_h_before = df['flyer_height'].mean()

    # Forward reassign at frame 10: flyer -> track 1 (displaces base to track 2)
    server._apply_reassign(role_map, 10, 'flyer', 1, 'forward')
    assert len(role_map) == 2 and role_map[1]['start'] == 10
    assert role_map[1]['roles'] == {'base': 2, 'flyer': 1}, role_map[1]
    assert role_map[0]['roles'] == {'base': 1, 'flyer': 2}   # untouched before frame 10

    df2, _, _ = server._compute_results(tracks, role_map, 1000, 1000, 30.0)
    # After the swap point the flyer is the low person, so mean height drops
    assert df2['flyer_height'].mean() < flyer_h_before

    # 'all' scope rewrites every segment
    server._apply_reassign(role_map, 0, 'flyer', 2, 'all')
    assert all(seg['roles']['flyer'] == 2 for seg in role_map)
    print("server role editing OK")


if __name__ == "__main__":
    main()
