"""Post-run analysis for Stunt CV: builds a per-frame metrics DataFrame from
tracked poses, renders an interactive Plotly report, derives rule-based
coaching insights, and provides a DeepSeek chat client so the user can ask
questions about the data.

All positional metrics are normalized to torso lengths (TL) so they are
camera-distance invariant, matching the live stats panel in main.py.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import requests

WOBBLE_WINDOW = 15          # frames, matches the live stats panel
HOLD_HEIGHT_FRACTION = 0.9  # a frame counts as "the hold" if flyer is >= 90% of peak height


def compute_metrics(rows, frame_w, frame_h, fps):
    """rows: per-frame dicts from the analysis pass —
    {frame, base_com_x, base_com_y, flyer_com_x, flyer_com_y, ref_torso, alignment_tl}
    (com values in pixels, missing people as None). Returns a DataFrame with
    normalized per-frame metrics."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    fps = fps or 30.0
    df['time'] = df['frame'] / fps
    torso = df['ref_torso']

    df['flyer_height'] = 1 - df['flyer_com_y'] / frame_h  # 0 = floor of frame, 1 = top

    for who in ('base', 'flyer'):
        dx = df[f'{who}_com_x'].diff()
        dy = df[f'{who}_com_y'].diff()
        df[f'{who}_velocity'] = np.hypot(dx, dy) / torso * fps  # TL/s
        wobble_x = df[f'{who}_com_x'].rolling(WOBBLE_WINDOW, min_periods=6).std()
        wobble_y = df[f'{who}_com_y'].rolling(WOBBLE_WINDOW, min_periods=6).std()
        df[f'{who}_wobble'] = (wobble_x + wobble_y) / 2 / torso  # TL

    df['plumb'] = (df['base_com_x'] - df['flyer_com_x']).abs() / torso           # TL
    df['plumb_signed'] = (df['flyer_com_x'] - df['base_com_x']) / torso          # + = flyer right of base

    height_score = df['flyer_height'] * 100
    wobble_score = (100 - df['flyer_wobble'] * 500).clip(lower=0)
    plumb_score = (100 - df['plumb'] * 100).clip(lower=0)
    df['stunt_score'] = height_score * 0.4 + wobble_score * 0.3 + plumb_score * 0.3
    return df


def find_hold_phase(df):
    """Returns (start_frame, end_frame) of the longest contiguous run where the
    flyer stays near peak height, or None if there's no usable height data."""
    heights = df['flyer_height'].dropna()
    if heights.empty:
        return None
    threshold = heights.max() * HOLD_HEIGHT_FRACTION
    in_hold = df['flyer_height'] >= threshold
    best, current = None, None
    for frame, flag in zip(df['frame'], in_hold.fillna(False)):
        if flag:
            current = (current[0], frame) if current else (frame, frame)
            if best is None or current[1] - current[0] > best[1] - best[0]:
                best = current
        else:
            current = None
    return best


def generate_insights(df, fps):
    """Rule-based coaching notes derived from the metrics DataFrame."""
    if df.empty or df['flyer_height'].dropna().empty:
        return "Not enough tracked data to generate insights (flyer was rarely detected)."
    fps = fps or 30.0
    lines = []

    peak_idx = df['flyer_height'].idxmax()
    peak = df.loc[peak_idx]
    lines.append(f"Peak flyer height at {peak['time']:.1f}s (frame {int(peak['frame'])}).")

    hold = find_hold_phase(df)
    if hold and hold[1] > hold[0]:
        h0, h1 = hold
        hold_df = df[(df['frame'] >= h0) & (df['frame'] <= h1)]
        duration = (h1 - h0) / fps
        lines.append(f"Hold phase: {duration:.1f}s ({h0}-{h1}), avg stunt score {hold_df['stunt_score'].mean():.0f}.")

        wobble = hold_df['flyer_wobble'].dropna()
        if not wobble.empty:
            lines.append(f"Flyer wobble during hold: avg {wobble.mean():.3f} TL, worst {wobble.max():.3f} TL "
                         f"at {df.loc[wobble.idxmax(), 'time']:.1f}s.")
            if wobble.mean() > 0.06:
                lines.append("Improvement: flyer wobble is high through the hold — focus on the flyer squeezing "
                             "and locking out, and the base keeping arms rigid.")

        plumb = hold_df['plumb'].dropna()
        signed = hold_df['plumb_signed'].dropna()
        if not plumb.empty:
            lines.append(f"Plumb line during hold: avg {plumb.mean():.3f} TL off center, worst {plumb.max():.3f} TL.")
            if abs(signed.mean()) > 0.15:
                side = "right" if signed.mean() > 0 else "left"
                lines.append(f"Improvement: the flyer sits consistently to the base's {side} — "
                             "work on centering at the load so weight stacks over the base.")

        base_wobble = hold_df['base_wobble'].dropna()
        if not base_wobble.empty and base_wobble.mean() > 0.05:
            lines.append("Improvement: the base is drifting under the stunt — likely absorbing the flyer's "
                         "motion with steps instead of a stable stance.")
    else:
        lines.append("No sustained hold phase detected — the flyer never stayed near peak height.")

    rise = df[df['frame'] <= peak['frame']]['flyer_velocity'].dropna()
    if not rise.empty:
        lines.append(f"Fastest movement on the way up: {rise.max():.1f} TL/s.")

    score = df['stunt_score'].dropna()
    if not score.empty:
        lines.append(f"Overall stunt score: avg {score.mean():.0f}, best {score.max():.0f}.")
    return "\n".join(lines)


def build_analysis_html(df, path, insights_text=""):
    """Writes an interactive Plotly report for the metrics DataFrame."""
    charts = [
        ('flyer_height', 'Flyer Height Over Time', 'Height (0-1, frame-relative)'),
        ('stunt_score', 'Stunt Score Over Time', 'Score (0-100)'),
        (['base_velocity', 'flyer_velocity'], 'Velocity Over Time', 'Velocity (TL/s)'),
        (['base_wobble', 'flyer_wobble'], 'Wobble Over Time', 'Wobble (TL)'),
        ('plumb_signed', 'Plumb Line Offset (signed: + = flyer right of base)', 'Offset (TL)'),
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write("<html><head><title>Stunt Analysis</title></head><body>\n")
        f.write("<h1 style='text-align:center;font-family:sans-serif'>Stunt Performance Analysis</h1>\n")
        if insights_text:
            f.write("<div style='max-width:800px;margin:0 auto;font-family:sans-serif;"
                    "background:#f4f4f4;padding:15px;border-radius:8px;white-space:pre-wrap'>"
                    f"{insights_text}</div>\n")
        include_js = 'cdn'
        for y, title, ylabel in charts:
            fig = px.line(df, x='time', y=y, title=title,
                          labels={'time': 'Time (s)', 'value': ylabel, 'variable': 'Performer'})
            f.write(fig.to_html(full_html=False, include_plotlyjs=include_js))
            include_js = False
        f.write("</body></html>")


def summarize_for_llm(df, fps, insights_text, max_samples=60):
    """Compact text context for the chat model: insights plus a downsampled
    metrics timeline. Kept small so it fits comfortably in a prompt."""
    parts = ["AUTOMATED INSIGHTS:", insights_text, "", "METRICS TIMELINE (downsampled):"]
    cols = ['time', 'flyer_height', 'base_velocity', 'flyer_velocity',
            'base_wobble', 'flyer_wobble', 'plumb_signed', 'stunt_score']
    cols = [c for c in cols if c in df.columns]
    step = max(1, len(df) // max_samples)
    sample = df[cols].iloc[::step].round(3)
    parts.append(sample.to_csv(index=False))
    parts.append("Units: TL = torso lengths (camera-invariant). "
                 "plumb_signed: + means the flyer is to the base's right.")
    return "\n".join(parts)


class DeepSeekClient:
    """Minimal OpenAI-compatible client for the DeepSeek API."""
    def __init__(self, api_key, model="deepseek-chat", base_url="https://api.deepseek.com"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip('/')

    def chat(self, messages):
        """messages: list of {'role': ..., 'content': ...}. Returns reply text."""
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
