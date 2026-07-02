"""Live per-frame stats for the side panel, headless (no Tkinter).

All positional metrics are normalized to torso lengths (TL) so they are
camera-distance invariant. LiveStats keeps the rolling CoM histories; the UI
just displays the formatted strings it returns.
"""
from collections import deque

import numpy as np

from pose_data import center_of_mass, torso_length, base_alignment_tl

WOBBLE_HISTORY = 15  # frames of CoM history for the wobble std-dev


class LiveStats:
    def __init__(self):
        self.last_com_base = None
        self.last_com_flyer = None
        self.base_com_history = deque(maxlen=WOBBLE_HISTORY)
        self.flyer_com_history = deque(maxlen=WOBBLE_HISTORY)

    def reset(self):
        self.last_com_base = None
        self.last_com_flyer = None
        self.base_com_history.clear()
        self.flyer_com_history.clear()

    def update(self, base_results, flyer_results, frame_w, frame_h, time_delta):
        """Advance one frame and return the formatted panel strings:
        {'base_vel', 'flyer_vel', 'base_wobble', 'flyer_wobble',
         'alignment', 'plumb', 'score'}"""
        base_lm = base_results.pose_landmarks if base_results else None
        flyer_lm = flyer_results.pose_landmarks if flyer_results else None

        base_com = center_of_mass(base_lm, frame_w, frame_h)
        flyer_com = center_of_mass(flyer_lm, frame_w, frame_h)
        base_torso = torso_length(base_lm, frame_w, frame_h)
        ref_torso = base_torso or torso_length(flyer_lm, frame_w, frame_h)

        out = {}

        # Velocity in torso lengths / second
        if base_com and self.last_com_base and time_delta > 0 and ref_torso:
            vel = np.linalg.norm(np.array(base_com) - np.array(self.last_com_base)) / time_delta / ref_torso
            out['base_vel'] = f"Base Vel: {vel:.2f} TL/s"
        else:
            out['base_vel'] = "Base Vel: N/A"
        self.last_com_base = base_com

        if flyer_com and self.last_com_flyer and time_delta > 0 and ref_torso:
            vel = np.linalg.norm(np.array(flyer_com) - np.array(self.last_com_flyer)) / time_delta / ref_torso
            out['flyer_vel'] = f"Flyer Vel: {vel:.2f} TL/s"
        else:
            out['flyer_vel'] = "Flyer Vel: N/A"
        self.last_com_flyer = flyer_com

        # 2D wobble: std dev across both axes over the history window
        if base_com:
            self.base_com_history.append(base_com)
        if flyer_com:
            self.flyer_com_history.append(flyer_com)

        base_wobble = self._wobble(self.base_com_history, ref_torso)
        flyer_wobble = self._wobble(self.flyer_com_history, ref_torso)
        out['base_wobble'] = f"Base Wobble: {base_wobble:.3f} TL" if base_wobble is not None else "Base Wobble: N/A"
        out['flyer_wobble'] = f"Flyer Wobble: {flyer_wobble:.3f} TL" if flyer_wobble is not None else "Flyer Wobble: N/A"

        alignment = base_alignment_tl(base_lm, frame_w, base_torso)
        out['alignment'] = f"Alignment: {alignment:.3f} TL" if alignment is not None else "Alignment: N/A"

        # Plumb Line: horizontal CoM offset between base and flyer
        if base_com and flyer_com and ref_torso:
            plumb = abs(base_com[0] - flyer_com[0]) / ref_torso
            out['plumb'] = f"Plumb Line: {plumb:.3f} TL"
        else:
            plumb = None
            out['plumb'] = "Plumb Line: N/A"

        # Stunt Score: flyer height 40%, flyer stability 30%, plumb line 30%
        if base_com and flyer_com and flyer_wobble is not None and plumb is not None:
            height_score = (1 - flyer_com[1] / frame_h) * 100
            wobble_score = max(0, 100 - flyer_wobble * 500)
            plumb_score = max(0, 100 - plumb * 100)
            score = height_score * 0.4 + wobble_score * 0.3 + plumb_score * 0.3
            out['score'] = f"Stunt Score: {score:.1f}"
        else:
            out['score'] = "Stunt Score: N/A"

        return out

    @staticmethod
    def _wobble(history, ref_torso):
        if len(history) > 5 and ref_torso:
            pts = np.array(list(history))
            return float(np.mean(np.std(pts, axis=0)) / ref_torso)
        return None
