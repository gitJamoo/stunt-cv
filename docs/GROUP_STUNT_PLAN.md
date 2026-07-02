# Group Stunt Support — Implementation Plan

Target scenario: **one flyer supported by up to 3 people** (e.g., two main
bases + a backspot), plus optional spotters who should stay grey. Today the
entire pipeline assumes exactly one base and one flyer; this document lays
out how each layer generalizes. Written for review before implementation.

---

## 1. Core idea

Keep the mechanism that already works — **roles bound to ByteTrack IDs,
sticky until a track dies or the user reassigns** — but replace the two
hardcoded roles with a role *set*:

- `flyer` — exactly one (multi-flyer pyramids are out of scope for now)
- `support_1 .. support_N` — N is configurable, 1–4 (N=1 reproduces today's
  behavior exactly, which is the backwards-compatibility story)
- everyone else — spotter (grey)

Supports get **positional labels at initialization** (sorted left→right in
frame: `support_1` = leftmost), then the labels stick to their track IDs.
We deliberately do *not* try to auto-classify "backspot vs main base" — from
a single 2D camera that distinction is guesswork; the user can mentally map
"support_2 = backspot" or we add renaming later.

## 2. Layer-by-layer changes

### 2.1 `tracking.py` — role model (the foundation)

**State dict** changes from six flat keys to a roles map:

```python
{
  'roles': {
      'flyer':     {'id': None, 'lm': None, 'miss': 0},
      'support_1': {'id': None, 'lm': None, 'miss': 0},
      ...                        # created per n_supports
  },
  'n_supports': 3,
  'initialized': False,
}
```

`PoseTracker.new_state(n_supports=1)` builds it. All the per-role logic that
currently loops over `('base', 'flyer')` loops over `state['roles']` instead
— miss counting, overlap re-bind (excluding IDs bound to *any* other role),
`invalidate_role_ids`.

**Initialization heuristic** (`_best_stack_group`, replaces `_best_stack_pair`):

1. Flyer = person whose lowest visible point (`pose_bottom_y`) is highest in
   frame. Unchanged.
2. Candidate supports = everyone whose bottom-y is *below* the flyer's.
3. Score each candidate by horizontal distance from the flyer's center
   (`|pose_avg_x(candidate) − pose_avg_x(flyer)|`); take the closest
   `n_supports`. People under the flyer are supports; someone standing two
   meters to the side is a spotter.
4. Sort the chosen supports by x → `support_1..N` left to right.
5. If fewer than `n_supports` people qualify, bind what we have and leave the
   rest unbound — the per-role re-bind machinery will pick people up as they
   enter, and the UI shows which roles are unfilled.

**Detection tuning for crowds:** three bases shoulder-to-shoulder under a
flyer is exactly the case NMS and keypoint models struggle with. Already
mitigated by `iou=0.7` + low-conf ByteTrack feed; two more knobs to expose
if group footage shows dropouts: `imgsz=1280` (small/overlapped people
resolve better; ~2× slower) and ByteTrack `track_buffer` 30→60 (tracks
survive longer occlusions). Both become explicit `PoseTracker` parameters
so they're tunable without code edits.

**API shape:** `detect_auto(frame, state, conf)` still returns per-role
results — now a dict `{'flyer': SmoothedResults|None, 'support_1': ..., ...}`
plus `tracker.raw_roles` (role → raw LandmarkList) for spotter filtering.
The old 2-tuple return goes away; both UIs and the server update in the
same commit.

### 2.2 `pose_data.py` — almost untouched

Geometry is person-agnostic already. One addition:

```python
def group_center_of_mass(com_list):  # mean of the supports' CoMs
```

Later refinement (not in v1): weight by proximity of each support's wrists to
the flyer — a support whose hands aren't under the flyer is contributing
less. Start unweighted; it matches how the pair version treats the base.

### 2.3 Smoothing — one `PoseSmoother` per role

Trivial: a dict of smoothers keyed by role name, created wherever the pair of
smoothers is created today (`main.py`, `server.py`, tests).

### 2.4 `stats.py` / `insights.py` — what the numbers mean with 3 supports

The conceptual change: **the flyer is measured against the support group's
combined center, not a single base.**

Per-frame row (wide format, N known at analysis time):

```
frame, flyer_com_x/y, ref_torso,
support1_com_x/y, support2_com_x/y, support3_com_x/y,
group_com_x/y                       # mean of present supports
```

Metric changes in `compute_metrics`:

| Metric | Pair (today) | Group (new) |
|---|---|---|
| Plumb line | flyer x vs base x | flyer x vs **group center** x |
| Base wobble | one base | per-support wobble + **group-center wobble** |
| Alignment | base's shoulder/hip/ankle stack | same, per support (averaged for the panel) |
| Stunt score | height/flyer-wobble/plumb | same formula, plumb vs group center |

New group-only metrics (the genuinely interesting part):

- **Dip sync** — per frame, the std-dev across supports of their vertical CoM
  velocity. When bases dip and extend together it's ~0; a late base spikes
  it. Report the worst window and an approximate lag ("support_3 moves
  ~80 ms after the others", from cross-correlating velocity series).
- **Load drift** — the flyer's x relative to each support over the hold; a
  monotonic drift toward one support = weight shifting onto one base.
- **Spacing stability** — pairwise distance between supports over time;
  catches bases creeping together under load.

`generate_insights` gains rules on top: out-of-sync dips ("bases are not
dipping together — count it out loud"), load drift direction, one support
wobbling much more than the others. `summarize_for_llm` includes the
per-support columns; the chat system prompt becomes "one flyer supported by
N bases".

With `n_supports=1` every formula degenerates to today's numbers, so the
pair smoke test doubles as the regression test.

### 2.5 Rendering (both UIs)

- Flyer: blue (unchanged). Supports: distinct warm colors
  (red / orange / gold for 1/2/3). Spotters: grey (unchanged).
- Small role labels drawn at each person's head (`F`, `S1`, `S2`, `S3`) —
  this also fixes an existing UX gap in pair mode.
- Web legend and the JS drawing code read a role→color map sent in the
  analysis JSON, so frontend and backend can't drift apart.
- Per-frame pose JSON changes from `{b, f, s}` to
  `{roles: {flyer: kp, support_1: kp, ...}, spotters: [kp...]}` — bump a
  `version` field in the analysis JSON; old cached analyses are simply
  re-run (cheapest migration, they're just caches).

### 2.6 UI for choosing and correcting roles

- **Stunt size selector**: a "Supports: 1–4" dropdown next to Analyze (web)
  and in Tracking Controls (Tk). Default 1 so existing behavior is the
  default. Stored in the analysis JSON so reloading a cached analysis shows
  the right roles.
- **Right-click menu (Tk)** generalizes to: Set as Flyer / Set as Support 1..N
  / Make Spotter (unassign). Assigning an ID that holds another role swaps,
  same as today.
- **CSV export**: `person_id` column keeps `base`/`flyer` when N=1
  (back-compat) and uses `support_1..N`/`flyer` when N>1.

### 2.7 What stays out of scope (v1)

- Multiple flyers / full pyramids — the "exactly one flyer" assumption runs
  deep in the metrics; revisit after 1-flyer groups work.
- Backspot auto-classification.
- Web click-to-assign roles (separate roadmap item; the role→ID model built
  here is its prerequisite).

## 3. Phasing (each phase lands green on the smoke test)

1. **Role model in `tracking.py`** + per-role smoothers + adapt both UIs and
   server to the dict-based API with `n_supports=1` hardwired. Pure
   refactor, zero behavior change, smoke test must pass untouched.
   (~medium; touches every caller)
2. **Group init heuristic + N>1 tracking** + stunt-size selector (web + Tk)
   + colors/labels. Testable on existing spotter footage by setting
   supports=2 and confirming a spotter gets promoted deliberately.
3. **Metrics/insights generalization** — group center, per-support columns,
   dip sync, load drift, spacing; charts for them in the web UI; LLM context.
4. **Polish**: group-specific insight rules tuned on real footage, Tk role
   menu, CSV naming, docs.

## 4. Risks / open questions for review

1. **Test footage**: I don't see any 3-base videos in `raw_videos/` — the
   heuristics and especially the sync metrics need real group footage to
   tune. Can you film or source a couple of clips (ideally: clean prep →
   dip → extension → cradle, camera roughly front-on)?
2. **Support count**: happy with an explicit user-chosen N (default 1),
   rather than auto-detection? Auto ("everyone under the flyer is a
   support") is doable but misfires on close-in spotters; my recommendation
   is explicit N now, auto as a later convenience.
3. **Labels**: are generic `support_1..3` (left→right) labels acceptable for
   v1, or do you want named roles (base_left / base_right / backspot) in the
   UI from the start? Named roles are purely cosmetic over the same model.
4. **Score**: keep the 40/30/30 (height / flyer wobble / plumb-vs-group)
   formula for groups, or should dip sync factor into the score? I'd keep
   the formula stable and report sync separately until we've seen it on
   real footage.
5. **Front-on camera assumption**: plumb/drift metrics assume a roughly
   front-facing view (x = lateral). Fine for pair mode today; groups make
   side-view footage more tempting. V1 keeps the assumption and the docs
   say so.
