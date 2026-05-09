# SARM partial-stage → zero-progress labeling

beads: lerobot-20

## problem

multistage record: N configured stages, operator advances K times via gamepad
STAGE_ADVANCE. current flush:

- stage 0 auto @ frame 0
- stages 1..K entered (K presses)
- `names=[stage_0..stage_K]`, `starts=[0, s1, ..., sK]`,
  `ends[i]=s_{i+1}-1` for i<K, `ends[K]=frame_counter-1`

train-time `find_stage_and_tau`:
- frames in stage K range → stage_idx=K, τ linear 0→1
- last frame → progress = bp[K+1]

for partial ep (K < N-1): last-entered stage **still rises to τ=1** →
terminal progress = bp[K+1] (e.g. ~0.61 for 2-of-3 with even proportions).
partial-failure frames mimic progress. trains SARM to think the stuck stage
"almost succeeded".

user wants: partial final stage frozen @ bp[K] (= "τ=0 within stage K"),
so failure eps plateau at the edge of the LAST COMPLETED stage.

## example (user)

config 4 stages. operator presses next twice. "stage 3 has 0 progress":
- stages 0,1 completed (operator advanced past) → normal τ rise
- stage 2 entered but never advanced → frozen at bp[2]
- stages 3 (never entered) → already omitted (stays bp[2])

## can we do it without touching SARM? YES.

proof: bp[K] = bp[K-1] + 1·(bp[K]-bp[K-1]). achievable by labeling
partial-stage frames as **"end of stage K-1 (τ=1)"** instead of
"stage K (τ rising)".

trick: drop partial last-entered stage from the per-ep annotation, extend
prev stage's end to ep_last_frame. SARM's `find_stage_and_tau` then clips
τ to 1 (compute_tau clamps 0..1) through the extension → progress =
bp[K] flat throughout partial frames.

### edge: operator presses 0 times (stuck in stage 0)
- entered = {0}. drop last → empty.
- flush returns `(None, None, None)` → subtask_names=None →
  `find_stage_and_tau` gives (0, 0.0) everywhere → progress=0. match user intent.

### edge: operator reaches last configured stage (max_entered == N-1)
- not partial. normal flush. last stage τ rises 0→1 as today.
- rationale: if operator reached last stage, treat as completed intent
  (rerecord-if-failed is already available via gamepad RERECORD event).
- alt: additionally gate on env-success signal. **open Q — ask user**.

## implementation (recording-side only)

### change 1: `stage_annotator.py::flush_episode_annotation`

```
entered = sorted(_stage_starts)           # [0, 1, ..., max]
if not entered: return (None,None,None)
max_idx = entered[-1]
final_reached = max_idx == len(stage_names) - 1

if final_reached:
    # status quo
    names, starts, ends = build from all entered
else:
    # drop partial last
    completed = entered[:-1]
    if not completed:
        return (None, None, None)   # 0-press case
    names  = [stage_names[i] for i in completed]
    starts = [_stage_starts[i] for i in completed]
    ends   = [_stage_starts[completed[k+1]] - 1 for k<last]
                    + [_frame_counter - 1]   # extend last completed
    # ALSO: track partial_extend_frames = _frame_counter - _stage_starts[max_idx]
    #       for temporal_proportions correction
```

add a way to surface `partial_extend_frames` alongside the 3-tuple.
options:
- return 4-tuple `(names, starts, ends, partial_extend_frames)`
- OR new getter `extension_frame_count() -> int`, called after flush

pick getter (less churn in callers that only care about names/starts/ends).

### change 2: `gym_manipulator.py::_write_stage_annotations_to_dataset`

today:
```
for names, starts, ends in ep_annotations:
    for n, s, e in zip(...):
        frame_counts[n] += max(0, e - s + 1)
```

patch: last-stage frame_count should EXCLUDE partial_extend_frames.
simplest: pass `ep_extensions: list[int]` alongside `ep_annotations`:
```
for (names, starts, ends), ext in zip(ep_annotations, ep_extensions):
    for i, (n, s, e) in enumerate(zip(...)):
        cnt = max(0, e - s + 1)
        if i == len(names) - 1:
            cnt -= ext           # strip extended partial-stage frames
        frame_counts[n] += cnt
```

temporal_proportions thus reflect real stage durations only; partial-stage
frames contribute zero to any stage's proportion. breakpoints stable.

### change 3: `control_loop` wiring

existing: `all_ep_stage_annotations.append(annot)` after save_episode.
new: also `all_ep_extensions.append(stage_annotator.extension_frame_count())`
   and pass to `_write_stage_annotations_to_dataset`.

### NO changes to:
- sarm_utils.find_stage_and_tau
- sarm_utils.normalize_stage_tau
- modeling_sarm.py / processor_sarm.py / configuration_sarm.py
- SARM training config or dataset schema
- sparse/dense parquet cols — same shape, just different contents per ep

## how to interpret user spec — 2 variants

variant A (simpler, recommended): "partial" = max_entered < N-1.
- reach last stage → always full progress 0→1.

variant B (strict): "partial" = max_entered < N-1 OR no env-success at term.
- even reaching last stage: frozen at bp[N-1] unless env signals success.
- needs passing `episode_succeeded: bool` to flush_episode_annotation,
  source = `transition.info[TeleopEvents.SUCCESS]` OR `terminated and reward>0`.

**ASK USER** which they want. default to A; B is a one-param extension.

## temporal_proportions bias check

with proposal A:
- success eps (all stages entered to N-1): contribute full per-stage counts.
- partial eps: contribute (K completed stages' real counts), partial-stage
  frames contribute 0.
- net: proportions ≈ success-ep-dominant. if dataset is mostly success
  (typical), breakpoints match success-only stats. partial-ep extensions
  don't skew.

no dataset schema changes needed. train code reads sparse_subtask_* +
temporal_proportions_sparse.json same as today.

## validation

1. unit: stage_annotator.py w/ mocked `STAGE_ADVANCE` sequences:
   - 0 presses in N=3 stages → flush returns (None,None,None)
   - 1 press in N=3 → names=[s0], ends[-1]=frame-1 (extended over partial s1)
   - 2 presses in N=3 → names=[s0,s1], normal ends (last-reached stage)
   - extension_frame_count() matches frame_counter - starts[partial_idx]
2. integ: record smoke ep via monkeypatched teleop w/ 1 press in N=3 config,
   inspect patched episodes.parquet + temporal_proportions_sparse.json.
3. train: retrain sim_assembling SARM on mixed full+partial multistage set,
   eval: partial eps should plateau at bp[K] ± noise; success eps still ≥0.9
   terminal. compare to old labels (which gave partial eps bp[K+1]).

## risks

- user's "stage 3 has 0 progress" might mean *literal* 0 (not bp[K]). if so,
  this plan is wrong — need variant C: subtask_names=None for whole ep if
  partial. but that kills the "progress through completed stages" the user
  explicitly asked for. reading their spec favors bp[K]-frozen interpretation.
  **CONFIRM with user before implementation.**

- temporal_proportions shift if dataset is partial-heavy: completed-stage
  counts dominate partial-stage counts → bp[K-1]→bp[K] stays same width,
  but bp[K]→bp[K+1] etc shrink (fewer frames contributing to later stages
  if most eps fail early). may want to weight by successful eps only.
  defer; collect a dataset, inspect.

- if user eventually picks variant B, need success signal plumbed to
  flush_episode_annotation. easy — 1 bool param.

## rollout

1. confirm spec w/ user (variant A vs B; bp[K] vs literal 0).
2. implement stage_annotator change + extension_frame_count getter.
3. implement _write_stage_annotations_to_dataset correction.
4. implement control_loop wiring.
5. unit + smoke tests.
6. record fresh multistage dataset, diff old vs new sparse_* contents.
7. retrain SARM, compare failure plateau.

## files touched

- `src/lerobot/processor/stage_annotator.py` — flush logic + new getter
- `src/lerobot/rl/gym_manipulator.py` — `_write_stage_annotations_to_dataset`
  signature + control_loop flush loop
- zero touch: sarm_utils, modeling_sarm, configs, JSON recording configs
