# SARM multi-stage recording port

## goal

lerobot-panda records SARM datasets w/ per-episode stage annotations (sparse+dense
subtask cols + temporal_proportions json). operator presses kbd key "n" mid-teleop
→ advance to next stage. port to lerobot fork; swap kbd → DualSense button.

## src (panda)

- `lerobot_panda/processor/stage_annotator.py` — `StageAnnotatorProcessorStep`.
  kbd listener (pynput), debounced, per-ep state (stage_starts dict, frame_counter).
  emits `info[stage_index/stage_name/stage_started_this_frame]`.
  `flush_episode_annotation()` → (names, starts, ends). stage0 auto=0.
- `lerobot_panda/rl/gym_manipulator.py`:
  - L705-730: wire annotator into action pipeline when `proc.stage_names` set.
  - L835-929: `_write_stage_annotations_to_dataset` — patches
    meta/episodes/*.parquet w/ sparse_subtask_{names,start_frames,end_frames},
    mirrors to dense_*, writes meta/temporal_proportions_{sparse,dense}.json.
  - L988-997 L1070-1078 L1169-1185 L1227-1234: flush per-ep, patch after finalize.

## diff vs lerobot fork

- no StageAnnotatorProcessorStep.
- no `stage_names` field in HILSerlProcessorConfig.
- GamepadController lacks stage-advance button handling.
- TeleopEvents enum lacks STAGE_ADVANCE.
- control_loop finalize: no annotation patch hook.

## design

### 1. kbd→button

- TeleopEvents.STAGE_ADVANCE = "stage_advance" (enum add).
- GamepadTeleopConfig.stage_advance_button: int = 0. (DualSense face-button,
  currently unused by code; user can remap via JSON.)
- GamepadController: track `_stage_advance_pending`; set on
  JOYBUTTONDOWN==stage_advance_button (no debounce needed — JOYBUTTONDOWN fires once
  per press). `consume_stage_advance() → bool` (read-and-clear).
  ctor takes `stage_advance_button` param.
- GamepadControllerHID: mirror (best-effort; default disabled if button map unclear
  — skip for now, only linux-pygame path needed for PS4/DualSense on this PC).
- GamepadTeleop: ctor passes cfg.stage_advance_button to controller.
  get_teleop_events() adds `TeleopEvents.STAGE_ADVANCE: consume_stage_advance()`.
- ps4_joystick teleop: same treatment IF time. skip unless needed — sim JSONs use
  `gamepad`, not `ps4_joystick`.

### 2. StageAnnotatorProcessorStep

- file: `src/lerobot/processor/stage_annotator.py`.
- `@ProcessorStepRegistry.register("stage_annotator")`.
- dataclass fields: `stage_names: list[str] = []`. NO pynput. NO threading lock.
- state: `_pending_advances: int`, `_frame_counter: int`,
  `_stage_starts: dict[int,int]`. (All init=False, not part of draccus surface.)
- `__call__(transition)`:
  - no-op if stage_names empty.
  - read `info[TeleopEvents.STAGE_ADVANCE]` (bool). true → `_pending_advances += 1`.
  - for each pending: pop next stage idx, record start=_frame_counter, warn if
    already at last.
  - emit `info[stage_index/stage_name/stage_started_this_frame]`.
  - `_frame_counter += 1`.
- `reset()`: clear counters, seed stage 0 @ frame 0.
- `flush_episode_annotation() → (names,starts,ends) | (None,None,None)`.
  same logic as panda.
- register in `src/lerobot/processor/__init__.py` so draccus finds subclass.

### 3. config

- `HILSerlProcessorConfig.stage_names: list[str] | None = None` (envs/configs.py).
- `HILSerlRobotEnvConfig`: no change — reached via `cfg.env.processor.stage_names`.

### 4. gym_manipulator.make_processors

- sim_assembling / gym_hil branch: append StageAnnotatorProcessorStep after
  InterventionActionProcessorStep when `cfg.processor.stage_names` non-empty.
- rc10 branch: same (for future panda-style real-robot use; cheap add).
- real_robot branch: skip (no use case here).

### 5. gym_manipulator.control_loop

- locate annotator: `for step in action_processor.steps: if isinstance(…): …`.
- track `all_ep_stage_annotations: list[tuple|None]`.
- `stage_annotation_resume_offset = dataset.num_episodes` after create (0 unless
  `--dataset.resume=true` is supported — check; if not, always 0).
- episode end (non-rerecord branch): append flush() result; warn when N_recorded
  < N_configured.
- after dataset.finalize(): call `_write_stage_annotations_to_dataset`.

### 6. `_write_stage_annotations_to_dataset`

- verbatim port. path: inside gym_manipulator.py OR util module. keep beside
  control_loop for locality (panda-style).

### 7. JSON

- new: `src/lerobot/rl/sim_assembling_sarm_multistage_record.json`. based on
  existing sim_assembling_sarm_env.json. add:

  ```
  "processor": {
      ...
      "stage_names": ["approach", "align", "insert"],
      "control_time_s": 30.0
  },
  "teleop": {
      "type": "gamepad",
      ...
      "stage_advance_button": 0
  }
  ```

- 3 stages match the assembling task (approach bottom→insert mid→insert cap).
  user can edit; the port preserves the free-form list.

## tests

files under `tests/` (NOT committed per user rule).

### unit: stage_annotator

`tests/processor/test_stage_annotator.py`:
- empty stage_names → passthrough (no info keys).
- reset seeds stage 0 @ frame 0.
- advance once → stage 1 start = current frame_counter.
- multi-advance in one step → processed each with same frame num (matches panda).
- advance past last → warning, no state change.
- flush empty eps → (None,None,None).
- flush mid-episode → starts monotonic, ends[k] = starts[k+1]-1, ends[-1] =
  frame_counter-1.

### unit: gamepad STAGE_ADVANCE

`tests/teleoperators/test_gamepad_stage_advance.py`:
- mock pygame event stream (JOYBUTTONDOWN button=X). verify
  `consume_stage_advance()` returns True once, then False. verify non-matching
  button never triggers.
- GamepadTeleop.get_teleop_events() returns
  `TeleopEvents.STAGE_ADVANCE=True` on the tick of a press.

### integration: pipeline wiring

`tests/rl/test_stage_annotator_wiring.py`:
- build minimal HILSerlRobotEnvConfig w/ `stage_names=["a","b","c"]` +
  mock env + mock teleop. assert `StageAnnotatorProcessorStep` present in
  action_processor.steps (after Intervention step). assert `info[stage_index]`
  propagates through a step.

### smoke: record w/ fake advance

`tests/rl/test_sim_multistage_record_smoke.py`:
- monkeypatch teleop.get_teleop_events to return STAGE_ADVANCE=True at
  frame==F for a known F. run record mode for 1 ep over headless sim. assert:
  - dataset created.
  - meta/temporal_proportions_sparse.json written, values sum≈1.
  - episodes.parquet has sparse_subtask_names of len ≥ 2.
- mark @pytest.mark.slow — needs mujoco EGL.

### manual: live record on DualSense

doc: record 1 ep, press button 0 twice mid-teleop, stop, inspect
`meta/episodes/*.parquet`, temporal_proportions json. out-of-scope for CI.

## risks

- button idx varies by SDL mapping; default 0 may collide on some pads. make
  configurable, print actual name at connect().
- JOYBUTTONDOWN debouncing: pygame fires once per press — holding does NOT
  repeat. no debounce logic needed (vs. panda's kbd listener which debounced
  hold). still guard with pending-counter pattern to decouple producer/consumer
  frames.
- resume_offset: check whether our record path supports resume. likely 0 always
  in v1 — wire the param but hard-code 0 till resume is added.
- temporal_proportions use `frame_counts[name]` keyed by exact stage_names →
  must be stable strings.

## rollout

1. beads plan.
2. implement (processor + teleop + config + wiring + json).
3. unit tests pass.
4. smoke test passes w/ mocked teleop.
5. user runs manual DualSense test. inspects. approves or requests changes.
6. (post-approval only) commit + push.
