# PR #3827 — `feat(unitree_g1): pySONIC wbc` — Review Comments

Reviewer: **CarolinePascal** (Collaborator)
Branch: `feat/unitree_g1_sonic_rebased` → `main`

Status legend: [ ] open · [x] done · [~] partially done / needs reply

---

## 1. `controllers/gr00t_locomotion.py` & `holosoma_locomotion.py` — import style
> "It's actually better to have local imports."

- **Ask:** Use relative (local) imports for sibling/parent-package modules instead of the
  full absolute `from lerobot.robots.unitree_g1.g1_utils import ...` path.
- **Resolution:** [x] Done — switched to `from ..g1_utils import (...)` (files live in
  the `controllers/` subpackage). Pushed in commit "fix relative imports".

## 2. `controllers/sonic_whole_body.py` — private imports across files
> "These are private, they should technically not be imported outside their original file."
> (re: `_ort_providers`, `_snapshot_ms` imported from `sonic_pipeline`)

- **Ask:** Don't import underscore-prefixed (private) symbols outside their defining module.
- **Resolution:** [x] Done — renamed `_ort_providers` → `ort_providers` and
  `_snapshot_ms` → `snapshot_ms` in `sonic_pipeline.py`; updated all call sites.

## 3. `controllers/sonic_whole_body.py` / `sonic_pipeline.py` — safe optional imports
> "This should be a safe import with require_package." (re: `import onnxruntime as ort`)

- **Ask:** Guard optional heavy deps (`onnxruntime`, and `onnx` in holosoma) the way the
  rest of the repo does, with `require_package` + an availability flag.
- **Resolution:** [x] Done — added `_onnxruntime_available` / `_onnx_available` flags in
  `import_utils.py`, guarded the imports (`if TYPE_CHECKING or _..._available: import ...`),
  and call `require_package("onnxruntime", extra="unitree_g1")` (and `onnx`) in the
  controllers' `__init__`.

## 4. `controllers/__init__.py` (line 17)
> (comment on the module docstring / `__all__` block)

- **Status:** [x] Marked **Resolved** on GitHub.

---

## 5. `controllers/sonic_pipeline.py` — whole-file readability
> "This file is dense and very technical, which makes it hard to read. Could you add some
> comments and docstrings so we can clearly identify what is each object's purpose?"

- **Ask:** Add module-level + class/function docstrings across the 1,297-line pipeline.
- **Resolution:** [x] Done — added an architecture module docstring (planner subprocess ↔
  encoder/decoder ↔ controller data flow, `encode_mode` meanings, IsaacLab↔MuJoCo joint
  ordering) plus docstrings on every class (`StandingEncoderDecoder`, `SonicPlanner`,
  `PlannerController`, `MovementState`, etc.) and the key functions. No behavior change.
  Committed as `806d28a8`.

---

## 6. `controllers/sonic_whole_body.py` (line 72) — `f"{SMPL_ACTION_PREFIX}0"`
> "Is the 0 intended?"

- **Ask:** Clarify whether checking the literal key `smpl.0` is intentional.
- **Answer:** Yes — it's a sentinel: the presence of the first element (`smpl.0`) signals
  that a full 720-element SMPL window was sent in this action; if absent, there's no SMPL
  reference this tick and the function returns `None`.
- **Action:** [ ] Reply to the thread and/or add a clarifying inline comment.

## 7. `smpl_stream.py` (lines 43–45) — `DEFAULT_SMPL_HOST` / `DEFAULT_SMPL_PORT`
> "This should live in the Pico config, right?"

- **Ask:** Move the default host/port into the Pico headset config.
- **Notes:** They already exist in `config_pico_headset.py` (`smpl_host`, `smpl_port`).
  The module constants remain as fallbacks for the non-teleop paths
  (`sonic_whole_body`'s `SONIC_SMPL_STREAM` direct subscription, and `pico_publisher`).
- **Action:** [ ] Either reply clarifying the config already owns these, or fold the
  defaults into the shared constants module (see #8).

## 8. `smpl_stream.py` (lines 47–51) — `WINDOW`, `N_JOINTS`, `JOINT_DIM`, `SMPL_OBS_DIM`
> "These variables are used in other files, consider putting them in a shared 'constant'
> file so that we have a single source of truth (probably in the unitree folder)."

- **Ask:** Deduplicate the SMPL geometry constants (also re-derived as `SMPL_ACTION_DIM = 720`
  in `sonic_whole_body.py`, and referenced in `pico_headset.py` / `smpl_to_dataset.py`).
- **Action:** [ ] Create a shared constants module (e.g.
  `robots/unitree_g1/smpl_constants.py` or add to `g1_utils.py`) and import everywhere.
  Can also absorb the host/port defaults from #7.

## 9. `smpl_stream.py` (line 53) — `class SmplStream`
> "Could be a Dataclass"

- **Ask:** Consider making `SmplStream` a `@dataclass`.
- **Notes:** Only the ctor args (`host/port/fps/stale_after_s/loop`) are dataclass-friendly;
  most of `__init__` sets up a live ZMQ socket + rolling buffers, which suits a plain class.
  A dataclass with `__post_init__` is possible but low value.
- **Action:** [ ] Reply explaining the socket setup makes a plain class cleaner (or convert
  if she feels strongly). Non-blocking "consider".

## 10. `unitree_g1.py` (lines 488–490) — forwarding `smpl.` / `root.` keys
> "Won't these keys be included in the REMOTE_KEYS?"

- **Ask:** Is the `smpl.*`/`root.*` forwarding loop redundant with the `REMOTE_KEYS` loop?
- **Answer:** No — `REMOTE_KEYS` is the fixed joystick/button set; the `smpl.*`/`root.*`
  keys are the ~724 whole-body reference floats, which are not in `REMOTE_KEYS`, so the
  separate loop is required.
- **Action:** [ ] Reply confirming (and/or add a clarifying comment).

## 11. `sonic_pipeline.py` — custom keyboard handling duplicates existing utility
> (paraphrased) "We already have a keyboard utility — don't reimplement keyboard control."

- **Ask:** Reuse lerobot's shared keyboard infrastructure instead of a bespoke reader.
- **Notes:** `lerobot/utils/keyboard_input.py` already provides `TerminalKeyListener` /
  `create_key_listener` (cbreak terminal reader + pynput backend, cross-platform). The
  `RawKeyboard`/`drain_keyboard`/`process_keyboard` in `sonic_pipeline.py` reimplemented a
  subset of this — and were **dead code**: nothing in the lerobot package imported them
  (only the standalone `sonic_python/test_sonic_planner.py` has its own separate copies).
- **Resolution:** [x] Done — removed the unused `RawKeyboard`, `drain_keyboard`,
  `process_keyboard` and their now-unused `sys`/`select`/`termios`/`tty` imports. The G1
  integration drives movement via the joystick path (`process_joystick`), which is kept.

---

## Summary of remaining work

| # | File | Type | Remaining |
|---|------|------|-----------|
| 1 | gr00t/holosoma | relative imports | done |
| 2 | sonic_whole_body | private imports | done |
| 3 | sonic_whole_body/pipeline | safe onnx import | done |
| 4 | controllers/__init__ | — | resolved |
| 5 | sonic_pipeline | docstrings | done |
| 6 | sonic_whole_body:72 | `smpl.0` sentinel | reply / comment |
| 7 | smpl_stream:43–45 | host/port in config | reply / move to shared |
| 8 | smpl_stream:47–51 | shared constants | refactor |
| 9 | smpl_stream:53 | dataclass? | reply (push back) |
| 10 | unitree_g1:488–490 | REMOTE_KEYS overlap | reply / comment |
| 11 | sonic_pipeline | reuse keyboard utility | done (removed dead code) |
