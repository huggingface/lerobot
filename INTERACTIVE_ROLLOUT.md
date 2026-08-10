# Interactive Rollout — Design Notes

Branch: `feat/add_interactive_rollout` · Status: Phases 1–2 committed; Round 2
(programmatic API, sentry support, muting v2, stdin move) implemented and tested,
uncommitted.

---

## 1. Vision

`lerobot-rollout` runs inference on a real robot: it connects hardware, loads the policy,
builds the processor pipelines, optionally records a dataset, and spins the control loop.
Today that is a **one-shot, fire-and-forget** program. You pass `--task="pick up the cube"`
on the command line, the robot starts moving immediately, and the only interaction left is
Ctrl-C. If you want a different instruction, you kill the process and pay the full startup
cost again — reconnecting motors, re-homing, re-loading a multi-GB VLA onto the GPU.

Since LeRobot gained subtask annotation and language conditioning, that model is the
bottleneck. The **north star** is a chat-style CLI over stdin, where the operator issues
commands _concurrently with the robot moving_:

```
/start                              begin (or resume) the policy control loop
/subtask Grab the red cube          re-instruct the policy on the fly
/ask what's the capital of France?  query an LLM while the robot keeps moving
/reset                              stop movement, return home, clear the subtask —
                                    but keep hardware and policy warm
/stop                               graceful shutdown
```

The unifying idea: **the expensive things (hardware, policy weights, processors) stay warm
across commands.** Only the cheap things — the instruction, the control loop — start and
stop. That turns a rollout from a batch job into a session you can steer.

## 2. Objective (scoped)

Phased, so each phase lands as a reviewable unit:

| Phase   | Scope                                                                                                                           | Status           |
| ------- | ------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| **1**   | `--interactive` flag, non-blocking stdin listener, command parser, `/start` `/reset` `/stop` `/help`                            | ✅ done          |
| **1.5** | Mute system logs so they stop fighting the prompt for the terminal                                                              | ✅ done          |
| **2**   | `/subtask <text>` — change the policy's instruction mid-run                                                                     | ✅ done          |
| **2.5** | Round 2: `RolloutController` public API, sentry recording support, muting v2 (errors surface), stdin listener → `lerobot/utils` | ✅ done (see §5) |
| **3**   | `/ask` + hierarchical task-vs-subtask semantics (LLM in the loop)                                                               | not started      |

An explicit constraint through Phases 1–2: **do not couple this to the language runtime yet.**
Build the mechanism; keep the door open.

## 3. Inspiration — three reference PRs

We read all three and deliberately implemented none of them verbatim.

**PR #4108 — online subtask switching.** Introduces a `PromptBroker` + `PromptListenerBase`

- `StdinPromptListener`, a `RuntimeContext.prompt_broker` field, `register_on_change`
  callbacks, an `--online_task_switching_flush` config flag, and `flush_action_queue()` /
  `_apply_pending_flush()` on `PreTrainedPolicy` — **with edits to 14 policy files** to call
  the flush at the top of `select_action`. Its architecture is designed for pluggable input
  sources (network, voice), which is the right long-term shape but more machinery than we
  need. _What we took:_ the core insight that a mid-run instruction change must invalidate
  actions precomputed under the old instruction, and that the flush must happen on a thread
  that is safe to touch policy state from.

**PR #4183 — experimental full-UX draft.** Achieves the whole north-star vision, but does
so by adding a `lerobot.runtime` / `language_runtime.py` that **duplicates** `BaseStrategy`,
`send_next_action`, and the rollout control loop. _What we took:_ the UX target and the
command vocabulary. _What we rejected:_ the parallel runtime — a second control loop is a
second thing to keep correct, and everything it does is already in `rollout/strategies/`.

**PR #4234 — policy-side edits enabling #4183's runtime.** Read for context on where the
language plumbing lands inside a policy. Relevant to Phase 3, not to what we built.

## 4. What we built, and why

Three commits on the branch:

```
072c697c0  feat(rollout): interactive v1
d3ee0b820  feat(rollout): mute logs in interactive mode
39c4e746f  feat(rollout): add subtask command
```

Cumulative footprint — one new module, one new test file, small surgical edits elsewhere:

```
 src/lerobot/rollout/interactive.py         | 580 +++++   (new)
 tests/test_interactive_rollout.py          | 788 +++++   (new)
 docs/source/inference.mdx                  |  87 +++
 src/lerobot/rollout/inference/base.py      |  66 +++
 src/lerobot/rollout/inference/rtc.py       |  61 +-
 src/lerobot/rollout/inference/sync.py      |  21 +-
 src/lerobot/scripts/lerobot_rollout.py     |  32 +-
 src/lerobot/policies/pretrained.py         |  24 +
 src/lerobot/rollout/strategies/core.py     |  21 +-
 src/lerobot/rollout/configs.py             |  18 +
 src/lerobot/rollout/__init__.py            |  16 +-
 src/lerobot/rollout/strategies/episodic.py |   4 +-
```

The ratio matters: **~1400 of ~1680 added lines are the new module and its tests.** The
existing rollout architecture was reused, not reshaped.

### 4.1 Segments over a linked event — the load-bearing idea

Every rollout strategy's control loop already polls `ctx.runtime.shutdown_event.is_set()`
to know when to stop. So instead of teaching strategies about interactivity, we **swap in a
smarter event**:

```python
class LinkedEvent(Event):
    """is_set() reflects the local flag OR a parent event."""
    def is_set(self) -> bool:
        return super().is_set() or self.parent.is_set()
```

`lerobot-rollout` wraps the `ProcessSignalHandler`'s shutdown event in a `LinkedEvent` when
`--interactive=true`. The session sets the **local** flag to end a run _segment_; SIGINT /
SIGTERM still arrive through the **parent**, so Ctrl-C behaves exactly as before.

`InteractiveSession.run()` then drives `strategy.run(ctx)` in restartable segments:

```
setup(ctx)  →  [idle]  →  /start → run(ctx) → /reset → [idle] → /start → run(ctx) → /stop  →  teardown(ctx)
                                                  ↑ hardware + policy stay warm throughout
```

**Zero strategy code changed** to support this. The only additions to `strategies/core.py`
were `reset_control_state()` (engine + interpolator + cached-observation reset, factored
out of `_init_engine` so a segment can restart cleanly) and making
`_return_to_initial_position` public.

### 4.2 Threading model

```
 listener thread  ──publishes flags / strings──▶  main thread
 (stdin reader)      never touches hardware        (session loop → strategy.run → control loop)
                     never mutates policy state
```

The listener only ever writes `threading.Event` flags and a lock-guarded string. Everything
that touches hardware or policy state happens on the thread that already owns it. This
mirrors the existing DAgger events pattern rather than inventing a new concurrency idiom.

### 4.3 stdin must be read with `os.read`, not `readline`

Non-obvious and load-bearing. The first implementation used `select()` + `stream.readline()`
and **two tests failed**: a buffered file object slurps _several_ lines off the file
descriptor in one syscall, after which `select` reports the drained fd as not-ready and the
buffered lines are never delivered. Pasted or piped command batches got stuck. The reader
now does `select()` + `os.read(fd, 4096)` + manual `\n` splitting, with a
blocking-`readline` fallback for streams without a `fileno()` (non-POSIX, test doubles).

Also: unlike `TerminalKeyListener`, this reader leaves the terminal in **canonical mode** —
the operator is typing chat commands, not pressing hotkeys.

### 4.4 EOF means stop

A closed stdin means there is no way left to command the robot, so EOF (Ctrl-D, or an
exhausted piped script) stops the session. An unexpected read error is treated the same way,
for the same reason. Consequence, documented: piped scripts must hold stdin open —

```bash
(printf '/start\n'; sleep 60; printf '/stop\n') | lerobot-rollout ... --interactive=true
```

### 4.5 Commands are last-write-wins

`/reset` and `/stop` cancel a still-pending `/start`, so the robot never starts moving after
the operator's most recent command said not to. Handlers set their intent flag _first_ and
the segment-stop event _second_; `_run_segment` clears the segment-stop flag _before_
re-checking the intent flags. A `/reset` racing a `/start` is therefore either seen before
the segment begins or ends it on its first tick.

### 4.6 Base strategy only (enforced by config validation)

`--interactive=true` with a recording strategy raises a `ValueError`. Two reasons: recording
strategies finalize their dataset inside `run()` (so `run()` is not restartable), and their
keyboard listeners contend with the command reader for the same TTY. This is a deliberate,
documented limitation — not an oversight.

### 4.7 Log muting (Phase 1.5)

Policy, robot and control-loop logs at every level interleave with the chat prompt and
destroy the typing UX. Simplest workable answer, per explicit request: **mute console output
for the duration of the session.**

- Every logger's console `StreamHandler` is raised above `CRITICAL` — **not just root**,
  because `transformers` and `datasets` attach their own stderr handlers with
  `propagate=False`.
- `warnings.simplefilter("ignore")`, with `warnings.filters` saved and restored.
- **File handlers are untouched** — anyone wanting a persistent log can attach one.
- Restored in `run()`'s `finally`, _before_ the closing `log_say`, so teardown logs are visible.

The obvious hazard: muting hides fatal errors. So `InferenceEngine` gained a
`failure_traceback` property, RTC captures its traceback in the fatal handler, and the
session prints it on failure. **Do not remove that when touching the failure path.**

"See both logs and prompt" — a pinned input line, `prompt_toolkit`-style — was deliberately
deferred: it needs a new dependency and a real TUI layer.

### 4.8 `/subtask` — the engine _is_ the broker

The pivotal call on Phase 2: **skip PR #4108's `PromptBroker`.** After Phase 1, the session
already owns the stdin thread and the parser, so a broker + listener base + on-change
callbacks + a new `RuntimeContext` field would be duplicate machinery — and callbacks firing
on the listener thread are exactly the cross-thread hazard we designed against.

Instead, `InferenceEngine` (the ABC every backend already implements) became the thread-safe
task holder:

```python
@property
def task(self) -> str: ...              # lock-guarded read

def set_task(self, task) -> bool:       # callable from ANY thread; True if it changed
    ...

def _take_task(self) -> tuple[str, bool]:   # consumed on the INFERENCE thread;
    ...                                     # returns (task, changed) and clears the edge
```

`/subtask` is then three lines: read `engine.task`, call `engine.set_task(text)`, print the
transition. No new module, no new context field, no callbacks.

**The flush problem, and why it got small.** When the instruction changes, a chunking policy
is still serving actions computed under the old one — up to `chunk_size` ticks of stale
behavior. PR #4108 solved this by adding `flush_action_queue()` / `_apply_pending_flush()`
to `PreTrainedPolicy` **and editing 14 policy files**, because its flush request arrived from
a foreign thread and had to be deferred to a safe point inside `select_action`.

Ours already runs _on_ the thread that calls `select_action`. So: one concrete method on
`PreTrainedPolicy` and **zero per-policy edits**.

```python
def drop_queued_actions(self) -> None:
    queues = getattr(self, "_queues", None)
    if isinstance(queues, dict) and ACTION in queues:
        queues[ACTION].clear()
    action_queue = getattr(self, "_action_queue", None)
    if action_queue is not None:
        action_queue.clear()
```

Two `getattr`s cover the repo's two queue idioms across all ~18 policies
(`_queues[ACTION]`: diffusion, smolvla, tdmpc, vqbet, wall_x, xvla, multi_task_dit, vla_jepa;
`_action_queue`: act, pi0, pi05, pi0_fast, eo1, evo1, groot, molmoact2, fastwam, lingbot_va).
Policies with no queue inherit a no-op.

**Why not `policy.reset()`?** That was the first implementation, and review caught it as too
blunt. For Diffusion it wipes the observation history, so the next chunk is planned from a
history of the current frame repeated — a visible discontinuity mid-motion. And ACT /
Diffusion / VQBeT / TDMPC don't read `task` at all, so they'd pay that jerk for nothing.
`drop_queued_actions` keeps episode state and drops only what is actually stale.

**RTC deliberately does _not_ flush.** Clearing its queue would leave the robot with no
commands for a full inference latency (~1 s on a VLA). Instead the next chunk is generated
under the new instruction and merged over the previous chunk's leftover prefix — the switch
lands within one inference and the motion stays continuous. That is exactly what RTC's
blending exists for. Documented per-backend in `inference.mdx`; no config flag, one sensible
default per backend.

**`/reset` restores the launch task on the listener thread.** Subtle and worth preserving:
the restore lives in `_cmd_reset`, not in `_reset_robot` (which runs later, on the main
thread). Otherwise `/reset` followed immediately by `/subtask` would be ordered by _service_
time rather than _command_ time, and the deferred restore would silently revert the new
instruction — deterministically so, for pasted or piped input. Both writers now run on the
same thread, so command order wins. There is a regression test driving this through a real pipe.

## 5. Round 2 — the feature becomes a library API

Four follow-up asks landed together (currently uncommitted on the branch):
make the components programmatic-API friendly (the priority), extend interactive
to recording where cheap, simplify muting / surface errors, and settle the
ssh/headless + `keyboard_input` question.

### 5.1 `RolloutController` — programmatic control

`interactive.py` bisected cleanly, so the generic control logic moved to a new
`rollout/controller.py`:

```python
controller = RolloutController(strategy, ctx, on_event=my_observer)
controller.serve()      # blocking loop (run it on whatever thread you like)
controller.start()      # -> bool: False when a segment is already running
controller.set_task(t)  # -> bool: re-instruct mid-run, from any thread
controller.reset()      # -> bool: True when the launch task was restored
controller.stop()
controller.task / .initial_task / .running / .failed / .failure_traceback
```

- **No I/O of its own** — no stdin, no prints, no log muting, no TTS. Every
  state transition that used to be a `print` is now a `RolloutEvent`
  (`SEGMENT_STARTED`, `SEGMENT_ENDED`, `RESET_STARTED/DONE/SKIPPED`,
  `ENGINE_FAILED`, `STOPPED`) emitted on the serve thread.
- **Thread-safe by lock, not by convention.** The old ordering guarantee
  (`/subtask` right after `/reset` must win) relied on both writes running on
  the single stdin thread. The controller serializes `start`/`reset`/`stop`/
  `set_task` with an internal lock, so the guarantee now holds for arbitrary
  caller threads — the prerequisite for network/voice front-ends.
- `InteractiveSession` shrank to a thin adapter: stdin listener + parser +
  rendering + muting; each command maps 1:1 onto a controller method, and the
  controller is exposed as `session.controller`.
- Exported from `lerobot.rollout`: `RolloutController`, `RolloutEvent`,
  `LinkedEvent`. `docs/source/inference.mdx` gained a **Programmatic control**
  section with a complete embedding example.

### 5.2 Sentry + interactive — recording while you steer

Decision, per the agreed criteria: the `/record` keyboard-handoff idea is
**medium-to-large** (listeners have no suspend/resume API and start at
creation, `esc` handlers are hardcoded and collide, pynput captures globally
while you type, and each strategy carries per-run stale flags) → rejected.
But the investigation showed **sentry has zero keyboard code** — the config
comment lumping it with the keyboard strategies was simply wrong — and its
only real blocker was one line: `with VideoEncodingManager(dataset)` inside
`run()` finalizes the dataset the first time `run()` returns, after which a
restarted segment would silently truncate the finalized parquet.

So `--interactive=true` now supports `--strategy.type=sentry`:

- **Finalization moved to `teardown()`** (which already called
  `dataset.finalize()`); `run()` is segment-restartable. Each segment saves
  complete episodes plus one tail partial episode; on a failed tail save the
  in-flight streaming encode is cancelled _and_ the half-mutated episode
  buffer is discarded (see §6, round 2).
- **Frames are labeled with `engine.dispatched_task`** — the instruction that
  generated the action actually sent — instead of a config or live-command
  snapshot (the writer already stores a task per frame). The RTC queue pairs
  each chunk with the task that generated it and every `get_action` pop
  updates the marker, so `/subtask` cannot relabel an old queued or
  interpolated action with the new instruction. This also resolved the
  "recorded frames ignore `/subtask`" open item for sentry.
- `episodes_since_push` hoisted to instance state so upload cadence survives
  segments.
- dagger / highlight / episodic stay excluded: keyboard conflicts plus per-run
  recording state that does not survive a restart.

### 5.3 Muting v2 — two lines, and errors surface

The ~30-line per-handler walk became `logging.disable(logging.WARNING)` with
the previous disable level restored afterwards. Strictly better coverage: the
gate applies before handler dispatch, so it covers `propagate=False` library
loggers _and_ loggers created mid-session (the old snapshot missed those) —
and **ERROR/CRITICAL now reach the console**, which the audit showed is safe:
no ERROR-level emitter fires periodically in healthy operation (the periodic
nuisances — slow-loop, camera hiccups — are WARNING-level and stay muted).
Documented trade-off: the gate also withholds INFO/WARNING from file handlers
during the session; acceptable because no default code path attaches one
(only `rl/actor`, `rl/learner`, `async_inference` pass `log_file`). The
`warnings` suppression stays (nothing calls `logging.captureWarnings`), and
`failure_traceback` surfacing stays as the belt-and-suspenders for fatal
engine errors.

### 5.4 stdin listener → `lerobot/utils/stdin_input.py`

The ssh/headless audit confirmed the listener was already the right design:
`select`+`os.read` works over SSH (the session pty is a normal fd), from
pipes, and headless — it's `keyboard_input`'s **pynput** backend that needs a
display server. Nothing in `keyboard_input` overlaps enough to reuse
(1-byte cbreak hotkey decoding vs canonical-mode line assembly), so
`StdinCommandListener` moved to a **new** utils module — deliberately not
into `keyboard_input.py`, which attempts a pynput import at module load.
Canonical import only: `lerobot.utils.stdin_input` (removed from
`lerobot.rollout`'s exports).

The move fixed a real bug the audit found: with `sys.stdin is None`
(daemonized processes), the blocking fallback died with an uncaught
`AttributeError` without firing `on_eof` — leaving a session idling with no
command channel. `start()` now treats a missing stream as immediate EOF.

## 6. Bugs the adversarial reviews caught

Four multi-agent review passes were run across the phases (28 / 5 / 27 / 12 agents;
findings adversarially verified before acting). The ones that mattered:

**Round 2 (2 confirmed, 0 refuted):**

- **Controller `start()` race → phantom segment.** `start()` gated on `_running`, but the
  serve loop cleared `_start_requested` _before_ setting `_running` — a second `start()`
  landing in that window (spanning `reset_control_state` and the SEGMENT_STARTED emission)
  returned `True` and re-armed the flag, which nothing consumed during the segment; the
  robot would start again, uncommanded, when the segment later ended on its own. Fixed:
  the serve loop consumes the request and sets `_running` atomically under the control
  lock, and `_running` spans the whole startup sequence.
- **Sentry poisoned episode buffer.** `save_episode` mutates the buffer in place (pops
  `size`/`task`) _before_ the fallible writes; a failed tail save left a half-mutated dict
  and the next segment's first `add_frame` crashed with `KeyError('size')`. Fixed: the
  except branch discards the buffer so `add_frame` recreates it.

**Rounds 1–3 (Phases 1–2):**

- **RTC stale observation (critical).** `RTCInferenceEngine.reset()` never cleared
  `_obs_holder["obs"]`. After `/reset` physically moved the robot home, the next `/start`
  computed its first chunk from the **pre-reset pose** — a lurch back toward where the arm
  used to be. Fixed by clearing the observation and adding a `_reset_epoch` counter so an
  in-flight chunk computed across a reset is discarded rather than merged. This also fixes a
  pre-existing DAgger staleness path.
- **Muting hid fatal errors** → `failure_traceback` capture + session print (§4.7).
- **Muting scope too narrow** → root-only missed `transformers` / `datasets`; `warnings`
  output bypassed logging entirely.
- **Command ordering** → `/reset` and `/stop` didn't cancel a pending `/start` (§4.5); the
  `/reset`-then-`/subtask` clobber (§4.8).
- **Flush too heavy** → `policy.reset()` → `drop_queued_actions()` (§4.8).
- **Empty-task rendering** → `''` replaced with `(none — set one with /subtask <text>)`.
- **Silent switch** → the confirmation now says "(applies from the next policy inference)",
  since the explanatory logs are muted.

## 7. Verification

After Round 2:

```
uv run --extra dataset pytest tests/test_interactive_rollout.py \
    tests/utils/test_stdin_input.py tests/test_rollout.py -q
    → 81 passed

pre-commit (all changed files)
    → 0 failures
```

Phase 1–2 numbers (still green at the time): 64 rollout/interactive tests;
223 passed / 5 skipped across `tests/policies/rtc`, factory, and common
(confirming the shared `pretrained.py` change); pre-commit 0 failures.

`tests/test_interactive_rollout.py` covers the parser, `LinkedEvent` semantics,
`RolloutController` (start/reset/stop/set_task flows, events, startup-race
rejection, failure surfacing, broken observers), session flows (start / reset /
restart / stop, cancel-pending-start, engine failure with traceback, natural
end, EOF, and a real `BaseStrategy` end-to-end), muting (INFO/WARNING blocked,
ERROR surfaces, pre-existing disable level restored), `/subtask` semantics,
sentry restartability + action-provenance labels + failed-tail-save recovery,
the engine task holder, the sync flush, and `drop_queued_actions`.
`tests/utils/test_stdin_input.py` covers the listener (select path, batched
lines, blocking fallback, EOF, handler errors, None-stdin, broken streams).

## 8. Extension points for Phase 3

The design was built to make `/ask` an additive change:

- **Command table.** `InteractiveSession._commands` is `name → (handler, arg hint, help)`.
  `/help` and the startup banner render from it, so a new command is documented for free.
- **Controller API.** New front-ends (network, voice, `/ask`'s LLM worker) call
  `RolloutController.start/reset/stop/set_task` from their own threads — the internal lock
  makes that safe — and observe `RolloutEvent`s instead of scraping terminal output.
- **Thread discipline.** A command handler runs on the listener thread and must only call
  controller methods. An LLM call belongs on its own worker thread so the robot keeps
  moving — precisely the concurrency `/ask` is meant to demonstrate.
- **Task holder.** `set_task` / `_take_task` already give any producer a safe way to
  re-instruct the policy. Hierarchical task-vs-subtask semantics (per #4183 / #4234) layer
  on top of it rather than replacing it.

Open items, deliberately not addressed:

- dagger / highlight / episodic remain non-interactive (keyboard conflicts + per-run
  recording state). Sentry is the supported recording path for interactive sessions.
- The "see logs and prompt simultaneously" TUI (pinned input line).
- Non-stdin input sources (network, voice) — now unblocked by `RolloutController`; #4108's
  pluggable-listener shape remains the reference for the transport layer.
