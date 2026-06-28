# Hot-Switching Language Prompts at Runtime

`lerobot-rollout` supports updating the language task prompt **while the robot is running** — no restart required.  
This is the foundation for voice-commanded robots: a speech-to-text (STT) process can pipe new tasks into the rollout script in real time.

---

## Quick Start

```bash
# Enable hot-switching via stdin — task switch takes effect immediately (flush mode)
lerobot-rollout \
    --strategy.type=base \
    --policy.path=user/my_smolvla_policy \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --task="pick up cube" \
    --hot_prompt=true \
    --hot_prompt_flush=true \
    --duration=0
```

Once running, type a new task in the terminal and press **Enter**:

```
> fold the towel
```

The log will immediately show:

```
INFO  Task switched: 'pick up cube' → 'fold the towel'
INFO  hot_prompt_flush=on — action queue will be cleared immediately on task switch
```

Every subsequent inference call uses the new task — no episode boundary, no restart.

```bash
# Alternatively, let the current action chunk drain before switching
# (smoother motion continuity, new task takes effect up to chunk_size ticks later)
lerobot-rollout \
    --strategy.type=base \
    --policy.path=user/my_smolvla_policy \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --task="pick up cube" \
    --hot_prompt=true \
    --hot_prompt_flush=false \
    --duration=0
```

---

## Configuration Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--hot_prompt` | `bool` | `false` | Enable runtime task switching |
| `--hot_prompt_source` | `str` | `"stdin"` | Where new prompts come from. Currently: `"stdin"` |
| `--hot_prompt_flush` | `bool` | `true` | Flush the policy action queue immediately on switch (set to `false` to let the current chunk drain) |

All three flags live in `RolloutConfig` ([src/lerobot/rollout/configs.py](../src/lerobot/rollout/configs.py)).

---

## How It Works

### Architecture Overview

```
External source (terminal / STT pipe / future HTTP)
         │
         │  new task string
         ▼
  ┌──────────────────────┐
  │     PromptBroker     │  ← thread-safe string holder
  │                      │
  │  register_on_change  │  ← optional callbacks fired on every switch
  └──────────┬───────────┘
             │ broker.get_task()
       ┌─────┴──────┐
       │            │
       ▼            ▼
  SyncEngine    RTCEngine       ← per-inference-call read
       │            │
       └─────┬──────┘
             │ task_str = broker.get_task()
             ▼
     Recording strategies        ← per-frame read
   (sentry / highlight / dagger)
```

### PromptBroker

[`src/lerobot/rollout/prompt_broker.py`](../src/lerobot/rollout/prompt_broker.py)

A lightweight thread-safe wrapper around a string:

```python
from lerobot.rollout.prompt_broker import PromptBroker

broker = PromptBroker(initial_task="pick up cube")
broker.get_task()                    # → "pick up cube"  (thread-safe)
broker.set_task("fold the towel")    # logs the transition, thread-safe
broker.get_task()                    # → "fold the towel"
```

`set_task` logs `Task switched: 'old' → 'new'` only when the value changes, so it is safe to call in a tight loop.

#### On-change callbacks

Any zero-argument callable can be registered to run in the listener thread immediately after a task switch:

```python
broker.register_on_change(my_callback)   # called on every task change
```

This is how the **action-queue flush** is wired (see below).

### PromptListenerBase / StdinPromptListener

Listeners run in a **daemon thread** and call `broker.set_task()` when a new prompt arrives.  
They implement a single abstract method `_listen(broker, shutdown_event)`.

`StdinPromptListener` uses `select.select` with a 0.5 s timeout so the thread exits cleanly when `shutdown_event` is set:

```python
from lerobot.rollout.prompt_broker import StdinPromptListener

listener = StdinPromptListener()
listener.start(broker, shutdown_event)   # non-blocking, daemon thread
```

### Wiring in the Context

`build_rollout_context()` ([`src/lerobot/rollout/context.py`](../src/lerobot/rollout/context.py)) creates the broker and starts the listener **before** building the inference engine:

```python
if cfg.hot_prompt:
    prompt_broker = PromptBroker(initial_task=task_str)
    if cfg.hot_prompt_source == "stdin":
        StdinPromptListener().start(prompt_broker, shutdown_event)
```

The broker is then stored on `RuntimeContext.prompt_broker` and passed to the inference engine factory.

### Inference Engines

Both `SyncInferenceEngine` and `RTCInferenceEngine` accept an optional `prompt_broker`.  
When set, they call `broker.get_task()` **on every inference call** instead of using the static `self._task` captured at startup:

```python
# SyncInferenceEngine.get_action()
_task = self._prompt_broker.get_task() if self._prompt_broker else self._task
observation = prepare_observation_for_inference(observation, self._device, _task, ...)
```

```python
# RTCInferenceEngine._rtc_loop()
_task = self._prompt_broker.get_task() if self._prompt_broker else self._task
obs_batch = prepare_observation_for_inference(obs_batch, policy_device, _task, ...)
obs_batch["task"] = [_task]
```

### Recording Strategies

All three recording strategies (`sentry`, `highlight`, `dagger`) resolve `task_str` **per frame** inside the control loop, so every dataset frame is labelled with the task that was active at the moment it was captured:

```python
task_str = (
    ctx.runtime.prompt_broker.get_task()
    if ctx.runtime.prompt_broker
    else (cfg.dataset.single_task if cfg.dataset else cfg.task)
)
frame = {**obs_frame, **action_frame, "task": task_str}
dataset.add_frame(frame)
```

This means a single episode can contain frames with **different task labels** if the operator switches tasks mid-episode.

---

## Action-Chunking Policies and the Flush Mechanism

### The chunking problem

Action-chunking policies (SmolVLA, Pi0, Diffusion Policy, ACT, etc.) do not call the neural network on every control tick. Instead they run **one expensive forward pass** to generate a *chunk* of N actions (e.g. `chunk_size=50` for SmolVLA), push them into an internal queue, and then pop one per tick without touching the model again.

This means that after a task switch:
- The broker and inference engine immediately see the new task string.
- But the policy still pops precomputed actions from the *old* task until the queue drains.
- For SmolVLA with `chunk_size=50` at 5 Hz that is up to **10 seconds** of stale behaviour.

### `PreTrainedPolicy.flush_action_queue()`

All policies inherit a `flush_action_queue()` method from `PreTrainedPolicy`
([`src/lerobot/policies/pretrained.py`](../src/lerobot/policies/pretrained.py)):

```python
policy.flush_action_queue()
```

Calling it **requests** that precomputed actions be discarded so that the very next `select_action()` call can trigger a fresh forward pass with the current (new) task string.

The request is intentionally deferred for thread safety: the prompt-listener callback runs in a background thread, while queue consumption (`popleft()`) happens in the main control loop. Clearing the queue directly from the listener thread can race with `select_action()`, so the base class records a flush request and the policy applies it on the next control tick.

At the base-class level the implementation is:

```python
# PreTrainedPolicy.flush_action_queue()
self._flush_requested = True

# PreTrainedPolicy._apply_pending_flush()
if hasattr(self, "_queues") and ACTION in self._queues:
    self._queues[ACTION].clear()          # SmolVLA, Diffusion, TDMPC, VQBeT, WallX, …
if hasattr(self, "_action_queue"):
    self._action_queue.clear()            # ACT, Pi0, EO1, Groot, Pi05, …
```

Current queue-backed policies call `_apply_pending_flush()` at the start of `select_action()`, so the deferred flush is now honored consistently across the rollout stack.

Policies without a queue inherit a silent no-op.

### Wiring flush to the broker

Pass `policy.flush_action_queue` as an on-change callback:

```python
broker.register_on_change(policy.flush_action_queue)
```

Now whenever the operator types a new task the listener thread records a flush request, and the **next** `get_action()` / `select_action()` cycle applies it safely from the main control thread before consuming any stale actions.

### Flush vs. drain — choosing the right behaviour

| Mode | Effect | When to use |
|------|--------|-------------|
| **Flush on switch** (default) | Queue cleared immediately; VLM re-runs on next tick (~1–2 s GPU time) | Maximum responsiveness; voice commands |
| **Drain first** | Current chunk runs to completion; new task takes effect naturally | Smooth motion continuity; avoid mid-trajectory interruption |

In the SmolVLA demo this is controlled by `--flush_on_switch` / `--no-flush_on_switch`.

---

## Sending Prompts

### Method 1 — Interactive terminal (stdin)

Run the script in a terminal and type prompts directly:

```bash
lerobot-rollout --hot_prompt=true --task="pick up cube" ...

# In the same terminal, type and press Enter:
> grasp the red block
> place it in the bin
```

### Method 2 — Unix pipe from STT

Pipe the output of any speech-to-text tool directly:

```bash
# whisper_stt_tool outputs one transcription per line to stdout
whisper_stt_tool --model=base | \
    lerobot-rollout \
        --hot_prompt=true \
        --hot_prompt_source=stdin \
        --task="waiting for command" \
        --strategy.type=base \
        --policy.path=user/my_smolvla_policy \
        ...
```

The STT process owns `stdout`; `lerobot-rollout` reads it on `stdin`.  
Any tool that writes one task string per line works — Whisper, Vosk, Google STT, etc.

### Method 3 — File + `tail -f`

Write tasks to a plain file from any terminal; `tail -f` streams new lines to the rollout's stdin as they arrive.
This avoids the blocking-open problem of FIFOs and works across conda environments.

```bash
# Terminal 1: create the file and start the rollout
> /tmp/robot_task   # create (or truncate) the file

lerobot-rollout \
    --hot_prompt=true \
    --hot_prompt_source=stdin \
    --task="initial task" \
    ... < <(tail -f /tmp/robot_task)

# Terminal 2: send commands at any time (append, not overwrite)
echo "pick up the blue block" >> /tmp/robot_task
echo "drop it in the bin"    >> /tmp/robot_task
echo "initial task"          >> /tmp/robot_task
```

> **Note:** use `>>` (append) in Terminal 2, not `>`.  A single `>` would truncate the file and `tail -f` would lose its position.

### Method 4 — Script / programmatic (future: HTTP)

A future `HttpPromptListener` will expose a small REST endpoint:

```bash
# Start rollout with HTTP source (not yet implemented)
lerobot-rollout --hot_prompt=true --hot_prompt_source=http --hot_prompt_port=8765 ...

# Send from anywhere
curl -X POST http://localhost:8765/task -d '{"task": "fold the towel"}'
```

---

## SmolVLA Demo (no robot required)

`examples/hot_prompt_smolvla_demo.py` loads a real SmolVLA checkpoint and feeds it synthetic (zero) camera frames and robot state, so you can test hot-prompt switching without any hardware.

```bash
# Terminal 1 — start the demo
conda activate lerobot_rollout
> /tmp/robot_task   # create the file
python examples/hot_prompt_smolvla_demo.py \
    --checkpoint /path/to/pretrained_model \
    --task "pick up the bottle" \
    --fps 0.5 \
    --device cuda \
    < <(tail -f /tmp/robot_task)

# Terminal 2 — send new tasks at any time
echo "grab the red cup"    >> /tmp/robot_task
echo "place it in the bin" >> /tmp/robot_task
echo "pick up the bottle"  >> /tmp/robot_task
```

The switch is visible in the logs immediately:

```text
INFO  Task switched: 'pick up the bottle' → 'grab the red cup'
INFO  [SyncInference] task='grab the red cup' | action[0:5]=...
```

The demo loop also logs the active task per tick, so you can verify exactly when the rollout starts executing under the new instruction.

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | *(required)* | Path to `pretrained_model/` directory |
| `--task` | `"pick up the bottle"` | Initial task string |
| `--fps` | `1.0` | Control loop frequency (Hz) |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--flush_on_switch` | `true` | Flush action queue on task change (use `--no-flush_on_switch` to drain first) |
| `--duration` | `0` | Run for N seconds; 0 = infinite |

### What happens inside

1. `PromptBroker` is created with the initial task.
2. `StdinPromptListener` daemon thread watches for new lines on stdin.
3. If `--flush_on_switch` (default): `broker.register_on_change(policy.flush_action_queue)` is called so a deferred flush request is recorded on every switch.
4. `SyncInferenceEngine` reads `broker.get_task()` on every `get_action()` call and injects it into the observation before calling the policy.
5. Queue-backed policies apply that pending flush at the start of `select_action()`, so stale buffered actions are discarded safely in the main control thread.
6. The preprocessor appends `\n` (`NewLineTaskProcessorStep`) then tokenizes the string before the VLM sees it.

The active task can be confirmed from the rollout logs:

```
[SyncInference] task='pick up the bottle' | action[0:5]=...
tick   12 | task='pick up the bottle' | action[0:5]=...
```

---

## Adding a New Prompt Source

1. Subclass `PromptListenerBase` in `src/lerobot/rollout/prompt_broker.py`:

```python
class MyCustomListener(PromptListenerBase):
    def _listen(self, broker: PromptBroker, shutdown_event: Event) -> None:
        while not shutdown_event.is_set():
            task = my_custom_source.read_next_task()   # blocking or polling
            if task:
                broker.set_task(task)
```

2. Register it in `build_rollout_context()` in `context.py`:

```python
elif cfg.hot_prompt_source == "my_custom":
    MyCustomListener().start(prompt_broker, shutdown_event)
```

3. Add the new key to the `hot_prompt_source` docstring in `configs.py`.

No changes needed in the inference engines or strategies — they always read from the broker.

---

## Files Changed

| File | Change |
|------|--------|
| [`src/lerobot/rollout/prompt_broker.py`](../src/lerobot/rollout/prompt_broker.py) | **New** — `PromptBroker` (with `register_on_change`), `PromptListenerBase`, `StdinPromptListener` |
| [`src/lerobot/rollout/configs.py`](../src/lerobot/rollout/configs.py) | Added `hot_prompt`, `hot_prompt_source`, and `hot_prompt_flush` to `RolloutConfig` |
| [`src/lerobot/rollout/context.py`](../src/lerobot/rollout/context.py) | `RuntimeContext.prompt_broker` field; broker creation + listener start in `build_rollout_context()` |
| [`src/lerobot/rollout/inference/sync.py`](../src/lerobot/rollout/inference/sync.py) | `prompt_broker` param; per-call `broker.get_task()` in `get_action()` |
| [`src/lerobot/rollout/inference/rtc.py`](../src/lerobot/rollout/inference/rtc.py) | `prompt_broker` param; per-call `broker.get_task()` in `_rtc_loop()` |
| [`src/lerobot/rollout/inference/factory.py`](../src/lerobot/rollout/inference/factory.py) | `prompt_broker` kwarg forwarded to both engine constructors |
| [`src/lerobot/rollout/strategies/sentry.py`](../src/lerobot/rollout/strategies/sentry.py) | Per-frame broker read instead of pre-loop capture |
| [`src/lerobot/rollout/strategies/highlight.py`](../src/lerobot/rollout/strategies/highlight.py) | Per-frame broker read instead of pre-loop capture |
| [`src/lerobot/rollout/strategies/dagger.py`](../src/lerobot/rollout/strategies/dagger.py) | Per-frame broker read in both `_run_continuous_recording` and `_run_corrections_only` |
| [`src/lerobot/rollout/__init__.py`](../src/lerobot/rollout/__init__.py) | Re-exports `PromptBroker`, `StdinPromptListener` |
| [`src/lerobot/policies/pretrained.py`](../src/lerobot/policies/pretrained.py) | Deferred `flush_action_queue()` + `_apply_pending_flush()` helper added to `PreTrainedPolicy` |
| `src/lerobot/policies/*/modeling_*.py` | Queue-backed policies call `_apply_pending_flush()` at the start of `select_action()` |
| [`tests/test_prompt_broker.py`](../tests/test_prompt_broker.py) | **New** — 11 unit tests (6 broker, 5 listener) |
| [`tests/test_rollout.py`](../tests/test_rollout.py) | 3 integration tests for broker + `SyncInferenceEngine` |
| [`examples/hot_prompt_demo.py`](../examples/hot_prompt_demo.py) | **New** — mock-robot demo, no GPU needed |
| [`examples/hot_prompt_smolvla_demo.py`](../examples/hot_prompt_smolvla_demo.py) | **New** — real SmolVLA checkpoint demo with synthetic frames; `--flush_on_switch` flag |

---

## Backward Compatibility

When `--hot_prompt=false` (the default), `prompt_broker` is `None` everywhere and all code falls back to the original static `self._task` / `cfg.task` path.  
**No behaviour change for existing users.**

