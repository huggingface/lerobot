# Hot-Switching Language Prompts at Runtime

`lerobot-rollout` supports updating the language task prompt **while the robot is running** — no restart required.  
This is the foundation for voice-commanded robots: a speech-to-text (STT) process can pipe new tasks into the rollout script in real time.

---

## Quick Start

```bash
# Enable hot-switching via stdin
lerobot-rollout \
    --strategy.type=base \
    --policy.path=user/my_pi0_policy \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --task="pick up cube" \
    --hot_prompt=true \
    --duration=0
```

Once running, type a new task in the terminal and press **Enter**:

```
> fold the towel
```

The log will immediately show:

```
INFO  Task switched: 'pick up cube' → 'fold the towel'
```

Every subsequent inference call uses the new task — no episode boundary, no restart.

---

## Configuration Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--hot_prompt` | `bool` | `false` | Enable runtime task switching |
| `--hot_prompt_source` | `str` | `"stdin"` | Where new prompts come from. Currently: `"stdin"` |

Both flags live in `RolloutConfig` ([src/lerobot/rollout/configs.py](../src/lerobot/rollout/configs.py)).

---

## How It Works

### Architecture Overview

```
External source (terminal / STT pipe / future HTTP)
         │
         │  new task string
         ▼
  ┌─────────────────┐
  │  PromptBroker   │  ← thread-safe string holder
  └────────┬────────┘
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
broker.get_task()        # → "pick up cube"  (thread-safe)
broker.set_task("fold the towel")  # logs the transition, thread-safe
broker.get_task()        # → "fold the towel"
```

`set_task` logs `Task switched: 'old' → 'new'` only when the value changes, so it is safe to call in a tight loop.

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
        --policy.path=user/my_pi0_policy \
        ...
```

The STT process owns `stdout`; `lerobot-rollout` reads it on `stdin`.  
Any tool that writes one task string per line works — Whisper, Vosk, Google STT, etc.

### Method 3 — Named pipe (FIFO)

Use a FIFO to decouple the STT process from the rollout process, or to send commands from a separate terminal:

```bash
mkfifo /tmp/robot_task

# Terminal 1: start rollout reading from the FIFO
lerobot-rollout \
    --hot_prompt=true \
    --hot_prompt_source=stdin \
    --task="initial task" ... < /tmp/robot_task

# Terminal 2: send commands at any time
echo "pick up the blue block" > /tmp/robot_task
echo "drop it in the bin"    > /tmp/robot_task
```

### Method 4 — Script / programmatic (future: HTTP)

A future `HttpPromptListener` will expose a small REST endpoint:

```bash
# Start rollout with HTTP source (not yet implemented)
lerobot-rollout --hot_prompt=true --hot_prompt_source=http --hot_prompt_port=8765 ...

# Send from anywhere
curl -X POST http://localhost:8765/task -d '{"task": "fold the towel"}'

# Gradio UI just wraps this endpoint
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
| [`src/lerobot/rollout/prompt_broker.py`](../src/lerobot/rollout/prompt_broker.py) | **New** — `PromptBroker`, `PromptListenerBase`, `StdinPromptListener` |
| [`src/lerobot/rollout/configs.py`](../src/lerobot/rollout/configs.py) | Added `hot_prompt: bool`, `hot_prompt_source: str` to `RolloutConfig` |
| [`src/lerobot/rollout/context.py`](../src/lerobot/rollout/context.py) | `RuntimeContext.prompt_broker` field; broker creation + listener start in `build_rollout_context()` |
| [`src/lerobot/rollout/inference/sync.py`](../src/lerobot/rollout/inference/sync.py) | `prompt_broker` param; per-call `broker.get_task()` in `get_action()` |
| [`src/lerobot/rollout/inference/rtc.py`](../src/lerobot/rollout/inference/rtc.py) | `prompt_broker` param; per-call `broker.get_task()` in `_rtc_loop()` |
| [`src/lerobot/rollout/inference/factory.py`](../src/lerobot/rollout/inference/factory.py) | `prompt_broker` kwarg forwarded to both engine constructors |
| [`src/lerobot/rollout/strategies/sentry.py`](../src/lerobot/rollout/strategies/sentry.py) | Per-frame broker read instead of pre-loop capture |
| [`src/lerobot/rollout/strategies/highlight.py`](../src/lerobot/rollout/strategies/highlight.py) | Per-frame broker read instead of pre-loop capture |
| [`src/lerobot/rollout/strategies/dagger.py`](../src/lerobot/rollout/strategies/dagger.py) | Per-frame broker read in both `_run_continuous_recording` and `_run_corrections_only` |
| [`src/lerobot/rollout/__init__.py`](../src/lerobot/rollout/__init__.py) | Re-exports `PromptBroker`, `StdinPromptListener` |
| [`tests/test_prompt_broker.py`](../tests/test_prompt_broker.py) | **New** — 11 unit tests (6 broker, 5 listener) |

---

## Backward Compatibility

When `--hot_prompt=false` (the default), `prompt_broker` is `None` everywhere and all code falls back to the original static `self._task` / `cfg.task` path.  
**No behaviour change for existing users.**
