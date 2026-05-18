# Language columns and recipes

Most LeRobot datasets ship with a single `task` string per episode — fine for
short, single-instruction skills, but not enough for the longer-horizon,
multi-modal robot policies the field is moving toward (high-level planning,
memory, interjections, VQA, tool use). To support those policies without
forking the dataset format, LeRobot extends `LeRobotDataset` with two optional
language columns and a small recipe layer that turns those rows into
chat-style training samples on the fly.

The design splits cleanly into three layers:

1. **Data in the dataset** — language annotations stored next to frames in
   `data/chunk-*/file-*.parquet` as two optional columns (`language_persistent`
   and `language_events`). Datasets without these columns keep their existing
   behavior.
2. **Recipe** — a YAML file that declares which annotation rows to bind and
   how to lay them out as chat turns (`role`, `content`, optional images,
   optional tool calls). Recipes are pure config; no Python required to add a
   new one.
3. **Training format** — at sample time, `RenderMessagesStep` resolves the
   recipe against the per-frame annotations and emits HF-style `messages` plus
   LeRobot-specific sidecars (`message_streams`, `target_message_indices`)
   that policy processors consume.

This page describes each layer in turn.

## Layer 1 — language columns in the dataset

The two optional columns live next to frame data in
`data/chunk-*/file-*.parquet`:

- `language_persistent`: a list of rows broadcast across every frame in an episode for state that remains active, such as `subtask`, `plan`, and `memory`.
- `language_events`: a list of rows only on the exact frame where an event was emitted, such as `interjection`, `vqa`, and speech tool calls.

Both columns share the same row shape (event rows omit `timestamp` because the
frame the row sits on already provides it):

```text
role: string
content: string | null
style: string | null
timestamp: float32        # persistent rows only
camera: string | null     # observation.images.* feature key, view-dependent rows only
tool_calls: list[Json] | null
```

The `camera` field tags rows whose `content` is grounded in a specific camera
view. Rows of view-dependent styles (`vqa` and `trace`) MUST set `camera` to
the matching `observation.images.*` feature key. Rows of every other style —
including `motion`, which describes robot-frame primitives in joint / Cartesian
terms — MUST leave `camera` as `null`. Pipeline writers and the validator
enforce this via `validate_camera_field(style, camera)`.

`meta/tasks.parquet` remains the canonical source for the task. The special `${task}` recipe binding always reads that task string and does not depend on language annotations.

### Architecture

The language stack itself has three internal modules backing layer 1:

1. `lerobot.datasets.language` defines the schema, style registry, and `column_for_style`.
2. `lerobot.datasets.language_render` resolves rows and renders messages.
3. `RenderMessagesStep` turns dataset samples into `messages`, `message_streams`, and `target_message_indices`.

`LeRobotDataset` stays recipe-agnostic. It passes `language_persistent` and `language_events` through when present, and unannotated datasets keep their existing behavior.

## Layer 2 — recipe anatomy

Recipes are YAML files backed by `TrainingRecipe` and `MessageTurn`. They
declare which annotation rows to pull (via `bindings`) and how to compose them
into chat turns (`messages`).

```yaml
messages:
  - { role: user, content: "${task}", stream: high_level }
  - { role: assistant, content: "${subtask}", stream: low_level, target: true }
```

A recipe can also branch into a weighted **blend** of sub-recipes. At sample
time, exactly one branch is selected deterministically from the sample index,
so different frames train different objectives (e.g. memory updates vs.
low-level execution vs. VQA) without any Python wiring.

### Temporal semantics

Persistent styles are active after emission until replaced:

- `active_at(t, style=subtask)`
- `nth_prev(style=memory, offset=1)`
- `nth_next(style=subtask, offset=1)`

Event styles only exist on their exact timestamp:

- `emitted_at(t, style=interjection)`
- `emitted_at(t, style=vqa, role=user, camera=observation.images.top)`
- `emitted_at(t, role=assistant, tool_name=say)`

Exact event matching has no tolerance window, so writers must stamp event rows with frame timestamps from the parquet data.

### View-dependent resolution

For view-dependent styles (`vqa` and `trace`), the resolver gains a
`camera=` filter parallel to `role=` and `tool_name=`. Datasets with multiple
cameras typically emit one (`vqa`, `user`) + (`vqa`, `assistant`) pair per
camera at the same timestamp; without `camera=`, those resolvers see two
matches and raise an ambiguity error. Recipes consume each camera through its
own binding plus a matching image block, e.g.

```yaml
ask_vqa_top:
  bindings:
    vqa_query: "emitted_at(t, style=vqa, role=user, camera=observation.images.top)"
    vqa: "emitted_at(t, style=vqa, role=assistant, camera=observation.images.top)"
  messages:
    - role: user
      stream: high_level
      if_present: vqa_query
      content:
        - { type: image, feature: observation.images.top }
        - { type: text, text: "${vqa_query}" }
    - {
        role: assistant,
        content: "${vqa}",
        stream: high_level,
        target: true,
        if_present: vqa,
      }
```

Add one such sub-recipe per camera the dataset records.

## Layer 3 — training format

Rendered samples use HF-style chat messages plus LeRobot sidecars:

```python
sample["messages"]
sample["message_streams"]
sample["target_message_indices"]
```

The renderer does not apply a tokenizer chat template. Policy processors decide how to serialize the messages for their backbone, which keeps the same dataset usable across SmolVLA, Pi0.5, and any future VLM that expects OpenAI-style chat messages.

## Graceful absence

If both language columns are missing, `None`, or empty, `RenderMessagesStep` is a no-op.
If an event-scoped branch is selected on a frame without the required event row, rendering returns `None`, allowing a loader to retry another sample.
