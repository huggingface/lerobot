# GR00T N1.6 Training & Inference Flow (LeRobot)

This document explains the current training and inference pipeline for the GR00T N1.6 policy
as implemented in this repo. It focuses on where data is loaded, batched, processed, passed
to the model, how loss is computed, and how normalization/unnormalization happens. Each
section lists the key function names (so you can search them), their inputs/outputs, and
important tensor shapes.

## Entry Points (Where to Start Reading)

- Training pipeline entrypoint: `lerobot/scripts/lerobot_train.py::train`
- Policy wrapper & forward: `lerobot/policies/gr00t_n1d6/modeling_gr00t_n1d6.py::Gr00tN1d6Policy`
- Core model & action head: `lerobot/policies/gr00t_n1d6/gr00t_n1d6.py::Gr00tN1d6` and `Gr00tN1d6ActionHead`
- Pre/post processors: `lerobot/policies/gr00t_n1d6/processor_gr00t_n1d6.py::make_gr00t_n1d6_pre_post_processors`

Key code anchors:

```417:492:src/lerobot/scripts/lerobot_train.py
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    ...
    batch = next(dl_iter)
    batch = preprocessor(batch)
    ...
    train_tracker, output_dict = update_policy(...)
```

```1032:1063:src/lerobot/datasets/lerobot_dataset.py
    def __getitem__(self, idx) -> dict:
        self._ensure_hf_dataset_loaded()
        item = self.hf_dataset[idx]
        ...
        if len(self.meta.video_keys) > 0:
            ...
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}
        ...
        item["task"] = self.meta.tasks.iloc[task_idx].name
        return item
```

```985:1213:src/lerobot/policies/gr00t_n1d6/processor_gr00t_n1d6.py
class Gr00tN1d6ProcessStep(ProcessorStep):
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        ...
        processed = self.processor([{"content": vla_step_data}])
        processed_list.append(processed)
        collated = self.processor.collator(processed_list).data["inputs"]
        transition[TransitionKey.OBSERVATION] = collated
        return transition
```

```206:417:src/lerobot/policies/gr00t_n1d6/gr00t_n1d6.py
    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        ...
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)
        return {"loss": loss, ...}
```

## Training Pipeline: Data Loading → Preprocessing → Model → Loss

### 1) Dataset & DataLoader

**Dataset class and sampling**

- `lerobot/datasets/lerobot_dataset.py::LeRobotDataset.__getitem__`
  - **Input**: `idx` frame index.
  - **Output**: dict with observation/action data (e.g., `observation.state`, `observation.images.*`,
    `action`, `task`, timestamps, etc.). Images are loaded from videos and optionally transformed.
  - **Important shapes**:
    - `observation.images.<view>` typically `[C, H, W]` (or `[T, C, H, W]` if delta indices used).
    - `observation.state` often `[state_dim]` (or `[T, state_dim]`).
    - `action` often `[action_dim]` (or `[T, action_dim]`).

- `lerobot/datasets/streaming_dataset.py::StreamingLeRobotDataset.__iter__`
  - Used when `cfg.dataset.streaming=True`. It uses `safe_shard(...)` to shard the dataset for streaming
    iteration across workers/processes.

**DataLoader creation**

- `lerobot/scripts/lerobot_train.py::train`
  - Uses `torch.utils.data.DataLoader` with `batch_size=cfg.batch_size`.
  - `Accelerator.prepare(...)` handles device placement and distributed sharding under `accelerate`.
  - When using the episode-aware sampler, shuffling is disabled and sampling is done by episode.

### 2) Preprocessor Pipeline (Training)

**Pipeline construction**

- `lerobot/policies/gr00t_n1d6/processor_gr00t_n1d6.py::make_gr00t_n1d6_pre_post_processors`
  - **Inputs**: `Gr00tN1d6Config`, optional `dataset_stats`.
  - **Outputs**: `(preprocessor_pipeline, postprocessor_pipeline)`.
  - **Preprocessor steps** (in order):
    1. `RenameObservationsProcessorStep`
    2. `AddBatchDimensionProcessorStep`
    3. `Gr00tN1d6ProcessStep`
    4. `DeviceProcessorStep`
  - **Where stats are wired**: `dataset_stats` are converted to the nested
    structure required by `StateActionProcessor` and passed into `Gr00tN1d6Processor`.

```1626:1769:src/lerobot/policies/gr00t_n1d6/processor_gr00t_n1d6.py
def make_gr00t_n1d6_pre_post_processors(...):
    ...
    processor = Gr00tN1d6Processor(...)
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        Gr00tN1d6ProcessStep(processor=processor, language_key="task"),
        DeviceProcessorStep(device=config.device),
    ]
```

**Gr00tN1d6ProcessStep → VLAStepData → Collation**

- `Gr00tN1d6ProcessStep.__call__`
  - **Input**: `EnvTransition` with keys:
    - `TransitionKey.OBSERVATION` containing `observation.state`, `observation.images.*`.
    - `TransitionKey.ACTION` containing action (training only).
    - `TransitionKey.COMPLEMENTARY_DATA` containing `task` string (language).
  - **Output**: `TransitionKey.OBSERVATION` replaced with collated model inputs.
  - **Process**:
    1. Convert per-batch images to `np.uint8`, shape `[H, W, C]`.
    2. Convert state/action tensors into dicts keyed by modality keys.
    3. Build one `VLAStepData` per batch element.
    4. Call `Gr00tN1d6Processor` and then `Gr00tN1d6DataCollator`.

```1122:1213:src/lerobot/policies/gr00t_n1d6/processor_gr00t_n1d6.py
for i in range(batch_size):
    ...
    vla_step_data = VLAStepData(...)
    processed = self.processor([{"content": vla_step_data}])
    processed_list.append(processed)
collated = self.processor.collator(processed_list).data["inputs"]
transition[TransitionKey.OBSERVATION] = collated
```

**Gr00tN1d6Processor (core preprocessing)**

- `Gr00tN1d6Processor.__call__`
  - **Input**: list of messages with `VLAStepData` containing:
    - `images`: dict of view → list of frames (each `np.ndarray`).
    - `states`: dict of joint_group → `np.ndarray`.
    - `actions`: dict of joint_group → `np.ndarray` (training only).
    - `text`: task description string.
    - `embodiment`: `EmbodimentTag`.
  - **Output**: dict of model inputs:
    - `state`: `torch.Tensor` `[T_state, max_state_dim]` (T_state is usually 1).
    - `action`: `torch.Tensor` `[max_action_horizon, max_action_dim]` (training only).
    - `action_mask`: `torch.Tensor` same shape as `action` with 1s for valid entries.
    - `vlm_content`: dict with `text`, `images`, `conversation` for VLM collation.
    - `embodiment_id`: `int` mapped from tag.
  - **Normalization + padding**:
    - Uses `StateActionProcessor.apply(...)` to normalize state/actions.
    - Pads action to `max_action_dim` and `max_action_horizon`, creates `action_mask`.
    - Pads state to `max_state_dim`.

```783:909:src/lerobot/policies/gr00t_n1d6/processor_gr00t_n1d6.py
normalized_states, normalized_actions = self.state_action_processor.apply(...)
...
normalized_actions = torch.cat([...], dim=-1)  # (t, d)
normalized_actions = torch.cat([...], dim=-1)  # (t, max_action_dim)
normalized_actions = torch.cat([...], dim=0)   # (max_action_horizon, max_action_dim)
action_mask = torch.ones_like(normalized_actions)
action_mask[action_horizon:] = 0
action_mask[:, action_dim:] = 0
...
normalized_states = torch.cat([...], dim=-1)
normalized_states = torch.cat([...], dim=-1)   # (T_state, max_state_dim)
...
transformed_inputs["state"] = normalized_states
transformed_inputs["action"] = normalized_actions
transformed_inputs["action_mask"] = action_mask
```

**VLM content creation**

- `Gr00tN1d6Processor._get_vlm_inputs` → `_apply_vlm_processing`
  - Stacks image tensors into `[T*V, C, H, W]` (T time steps, V views).
  - Builds `vlm_content` with a single conversation containing the text and images.

```911:948:src/lerobot/policies/gr00t_n1d6/processor_gr00t_n1d6.py
stacked_images = torch.stack([...], dim=1).flatten(0, 1).numpy()  # (T*V, C, H, W)
vlm_inputs = self._apply_vlm_processing(stacked_images, language)
```

**Collation for VLM tokens**

- `Gr00tN1d6DataCollator.__call__`
  - **Input**: list of per-example dicts (each may contain `vlm_content`, `state`, `action`, `action_mask`).
  - **Output**: `BatchFeature(data={"inputs": batch})` where `batch` contains:
    - `input_ids`, `attention_mask`, `pixel_values` (from `AutoProcessor`).
    - `state`, `action`, `action_mask` (stacked into batch dimension).

```514:543:src/lerobot/policies/gr00t_n1d6/processor_gr00t_n1d6.py
if key == "vlm_content":
    vlm_inputs = self.processor(text=text_list, images=image_inputs, return_tensors="pt", padding=True)
    for k, v in vlm_inputs.items():
        batch[k] = v
...
else:
    batch[key] = torch.from_numpy(np.stack(values))
```

### 3) Policy Forward & Model Loss (Training)

- `Gr00tN1d6Policy.forward` (wrapper)
  - **Input**: `batch` dict with tensors from preprocessor (includes `state`, `action`, VLM tokens).
  - **Output**: `(loss, loss_dict)` where `loss` is a scalar tensor.
  - **Key behaviors**:
    - Pads/truncates `state` and `action` to the checkpoint’s expected dimensions.
    - Creates/aligns `action_mask` if missing.
    - Calls `Gr00tN1d6.forward` under bf16 autocast.

```237:617:src/lerobot/policies/gr00t_n1d6/modeling_gr00t_n1d6.py
groot_inputs = {...}
...
outputs = self._groot_model.forward(groot_inputs)
loss = outputs.get("loss")
return loss, {"loss": loss.item()}
```

- `Gr00tN1d6.forward` → `Gr00tN1d6ActionHead.forward`
  - **Inputs**:
    - Backbone outputs: `backbone_features` `[B, seq_len, backbone_embedding_dim]`
    - Action inputs: `state` `[B, state_dim]`, `action` `[B, action_horizon, action_dim]`,
      `embodiment_id` `[B]`, `action_mask` `[B, action_horizon, action_dim]`
  - **Outputs**:
    - `loss` scalar, `action_loss` tensor, `action_mask`, intermediate features.
  - **Loss**: masked MSE between predicted velocity and target velocity (flow matching).

```206:417:src/lerobot/policies/gr00t_n1d6/gr00t_n1d6.py
state = action_input.state
actions = action_input.action
...
noisy_trajectory = (1 - t) * noise + t * actions
velocity = actions - noise
...
pred = self.action_decoder(model_output, embodiment_id)
pred_actions = pred[:, -actions.shape[1] :]
action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
loss = action_loss.sum() / (action_mask.sum() + 1e-6)
```

## Inference Pipeline: Observation → Action (Normalized → Unnormalized)

### 1) Preprocess Observations (Same Preprocessor)

For inference, the same preprocessor pipeline is used. In `Gr00tN1d6ProcessStep.__call__`, if
`TransitionKey.ACTION` is `None`, the processor switches to eval mode so it can handle missing actions.

```1096:1100:src/lerobot/policies/gr00t_n1d6/processor_gr00t_n1d6.py
if action is None:
    self.processor.eval()
```

### 2) Predict Normalized Actions

- `Gr00tN1d6Policy.predict_action_chunk`
  - **Input**: batch with `state`, VLM tokens, `embodiment_id`.
  - **Output**: normalized action chunk `[B, action_horizon, action_dim]`.
  - **Note**: the action is still normalized and relative; postprocessor handles decoding.

```619:679:src/lerobot/policies/gr00t_n1d6/modeling_gr00t_n1d6.py
outputs = self._groot_model.get_action(groot_inputs)
actions = outputs.get("action_pred")
return actions
```

- `Gr00tN1d6Policy.select_action`
  - Maintains a queue of predicted action chunks and returns a single timestep action.

### 3) Postprocess (Unnormalize + Relative→Absolute + Smoothing)

- `Gr00tN1d6UnnormalizerStep.__call__`
  - **Input**: `PolicyAction` (normalized), plus optional `raw_state` in
    `TransitionKey.COMPLEMENTARY_DATA` for relative→absolute conversion.
  - **Output**: unnormalized action tensor (optionally smoothed).
  - **Behavior**:
    1. Converts `[B, action_dim]` to `[B, 1, action_dim]` for decoding.
    2. Calls `Gr00tN1d6Processor.decode_action` → `StateActionProcessor.unapply_action`.
    3. Optionally truncates to `action_dim` from stats or `output_features`.
    4. Applies simple temporal smoothing (keeps gripper unsmoothed).

```1470:1569:src/lerobot/policies/gr00t_n1d6/processor_gr00t_n1d6.py
decoded_actions = self._processor.decode_action(...)
...
decoded_actions_tensor = torch.cat(action_tensors, dim=-1)
...
smoothed_action = torch.mean(torch.stack(list(self._action_buffer)), dim=0)
smoothed_action[:, 5] = decoded_actions_tensor[:, 5]
new_transition[TransitionKey.ACTION] = smoothed_action
```

## Normalization & Unnormalization Details

### State/Action Normalization (Training & Inference Preprocess)

- `StateActionProcessor.set_statistics` and `_compute_normalization_parameters`
  - Builds `min`, `max`, `mean`, `std` tensors for each joint group.
  - Supports optional percentiles (`q01`/`q99`).

```149:206:src/lerobot/policies/gr00t_n1d6/processor_gr00t_n1d6.py
def _compute_normalization_parameters(self) -> None:
    ...
    self.norm_params[embodiment_tag][modality][joint_group] = {
        "min": min_vals,
        "max": max_vals,
        "dim": np.array(range_vals.shape[0]),
        "mean": mean_vals,
        "std": std_vals,
    }
```

- `StateActionProcessor.apply_state`
  - **Input**: dict of joint_group → raw state arrays.
  - **Output**: dict of normalized state arrays.
  - Strategies: sin/cos encoding, mean/std, or min/max normalization.

```207:260:src/lerobot/policies/gr00t_n1d6/processor_gr00t_n1d6.py
if sin_cos_keys and joint_group in sin_cos_keys:
    normalized_values[joint_group] = apply_sin_cos_encoding(state[joint_group])
elif ... mean/std ...:
    normalized = normalize_values_meanstd(...)
else:
    normalized = normalize_values_minmax(...)
```

- `StateActionProcessor.apply_action`
  - **Input**: dict of joint_group → action arrays, plus optional `state` for relative conversion.
  - **Output**: dict of normalized action arrays.
  - Performs absolute→relative conversion (if configured), then normalizes.

### Action Unnormalization (Postprocess)

- `StateActionProcessor.unapply_action` (called via `Gr00tN1d6Processor.decode_action`)
  - **Input**: dict of normalized action arrays.
  - **Output**: dict of unnormalized (absolute) action arrays.
  - Applies mean/std or min/max unnormalization, then relative→absolute conversion.

## Tensor Shapes Summary (Most Common Path)

These are the typical shapes after preprocessing (single-step state, action horizon `H`,
action dimension `D`, state dimension `S`):

- **Raw dataset item** (`LeRobotDataset.__getitem__`):
  - `observation.state`: `[S]` or `[T_state, S]`
  - `observation.images.<view>`: `[C, H_img, W_img]` (uint8 or float)
  - `action`: `[D]` or `[T_action, D]`

- **After `Gr00tN1d6Processor.__call__`**:
  - `state`: `[T_state, max_state_dim]` (padded)
  - `action`: `[max_action_horizon, max_action_dim]` (training only)
  - `action_mask`: same shape as `action`
  - `vlm_content`: `{text, images, conversation}`

- **After `Gr00tN1d6DataCollator`**:
  - `state`: `[B, T_state, max_state_dim]` (stacked)
  - `action`: `[B, max_action_horizon, max_action_dim]` (training only)
  - `action_mask`: `[B, max_action_horizon, max_action_dim]`
  - `input_ids`, `attention_mask`, `pixel_values`: batch-first (from HF processor)

- **Model forward (action head)**:
  - Backbone features: `[B, seq_len, backbone_embedding_dim]`
  - Action head outputs:
    - `loss`: scalar
    - `action_loss`: `[B, action_horizon, action_dim]`

- **Inference output**:
  - `predict_action_chunk`: `[B, action_horizon, action_dim]` (normalized)
  - `select_action`: `[B, action_dim]` (normalized)
  - After unnormalizer: `[B, action_dim]` (absolute, smoothed)

## Notes & Gotchas

- **Checkpoint padding**: `Gr00tN1d6Policy.forward` pads/truncates `state` and `action` to the
  checkpoint’s expected `max_state_dim` and `max_action_dim`. If your dataset uses smaller
  dims, zeros are appended.
- **Action horizon**: `Gr00tN1d6Processor` pads actions to `max_action_horizon` and constructs
  an `action_mask`. The model loss uses this mask.
- **Relative actions**: When `use_relative_action=True`, actions are converted to relative in
  preprocessing and converted back to absolute in postprocessing (requires `raw_state`).
- **Data sharding**: For streaming datasets, `StreamingLeRobotDataset` shards the dataset via
  `safe_shard(...)`. For non-streaming datasets, `accelerate` handles sharding after
  `accelerator.prepare(...)`.


