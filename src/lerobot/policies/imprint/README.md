# IMPRINT

IMPRINT learns a multi-stream latent memory for a robot policy. The paper is
[Memory Without Curation](https://arxiv.org/abs/2610.14406). The reference code
is Apache-2.0 at [amazon-far/imprint](https://github.com/amazon-far/imprint).
The project page is <https://yilin-wu98.github.io/imprint_website/>.

This package is a bounded PyTorch policy for LeRobot. It follows the paper's
memory update, stage-1 loss, and flow-matching action loss at a small width.
It does not load the paper's Pi0.5 checkpoint, Wan 2.2 tokenizer, or DINOv3 encoder.

## Source map

| Paper or reference name | Code |
| --- | --- |
| `CosmoBottleneckMemory` | `FullAttentionMemory` |
| Language FiLM, skipped when language is missing | `language_condition_queries` |
| Event order `action_proprio_then_finalized_visual_v1` | `chunk_type_ids` |
| Gated delta step and carried state | `gated_delta_step`, `RecurrentMemory.update_incremental` |
| Full event scan equals the incremental scan on dense chunks | `RecurrentMemory.forward_sequence` |
| Stage-1 decoder, then discarded | `ImprintDecoder`, `training_stage="memory"` |
| Stage-1 weights 1, 0.5, 0.5, 0.5 | `memory_pretraining_loss` |
| Action-expert cross-attention plus FiLM | `MemoryModulation` |
| Causal action `a_t = a_{t-1}`, first step zero | dataset `action_delta_indices`, and `causal_action_stream` for a raw executed stream |
| Pi0.5 flow match `x_t = t * noise + (1 - t) * action`, target `noise - action` | `ImprintPolicy._joint_forward` |

The reference `chunkwise_wy` fast path is not implemented. The event-wise scan is the checked path.

## Smallest training step

```python
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.imprint.configuration_imprint import ImprintConfig
from lerobot.policies.imprint.modeling_imprint import ImprintPolicy

config = ImprintConfig(
    device="cpu",
    input_features={
        "observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    },
    output_features={
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    },
)
policy = ImprintPolicy(config)
policy.train()
batch = {
    "observation.images.cam": torch.rand(1, config.history_length, 3, 16, 16),
    "observation.state": torch.randn(1, config.history_length, 6),
    "action": torch.randn(1, config.history_length + config.chunk_size, 6),
}
loss, logs = policy(batch)
loss.backward()
```

CLI: `--policy.type=imprint`.

Set the same camera, state, and action features your dataset provides. The
defaults are a CPU-sized model (`memory_dim=64`, `num_latents=8`). Paper RMBench
uses width 1024 and 256 queries. Those values are comments on the config, not the defaults.

## Batch contract

Joint training (`training_stage="joint"`, the default):

- Images and `observation.state` have time length `history_length`. Index 0 is the oldest frame. The last index is the current frame. `observation_delta_indices` is `1 - history_length ... 0`.
- `action` has time length `history_length + chunk_size`. The first `history_length` steps are the causal actions for that window. The last `chunk_size` steps are the flow-matching target, starting at the current frame.
- `action_delta_indices` is `-history_length ... chunk_size - 1`. A LeRobot dataset that honors those indices builds this tensor. Do not shift it again.

Stage 1 (`training_stage="memory"`):

- Images, state, and `action` all have time length `history_length`.
- `action_delta_indices` is `-history_length ... -1`. That window is already the causal stream.
- `episode_progress` has shape `(B,)` and is the current step divided by the last demonstration step. `lerobot-train` fills it from `frame_index` and the episode length. A hand-built batch must pass it.
- Optional `query_steps` has shape `(B, Q)` and selects past frames to reconstruct. If it is missing, the policy samples `num_query_samples` indices.

After stage 1, keep the memory and drop the decoder:

```python
import copy
from pathlib import Path

from lerobot.policies.imprint.configuration_imprint import ImprintConfig
from lerobot.policies.imprint.modeling_imprint import ImprintPolicy
from lerobot.policies.imprint.processor_imprint import make_imprint_pre_post_processors

stage1_dir = Path("outputs/train/<run>/checkpoints/<step>/pretrained_model")
joint_config = copy.deepcopy(ImprintConfig.from_pretrained(stage1_dir))
joint_config.training_stage = "joint"
joint = ImprintPolicy(joint_config)
report = joint.load_memory_weights(stage1_dir)
joint_init = Path("outputs/imprint_joint_init")
joint.save_pretrained(joint_init)
preprocessor, postprocessor = make_imprint_pre_post_processors(joint.config)
preprocessor.save_pretrained(joint_init)
postprocessor.save_pretrained(joint_init)
```

`report["loaded"]` lists the copied tensors. The decoder keys are in `report["skipped_decoder"]`. The joint action expert and the current-frame projections start at their new initialization. The two processor files are what `lerobot-train --policy.path=outputs/imprint_joint_init` loads.

`observation.language.embedding` has shape `(B, lang_dim)` when you have an external language vector. If the key is missing, language FiLM is skipped. Text strings are ignored.

`select_action` takes one environment step, shape `(B, C, H, W)` or `(B, 1, C, H, W)` for images and `(B, D)` or `(B, 1, D)` for state. Call `reset()` at every episode boundary. The batch size must stay the same until `reset()`. `predict_action_chunk` recomputes memory from the batch and does not change the episode state.

## Limits

- The image stem is a stride-2 convolution and a spatial mean. It is not Wan 2.2 and not DINOv3.
- A finalized chunk is the mean of `chunk_frames` stem tokens. It is not a Wan latent.
- No JAX checkpoint, Orbax loader, or Hugging Face weight file from `Amazon-FAR/imprint_models` is read. `load_memory_weights` copies this package's own stage-1 memory, not those checkpoints.
- Stage 1 and joint training use separate configs. The joint model has no decoder. `load_memory_weights` is the handoff.
- `chunkwise_wy` is not implemented. Long histories in this port use the Python event loop.
- Standard `lerobot-train` supplies the causal actions only when the dataset uses `action_delta_indices`.
- This package does not report RMBench, RoboCasa, or real-robot success rates.
- The paper's training stack is Linux x86_64, Python 3.11, CUDA 12.8, four 80 GB GPUs, frozen Wan 2.2 and DINOv3, and a Pi0.5 flow-matching expert. That run is not launched here.
- The optimizer preset is AdamW at `1e-4` for this small policy. The paper's stage-2 peak is `2.5e-5` on the VLA.
