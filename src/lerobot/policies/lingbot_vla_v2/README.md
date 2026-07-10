# LingBot-VLA 2.0

`lingbot_vla_v2` is the LeRobot policy wrapper for LingBot-VLA 2.0. It combines a
Qwen3-VL vision-language backbone with a sparse MoE Qwen2 action expert and flow-matching
continuous action generation over the canonical 55-D robot state/action space.

Use this policy through the standard LeRobot interfaces: `lerobot-train`,
`make_policy_config`, `make_policy`, `from_pretrained`, and `predict_action_chunk` /
`select_action`.

## Install

Install LeRobot with the optional LingBot-VLA v2 dependencies:

```bash
pip install -e ".[lingbot-v2]"
```

The default config expects a Qwen3-VL processor/tokenizer and a LingBot-VLA v2 checkpoint:

```text
Qwen/Qwen3-VL-4B-Instruct
robbyant/lingbot-vla-v2-6b
```

Use local paths with `--policy.processor_path`, `--policy.tokenizer_path`, or
`--policy.pretrained_name_or_path` when running offline.

## Train

Use the normal LeRobot training CLI and select the policy with `--policy.type`:

```bash
lerobot-train \
  --dataset.repo_id=<repo_id> \
  --dataset.root=<dataset_root> \
  --policy.type=lingbot_vla_v2 \
  --policy.robot_config_path=<robot_config.yaml> \
  --policy.norm_stats_path=<norm_stats.json> \
  --policy.processor_path=<qwen3_vl_processor_or_model_path> \
  --policy.tokenizer_path=<qwen3_vl_processor_or_model_path> \
  --policy.device=cuda \
  --batch_size=1 \
  --steps=5000 \
  --save_freq=2500 \
  --output_dir=outputs/train/lingbot_vla_v2
```

The robot config maps dataset keys into the canonical LingBot slots. The norm-stats JSON is
used by the LingBot feature transform, so the saved LeRobot processor pipeline does not use
the generic LeRobot normalizer/unnormalizer steps.

## Resume

Resume from a saved LeRobot checkpoint by passing the checkpoint's `train_config.json`:

```bash
lerobot-train \
  --resume=true \
  --config_path=outputs/train/lingbot_vla_v2/checkpoints/005000/pretrained_model/train_config.json \
  --steps=30000 \
  --save_freq=5000 \
  --output_dir=outputs/train/lingbot_vla_v2_resume
```

## Load A Policy

```python
from lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 import LingbotVLAV2Policy

policy = LingbotVLAV2Policy.from_pretrained(
    "outputs/train/lingbot_vla_v2/checkpoints/005000/pretrained_model"
)
policy.to("cuda").eval()
```

For batched LeRobot observations, use `select_action`. For open-loop action chunks, use
`predict_action_chunk`; it returns a `(batch, chunk_size, action_dim)` tensor after the
policy postprocessing path.

## Tests

Run the lightweight registration and config tests:

```bash
pytest -q tests/policies/lingbot_vla_v2/test_lingbot_vla_v2.py
```

Run the feature-transform tests when a local Qwen3-VL processor is available:

```bash
export LINGBOT_VLA_V2_QWEN3VL=/path/to/Qwen3-VL-4B-Instruct
pytest -q tests/policies/lingbot_vla_v2/test_feature_transform.py
```

Run the optional Triton grouped-MoE parity test on a CUDA machine with Triton installed:

```bash
pytest -q tests/policies/lingbot_vla_v2/test_triton_moe.py
```

## Implementation Notes

The Qwen3-VL backbone adaptation and sparse-MoE action expert are vendored from the upstream
LingBot-VLA 2.0 implementation and Hugging Face Transformers, with Apache-2.0 license headers
retained. FlashAttention is optional. The default attention implementation is eager; `sdpa`,
`fa2`, `flex`, and `flex_cached` can be selected through the policy config when the runtime
supports them.

For MoE inference, the fused expert path tries the optional upstream Triton kernel first, then
the in-tree Triton grouped-GEMM backend, and finally the grouped-by-expert eager fallback. The
training path uses the eager fallback for autograd stability.
