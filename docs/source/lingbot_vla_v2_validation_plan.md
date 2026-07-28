# LingBot-VLA 2.0 Upstream Checkpoint Validation Plan

This checklist tracks the remaining work for PR #3967 after the reviewer asked
whether `robbyant/lingbot-vla-v2-6b` can be loaded directly in LeRobot and whether
the port has been checked against a known-good pretrained model.

## Implementation Checklist

- [x] Use the standard LeRobot pretrained entry point: `--policy.path=robbyant/lingbot-vla-v2-6b`
  without also passing `--policy.type`.
- [x] Add raw upstream checkpoint detection for `robbyant/lingbot-vla-v2-6b` and local directories
  with sharded safetensors/index files.
- [x] Implement `LingbotVLAV2Policy.from_pretrained(...)` so it keeps the existing LeRobot
  checkpoint path and adds a raw upstream checkpoint path.
- [x] Implement a small key-remap/filter layer for raw upstream tensors.
- [x] Fail loudly on missing keys, unexpected keys outside the documented ignore list, or any
  tensor shape mismatch.
- [x] Keep allowed skipped tensors limited to disabled distillation/depth/align heads when
  `use_depth=False`.
- [x] Make raw upstream processor construction work when `make_pre_post_processors(...)` receives
  `pretrained_path`, instead of trying to load serialized LeRobot processor JSON from the raw
  upstream checkpoint directory.
- [x] Rename the optional dependency extra from `lingbot-v2` to `lingbot_vla2` in code and docs.
- [x] Remove confirmed dead v1-era code in `modeling_lingbot_vla_v2_base.py` if it is not needed
  by the v2 LeRobot wrapper.
- [x] Update public docs to explain raw upstream checkpoints versus LeRobot-saved checkpoints.

## CI-Light Tests

- [x] Add a pure CPU unit test for raw checkpoint detection.
- [x] Add a pure CPU unit test for key remapping.
- [x] Add a pure CPU unit test proving allowed distillation/depth/align tensors are filtered.
- [x] Add a pure CPU unit test proving unknown unexpected tensors raise an error.
- [x] Add a pure CPU unit test proving missing required model tensors raise an error.
- [x] Add a processor factory unit test proving a raw upstream checkpoint path builds fresh
  LingBot processors instead of loading serialized processor JSON.
- [x] Run:

```bash
ruff check --config pyproject.toml src/lerobot/policies/lingbot_vla_v2 tests/policies/lingbot_vla_v2
ruff format --check --config pyproject.toml src/lerobot/policies/lingbot_vla_v2 tests/policies/lingbot_vla_v2
pytest -q tests/policies/lingbot_vla_v2
```

## A100 Heavy Tests

- [x] Inspect the real upstream checkpoint config and make the raw loader use the checkpoint's
  architecture values, not generic training defaults.
- [x] Load the local upstream checkpoint:

```bash
LingbotVLAV2Policy.from_pretrained(
    "/home/liuyue/lvla_scratch/lingbot-vla-v2-6b",
    config=config,
    local_files_only=True,
)
```

Acceptance:

- [x] zero shape mismatches.
- [x] zero missing required tensors.
- [x] all unexpected tensors are from the documented disabled depth/distillation/align heads.
- [x] log output clearly says the raw upstream checkpoint path was used.

## LeRobot Runtime Tests

- [x] Start a one-step train smoke with the raw upstream checkpoint through standard CLI:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --dataset.repo_id=TommyZihao/lerobot_zihao_shake_hands_full \
  --dataset.root=/home/liuyue/lerobot_zihao_shake_hands_full \
  --policy.path=/home/liuyue/lvla_scratch/lingbot-vla-v2-6b \
  --policy.robot_config_path=/home/liuyue/lvla_scratch/configs/so101_robot_config.yaml \
  --policy.norm_stats_path=/home/liuyue/lvla_scratch/configs/so101_norm_stats.json \
  --policy.processor_path=/home/liuyue/lvla_scratch/Qwen3-VL-4B-Instruct \
  --policy.tokenizer_path=/home/liuyue/lvla_scratch/Qwen3-VL-4B-Instruct \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.attention_implementation=eager \
  --policy.loss_type=L1_fm \
  --batch_size=1 \
  --steps=1 \
  --save_checkpoint=false \
  --num_workers=0 \
  --wandb.enable=false
```

Acceptance:

- [x] reaches `step:1`.
- [x] prints a finite loss.
- [x] confirms raw upstream weights were loaded.

- [ ] Run an inference smoke on one dataset frame:
  - [ ] processor produces model-ready tensors.
  - [ ] `select_action` or `predict_action_chunk` returns finite actions.
  - [ ] `_postprocess_actions()` maps actions back to raw LeRobot action dimensions.

- [ ] Save after a tiny run and reload the saved LeRobot checkpoint:
  - [ ] saved checkpoint reloads through the normal LeRobot path.
  - [ ] saved checkpoint does not run the raw upstream key remapper.

## Parity Tests Against Upstream

- [ ] Re-run fixed-input parity using weights loaded through the new
  `LingbotVLAV2Policy.from_pretrained(raw_upstream_checkpoint)` path.
- [ ] Compare prefix embeddings.
- [ ] Compare Qwen3-VL KV cache.
- [ ] Compare suffix/state/action/time embeddings.
- [ ] Compare MoE router logits and top-k expert selection.
- [ ] Compare final action or training loss under fp32/eager.
- [ ] Document any remaining fused/Triton MoE numerical delta as a kernel precision difference.

## Reviewer Reply Requirements

- [ ] State that `--policy.path=robbyant/lingbot-vla-v2-6b` is the supported raw upstream
  checkpoint entry point.
- [ ] State whether the PR still vendors custom dual-stream forward code and why.
- [ ] List the dead-code removal.
- [ ] List the optional-extra rename to `lingbot_vla2`.
- [ ] Report CI-light test results.
- [ ] Report A100 raw-checkpoint load and one-step train smoke results.
- [ ] Be explicit that full benchmark rollout is only claimed if matching benchmark assets were
  actually run.
