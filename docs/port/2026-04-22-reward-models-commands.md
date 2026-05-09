# end-to-end cmds: manual / binary-classifier / SARM HIL-SERL pipelines

branch: feature/reward-models-port
sim env: `gym_hil/PandaPickCubeGamepad-v0` (needs `MUJOCO_GL=egl` for headless)
extras: `uv pip install -e '.[hilserl,test]'`

## common prep (once)

```bash
# pin display for offscreen render in mujoco (put in .zshrc if long-lived)
export MUJOCO_GL=egl

# (optional) login to HF if pushing demos to hub
huggingface-cli login
```

## scenario 1 — manual (teleop SUCCESS btn)

simplest path. reward comes from gamepad SUCCESS btn pressed during teleop.
no classifier, no training. env config: `src/lerobot/rl/gym_hil_manual_env.json`.

```bash
# 1. record demos while teleop-ing (press SUCCESS btn at end of successful ep)
lerobot-record \
    --config_path=src/lerobot/rl/gym_hil_manual_env.json \
    --dataset.repo_id=<your-hf-user>/gym_hil_pick_manual_demos \
    --dataset.num_episodes_to_record=20 \
    --dataset.push_to_hub=false

# 2. launch HIL-SERL training (actor + learner, same config, diff modes).
#    open 2 terminals.

# term A — learner
lerobot-train \
    --config_path=src/lerobot/rl/train_hil_serl_env.json \
    --env.processor.reward_model.type=manual \
    --dataset.repo_id=<your-hf-user>/gym_hil_pick_manual_demos

# term B — actor (same cfg, diff role; see actor_cli)
python -m lerobot.rl.actor \
    --config_path=src/lerobot/rl/train_hil_serl_env.json \
    --env.processor.reward_model.type=manual \
    --dataset.repo_id=<your-hf-user>/gym_hil_pick_manual_demos
```

notes:
- `reward_model.type=manual` disables the dispatcher. reward is set by env + teleop SUCCESS button via `InterventionActionProcessorStep`.
- if using existing `gym_hil_serl_env.json` w/o `reward_model` block, same behaviour (manual is default).

## scenario 2 — binary image classifier (CNN)

offline train classifier on collected demos, then plug into HIL-SERL.

```bash
# 1. record success demos (same cmd as scenario 1)
lerobot-record \
    --config_path=src/lerobot/rl/gym_hil_manual_env.json \
    --dataset.repo_id=<you>/gym_hil_pick_cnn_success \
    --dataset.num_episodes_to_record=20

# 2. record failure demos (same thing, no SUCCESS btn press)
lerobot-record \
    --config_path=src/lerobot/rl/gym_hil_manual_env.json \
    --dataset.repo_id=<you>/gym_hil_pick_cnn_failure \
    --dataset.num_episodes_to_record=20

# 3. (optional) crop+resize if cameras aren't 128x128
lerobot-edit-dataset crop-roi \
    --repo-id <you>/gym_hil_pick_cnn_success \
    --resize 128 128 \
    --output-repo-id <you>/gym_hil_pick_cnn_success_cropped

# 4. split classifier dataset train/val (frame stride)
lerobot-split-reward-dataset \
    --src-repo-id <you>/gym_hil_pick_cnn_success_cropped \
    --val-stride 4

# 5. train classifier (wrapper applies patch)
lerobot-train-reward-classifier \
    --policy.type=reward_classifier \
    --dataset.repo_id=<you>/gym_hil_pick_cnn_success_cropped-train \
    --output_dir=outputs/gym_hil_cnn_reward \
    --job_name=gym_hil_cnn_reward \
    --steps=2000 --batch_size=32

# 6. sanity-check: loading + predicting on single frame
python -c "
from lerobot.processor.reward_model import CNNRewardConfig, CNNRewardProcessorStep
step = CNNRewardProcessorStep(config=CNNRewardConfig(
    pretrained_path='outputs/gym_hil_cnn_reward/checkpoints/last/pretrained_model',
    device='cuda', success_threshold=0.7))
print('classifier loaded:', step._classifier is not None)
"

# 7. edit gym_hil_cnn_env.json -> set pretrained_path to the checkpoint above

# 8. launch HIL-SERL w/ CNN reward (two terminals, same as scenario 1)
lerobot-train \
    --config_path=src/lerobot/rl/train_hil_serl_env.json \
    --env.processor.reward_model.type=cnn \
    --env.processor.reward_model.pretrained_path=outputs/gym_hil_cnn_reward/checkpoints/last/pretrained_model \
    --env.processor.reward_model.success_threshold=0.7 \
    --env.processor.reward_model.device=cuda \
    --dataset.repo_id=<you>/gym_hil_pick_cnn_success_cropped

python -m lerobot.rl.actor \
    --config_path=src/lerobot/rl/train_hil_serl_env.json \
    --env.processor.reward_model.type=cnn \
    --env.processor.reward_model.pretrained_path=outputs/gym_hil_cnn_reward/checkpoints/last/pretrained_model \
    --env.processor.reward_model.success_threshold=0.7 \
    --env.processor.reward_model.device=cuda \
    --dataset.repo_id=<you>/gym_hil_pick_cnn_success_cropped
```

## scenario 3 — SARM (stage-aware reward model)

needs both success + failure demos, annotation, SARM training, then relabel offline buffer.

```bash
# 1. record demos (same as scenario 2: success + failure sets)
lerobot-record --config_path=src/lerobot/rl/gym_hil_manual_env.json \
    --dataset.repo_id=<you>/gym_hil_pick_sarm_success \
    --dataset.num_episodes_to_record=30
lerobot-record --config_path=src/lerobot/rl/gym_hil_manual_env.json \
    --dataset.repo_id=<you>/gym_hil_pick_sarm_failure \
    --dataset.num_episodes_to_record=30

# 2. crop+resize to 128x128 (skip if already that size)
lerobot-edit-dataset crop-roi --repo-id <you>/gym_hil_pick_sarm_success --resize 128 128 --output-repo-id <you>/gym_hil_pick_sarm_success_cropped
lerobot-edit-dataset crop-roi --repo-id <you>/gym_hil_pick_sarm_failure --resize 128 128 --output-repo-id <you>/gym_hil_pick_sarm_failure_cropped

# 3. merge + annotate + split (dense_only mode w/ idle/task subtasks)
lerobot-prepare-sarm-data \
    --success-repo-id <you>/gym_hil_pick_sarm_success_cropped \
    --failure-repo-id <you>/gym_hil_pick_sarm_failure_cropped \
    --output-repo-id <you>/gym_hil_pick_sarm_combined \
    --val-stride 4

# 4. train SARM (single_stage if 1-stage task, else dense_only)
lerobot-train \
    --policy.type=sarm \
    --policy.annotation_mode=single_stage \
    --dataset.repo_id=<you>/gym_hil_pick_sarm_combined-train \
    --output_dir=outputs/gym_hil_sarm \
    --job_name=gym_hil_sarm \
    --steps=5000 --batch_size=16

# 5. relabel offline buffer w/ trained SARM (mode MUST match online!)
#    use delta for SAC-friendly potential-based shaping.
lerobot-relabel-sarm \
    --src-repo-id <you>/gym_hil_pick_sarm_success_cropped \
    --sarm-checkpoint outputs/gym_hil_sarm/checkpoints/last/pretrained_model \
    --reward-mode delta \
    --task pick_cube \
    --device cuda

# 6. sanity: inference on stub obs
python -c "
from lerobot.processor.reward_model import SARMRewardConfig, SARMRewardProcessorStep
step = SARMRewardProcessorStep(config=SARMRewardConfig(
    pretrained_path='outputs/gym_hil_sarm/checkpoints/last/pretrained_model',
    device='cuda', task='pick_cube', reward_mode='delta',
    stats_dataset_repo_id='<you>/gym_hil_pick_sarm_success_cropped'))
print('SARM loaded:', step._model is not None)
"

# 7. launch HIL-SERL w/ SARM (two terminals)
lerobot-train \
    --config_path=src/lerobot/rl/train_hil_serl_env.json \
    --env.processor.reward_model.type=sarm \
    --env.processor.reward_model.pretrained_path=outputs/gym_hil_sarm/checkpoints/last/pretrained_model \
    --env.processor.reward_model.task=pick_cube \
    --env.processor.reward_model.reward_mode=delta \
    --env.processor.reward_model.success_threshold=0.9 \
    --env.processor.reward_model.stats_dataset_repo_id=<you>/gym_hil_pick_sarm_success_cropped \
    --env.processor.reward_model.device=cuda \
    --dataset.repo_id=<you>/gym_hil_pick_sarm_success_cropped_sarm_delta

python -m lerobot.rl.actor \
    --config_path=src/lerobot/rl/train_hil_serl_env.json \
    --env.processor.reward_model.type=sarm \
    --env.processor.reward_model.pretrained_path=outputs/gym_hil_sarm/checkpoints/last/pretrained_model \
    --env.processor.reward_model.task=pick_cube \
    --env.processor.reward_model.reward_mode=delta \
    --env.processor.reward_model.success_threshold=0.9 \
    --env.processor.reward_model.stats_dataset_repo_id=<you>/gym_hil_pick_sarm_success_cropped \
    --env.processor.reward_model.device=cuda \
    --dataset.repo_id=<you>/gym_hil_pick_sarm_success_cropped_sarm_delta
```

## gotchas

- **reward_mode MUST match online & offline** — if you relabel `--reward-mode delta`, you MUST set `reward_model.reward_mode=delta` online. mismatch => SAC Q-target dist shift => bad training.
- **`task` text MUST match SARM training** — empty string is valid; any mismatch => CLIP OOD => near-zero progress.
- **`stats_dataset_repo_id`** — point at source dataset w/ populated `meta.stats`. a split dataset typically has empty stats => normalizer identity => SARM sees OOD state => near-zero output.
- **`MUJOCO_GL=egl`** — required on headless machines. GLFWError in logs => missing.
- **gym_hil gamepad env** requires pygame + X (for teleop). if running only sim+random actor for smoke, use `PandaPickCubeBase-v0` (no gamepad wrapper).
- **gRPC actor+learner** — two processes. same cfg, actor reads env + sends transitions, learner optimizes + streams params. port via `--server.port` if default is busy.

## quick sanity tests (no training run)

```bash
# all reward-model unit tests (28+)
MUJOCO_GL=egl uv run pytest tests/processor/reward_model/ -v

# sim integration (gym_hil env + reward step plug)
MUJOCO_GL=egl uv run pytest tests/rl/test_gym_hil_integration.py tests/rl/test_gym_manipulator_reward_wiring.py -v
```

## debugging checklist (SARM returns 0)

1. `task` string matches exactly? (`--task ''` is valid if trained w/ empty)
2. `stats_dataset_repo_id` points at ds w/ populated `meta.stats`? (not a split)
3. CLIP text embedding cached? (check log: `SARM: CLIP text encoding cached`)
4. `reward_mode` matches offline relabel mode?
5. image key matches training? (default `observation.images.front`)
6. `eval_every_n_steps`=1 for debugging (sync every step, no async lag)
