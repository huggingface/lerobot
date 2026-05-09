# test plan: reward models port (sim only)

branch: feature/reward-models-port
scope: unit + integration + smoke e2e. sim only. hw out of scope.

## dependencies

- `uv pip install -e '.[hilserl,test]'` (gym-hil, pytest)
- sim env: gym_hil/PandaPickCubeGamepad-v0
- tmp dirs via pytest tmp_path for artifacts

## layers

### L1 unit (pure-python, fast, no torch runs)

tests/processor/reward_model/
- `test_base.py`
  - BaseRewardProcessorStep subclass (fake compute_reward=const)
  - reward<1 -> no update. reward=1 -> reward=success_reward, done=True iff terminate_on_success
  - info["reward_classifier_frequency"] populated
  - observation=None -> passthrough

- `test_height_gripper.py`
  - state=[z,g,...] w/ z>=thr & g<thr -> 1.0
  - z>=thr, g>=thr -> 0.0
  - z<thr, g<thr -> 0.0
  - indices z_index, gripper_index respected

- `test_classifier_patch.py`
  - import `lerobot.processor.reward_model` -> patch applied (Classifier.predict_reward is replaced)
  - re-import idempotent (no double patch)
  - patched __init__ accepts arbitrary kwargs

- `test_dispatch.py`
  - `_build_reward_model_step(None)` -> None
  - type=height_gripper -> HeightGripperRewardStep
  - type=cnn w/ pretrained_path=None -> CNNRewardProcessorStep (model=None path)
  - type=sarm w/ pretrained_path=None -> SARMRewardProcessorStep (model=None path OR cfg probe mode)
  - type=unknown -> ValueError

### L2 integration (torch, small fake models)

tests/processor/reward_model/
- `test_cnn_classifier_end_to_end.py`
  - synthesize 10 random 128x128 images labelled by "mean brightness > 0.5" (deterministic)
  - instantiate upstream Classifier w/ tiny backbone (or monkeypatch predict to mock)
  - save to tmp_path via `Classifier.save_pretrained`
  - load via CNNRewardProcessorStep(pretrained_path=tmp)
  - feed transition w/ image -> assert reward in {0, 1}
  - assert info frequency > 0

- `test_sarm_processor_ring_buffer.py`
  - sim SARMRewardModel as torch.nn.Module stub (returns monotonic progress scaled by step)
  - feed a reset, then N steps with random obs
  - assert ring buffer grows then slides
  - assert reward_mode=binary crosses threshold exactly once
  - assert reward_mode=delta = progress_t - progress_{t-1}
  - assert reward_mode=dense = raw progress
  - reset clears buffer

- `test_sarm_window_padding.py`
  - assert inference window = real past + replicated current frames when t < n_obs_steps
  - frame_index default = n_obs_steps // 2

- `test_split_dataset.py`
  - build tiny LeRobotDataset in tmp (5 ep, 20 frames ea)
  - call split_dataset(val_stride=4)
  - assert train+val non-empty, disjoint frames, preserves stats (stats_dataset_repo_id note)

### L3 gym_hil sim integration

tests/rl/
- `test_gym_hil_env_smoke.py`
  - make env via JSON cfg (headless, no teleop, no reward_model)
  - env.reset() -> obs has "observation.state", at least one image key
  - 5 env.step(random) -> returns (obs, r, term, trunc, info) w/ expected shapes
  - mark skipif `gym_hil` not importable

- `test_gym_hil_with_height_gripper.py`
  - load env + reward_model cfg (type=height_gripper w/ sim-appropriate indices)
  - run 50 random steps -> reward field present, integer in [0,1]
  - threshold chosen so some trajectories hit reward=1 (probe indices first)
  - skipif gym_hil missing

- `test_gym_hil_with_stub_cnn.py`
  - build stub Classifier that returns 1.0 iff mean brightness > thr
  - save to tmp, plug via RewardClassifierProcessorStep (fork's existing) OR new CNN step
  - 20 random steps -> reward sometimes 1

- `test_gym_hil_with_stub_sarm.py`
  - stub SARMRewardModel returning `min(step / 20, 1.0)` as progress
  - plug via SARMRewardProcessorStep w/ reward_mode=delta
  - 40 steps -> assert reward sum ≈ 1.0 (telescoping)

### L4 end-to-end HIL-SERL training smoke

tests/rl/
- `test_hil_serl_train_no_reward_model_smoke.py` [AUTONOMOUS RUN, NOT IN PYTEST]
  - launch actor+learner via fork's lerobot_train_hil_serl cfg
  - no reward_model cfg (baseline, reward from env)
  - run ~60 s or ~200 learner steps
  - assert: log stream contains loss steps, no crash, wandb dryrun or disabled
  - PURPOSE: verify fork's HIL-SERL pipeline works before porting

- `test_hil_serl_train_with_stub_sarm_smoke.py` [AUTONOMOUS RUN]
  - same as above but plug reward_model type=sarm with stub checkpoint
  - run ~60 s
  - assert: reward field observed in actor logs, SARM step not raising

## synthetic demo data generation

tests/fixtures/
- `make_tiny_classifier_dataset.py`
  - gym_hil rollout w/ scripted policy (gravity+open gripper) for 3 success eps + 3 fail eps
  - save as LeRobotDataset (local dir, not hub)
  - reward label = gym_hil env's own reward at final frame (binary)
  - used by CNN classifier training smoke

- `make_tiny_sarm_dataset.py`
  - gym_hil rollout w/ random policy, pad eps to 30 frames, label task="lift_cube"
  - save 10 eps as LeRobotDataset
  - used by sarm training smoke

train smoke (optional, may skip if slow):
- `test_cnn_classifier_train_smoke.py`
  - run `lerobot_train_reward_classifier.py --dataset=<tiny> --steps=50 --batch=4`
  - load resulting checkpoint via CNNRewardProcessorStep
  - assert predict_reward runs w/o AttributeError
- `test_sarm_train_smoke.py`
  - run `lerobot-train policy=sarm dataset=<tiny> steps=20 batch=2 annotation_mode=single_stage`
  - load via SARMRewardProcessorStep, predict on a stub obs window
  - assert reward in [0, 1]

## env probe (run once, record findings in this doc)

before L3: run `uv run python -c "import gym_hil; import gymnasium as gym; env=gym.make('gym_hil/PandaPickCubeGamepad-v0'); o,_=env.reset(); print(list(o.keys())); print({k: v.shape if hasattr(v,'shape') else v for k,v in o.items()})"`
- capture obs key names (esp image keys, state dim) → populate sim cfg JSONs accordingly
- update height_gripper indices

## test commands

```
uv run pytest tests/processor/reward_model/ -v
uv run pytest tests/rl/test_gym_hil_*.py -v
# autonomous smoke (not pytest):
uv run lerobot_train_hil_serl ... (from fork) --config rl/gym_hil_serl_env.json --max-steps 200
```

## success criteria

- all L1 + L2 green
- L3 green when gym_hil installed
- L4 baseline smoke (no reward model) green
- L4 SARM smoke green (no crash, reward field populated)
- cnn + sarm train smoke produces valid checkpoints (if run)

## deferred / nice-to-have

- real model training benchmarks (slow, not in port scope)
- hw tests (out of scope)
- performance/latency benchmarks
