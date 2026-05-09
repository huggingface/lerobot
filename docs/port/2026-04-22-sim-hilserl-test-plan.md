# test plan: sim HIL-SERL 3 scenarios

branch: feature/reward-models-port
scope: sim-only, headless-safe (MUJOCO_GL=egl, SDL_VIDEODRIVER=dummy)

## layers

### L1 unit: env adapter
tests/envs/test_sim_assembling.py
- gym.make('sim_assembling/AssembleBase-v0', ...) returns non-None env
- env.reset() emits `observation.state` (shape 15,float32), `observation.images.front` (3,128,128) uint8/float, `observation.images.wrist` (3,128,128)
- action space = Box(low=-1, high=1, shape=(4,)) (3D delta + gripper discrete → last dim)
- env.step(zeros_action) passes + obs keys stable
- fast mode: 20 steps < 2x wall-clock-of-equivalent-realtime

### L1 unit: dualsense teleop
tests/teleoperators/test_dualsense_sim_headless.py
- no controller connected (SDL_VIDEODRIVER=dummy): adapter skips connect + emits zero action, no events (graceful)
- stub a fake PS4Joystick via monkeypatch; verify `get_teleop_events()` returns dict w/ keys `is_intervention,success,terminate_episode,rerecord_episode`
- edge-trigger: after button press, event appears once then clears

### L1 unit: reward dispatcher on sim cfg
tests/rl/test_sim_reward_wiring.py (parametric, 3 types)
- reward_model.type=manual → no reward step in pipeline
- reward_model.type=height_gripper → HeightGripperRewardStep attached (uses state idx 14 for z, idx  for gripper — TBD depending on adapter layout)
- reward_model.type=cnn / sarm (no ckpt) → step built, model None, no crash

### L2 integration: sim rollout w/ reward step
tests/rl/test_sim_rollout_with_reward.py
- 30-step random rollout via the adapted env w/ HeightGripperRewardStep (impossible thresholds → reward=0 everywhere)
- rollout w/ SARMRewardProcessorStep(pretrained_path=None) → progress=0, but pipeline doesn't crash
- rollout w/ CNNRewardProcessorStep(pretrained_path=None) → reward=0, pipeline ok

### L2 integration: synthetic dataset generation
tests/fixtures/sim_synth_dataset.py + tests/rl/test_sim_synth_dataset.py
- fixture script produces:
  - 3 success eps (scripted: move toward bottom, close gripper, move up, press "success"=reward+done=True)
  - 3 failure eps (scripted: random walk, no success)
- write to tmp LeRobotDataset (local root, no hub push)
- assert: dataset.num_episodes=6, reward range {0, 1}, done=True only at episode terminals
- (optional slow) run `split_dataset` + `lerobot-train-reward-classifier --steps=5` smoke OR skip this — just verify the pipeline scripts are callable
- (optional slow) SARM annotated-split via `prepare_sarm_data` smoke — mark slow, skip in CI

### L3 end-to-end smoke: 3 scenarios
tests/rl/test_sim_three_scenarios_smoke.py (marked slow/autonomous)
each test: build env from the respective JSON cfg, run 10 env.step() calls w/ random action
- scenario manual: reward_model.type=manual → rollout succeeds
- scenario cnn: reward_model.type=cnn, pretrained_path=None (stub) → rollout succeeds, reward is 0
- scenario sarm: reward_model.type=sarm, pretrained_path=None (stub) → rollout succeeds, reward is 0

### L4 training loop smoke (optional, SKIP by default)
actor+learner via gRPC is heavy — skip. Validated indirectly by L3 + existing fork test_actor_learner tests.

## synth dataset strategy

scripted policy for sim (no real gamepad needed):
- success policy: for ep=0..N:
  - phase 1 (steps 0..20): hover above `bottom` obj (use env's initial obs + `objects.bottom.pos`)
  - phase 2 (steps 20..30): descend to bottom obj z + 0.02
  - phase 3 (steps 30..40): close gripper (discrete action=2)
  - phase 4 (steps 40..60): move up
  - phase 5 (final): mark reward=1, done=True
- failure policy: random xyz deltas, no reward/done signal

all scripted writes skip the gamepad entirely — demo recording happens via a lerobot-record-like internal loop. Keep this a fixture, NOT a replacement for the real teleop.

## success criteria

- all L1 green
- L2 integration green (rollout + synth dataset)
- L3 smoke green for 3 scenarios
- no regressions in prior 36 reward-model tests
- no modifications to simulator_for_IL_RL
- commits (src only) build & install cleanly

## env flags for test runner

```
MUJOCO_GL=egl SDL_VIDEODRIVER=dummy uv run pytest tests/envs/test_sim_assembling.py tests/teleoperators/test_dualsense_sim_headless.py tests/rl/test_sim_*.py -v
```

## deferred / nice-to-have

- reward convergence benchmarks
- SARM training convergence on sim (very slow, out of scope)
- real-DualSense hardware test (needs user, manual)
