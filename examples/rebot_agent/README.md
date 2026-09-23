# Steerable ReBot policy with Astra planning

This experiment uses LeRobot main's datasets, language recipes, WALL-OSS-Flow,
processors, rollout strategies, and `/autosteer` controller. Astra chooses a steering
instruction from images and recent observation/command history. The VLA produces
all robot actions. There is no separate hybrid runtime or direct-action tool service.

See [the guide](../../docs/source/steerable_rebot.mdx) for the paper mapping, data
contract, annotation review, training, and physical evaluation. The reusable research
and deployment goal is [astra_goal.md](astra_goal.md).

## Train on the science cluster

Connect with `sft ssh hpc-cluster-science-login-81-129`, obtain a single-node scheduler
allocation of **at most four H100s**, then install `uv sync --locked --extra training
--extra wallx`. This launcher runs inside that allocation; it does not allocate GPUs.

```bash
# Inspect configuration without loading a model.
uv run examples/rebot_agent/train_wall_oss_flow.py --gpus 4 --smoke --dry-run --output outputs/wall_preview
# Semantic baseline: 80% current subtask / 20% overall task.
uv run examples/rebot_agent/train_wall_oss_flow.py --gpus 4 --smoke --output outputs/wall_smoke
# Richer model: 80% uniformly sampled reviewed commands / 20% task.
uv run examples/rebot_agent/train_wall_oss_flow.py --gpus 4 --steering-manifest /path/to/reviewed/steering_manifest.json --output outputs/wall_steerable
```

Use a fresh output directory for each launch. Check finite losses, memory, held-out
loss, and checkpoint reload after the smoke test before launching the full run.
`training.json` pins the dataset and native base-model revision and configures WALL-OSS-Flow, using native
`policy.pretrained_name_or_path` initialization. The 14 ReBot dimensions are padded
and masked internally. The full run defaults to 20,000 steps and batch one per GPU;
these are starting settings that still require hardware validation.

## Use Astra through the language runtime

Add these options to your existing, calibrated `lerobot-rollout` command, keeping
its robot ports/cameras and trained `--policy.path`:

```bash
--interactive=true --inference.type=sync --interpolation_multiplier=1 \
--planner.enabled=true --planner.model=gpt-6-astra \
--planner.camera_keys='["base","left_wrist","right_wrist"]' \
--planner.styles='["task","subtask"]' \
--planner.log_path=outputs/rebot_planner/decisions.jsonl \
--autosteer_interval_s=2
```

Set `OPENAI_API_KEY` on the robot host. Use an API model ID available to that account;
model access has not been tested here. Camera keys refer to processed robot
observations. After `/start`, the external planner keeps the VLA idle until you enter:

```text
/autosteer Put the white tape roll into the black bin.
```

Astra observes and emits a command, the VLA acts, and Astra observes again. Extend
`planner.styles` to `motion`, `point`, and `combination` only for a checkpoint
trained and validated on those styles. Following the paper's final planner, leave
`trace` disabled by default; use it only in controlled, validated experiments.
`/subtask <text>` switches to human language
steering; `/reset` ends the segment; `/stop` closes the session. Planner failures,
uncertainty, or a reported completion hold action production; resume with an explicit
new `/autosteer` or `/subtask`. Completion is an assessment, not a verified success label.

The initial external-planner adapter uses synchronous inference: robot action
production waits while the API call runs. It reuses main's queue invalidation and
language switching. The interval is seconds of execution, not the paper's exact
20-step cadence. Measure latency and command compliance before tuning it.
