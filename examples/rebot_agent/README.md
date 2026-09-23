# ReBot physical agent experiment

Start with [the physical agent guide](../../docs/source/physical_agent_loop.mdx).
`session.json` pins the annotated dataset and defines WALL-OSS-Flow, SmolVLA, and Pi0.5 candidates.
`steerable_80_20.yaml` trains actions on subtask instructions (80%) and overall tasks
(20%) through LeRobot's existing weighted recipe renderer.

The selected first model is **WALL-OSS-Flow**: `/train wall_oss_flow_80_20_v1`.
Install the `wallx` and `training` extras on the training host. Its native base loads
with `policy.type=wall_x` and `policy.pretrained_name_or_path`, using the candidate's
`policy_type` field. A trained LeRobot checkpoint uses the ordinary `base_model` path
without this field. This candidate replaces WALL-X's default text-target recipe with
a low-level task pass-through; the dataset's 80/20 recipe selects the action instruction.
The 14 ReBot
dimensions are padded/masked to 20 internally and action outputs remain 14-dimensional.
The example starts with batch size one; profile the actual host before increasing it.

For the science cluster, enter through `sft ssh hpc-cluster-science-login-81-129`
and use its scheduler to allocate at most four H100 GPUs on one node. Install the
environment with `uv sync --locked --extra training --extra wallx`. The launcher below
runs inside an existing allocation; it does not submit a scheduler job or select a
partition. From the repository root:

```bash
# Inspect locally without loading a model or using GPUs.
uv run examples/rebot_agent/train_wall_oss_flow.py --gpus 4 --smoke --dry-run --output outputs/rebot_wall_preview
# Inside the allocated GPU job: ten steps, held-out validation, and a checkpoint save.
uv run examples/rebot_agent/train_wall_oss_flow.py --gpus 4 --smoke --output outputs/rebot_wall_smoke
# After checking finite losses, memory use, and checkpoint reload, launch the full run.
uv run examples/rebot_agent/train_wall_oss_flow.py --gpus 4 --output outputs/rebot_wall_v1
```

Use a fresh output directory for each launch. The launcher records its configuration,
code revision, tracked diff, and argument vector, and propagates training failures to
the scheduler. It uses DDP and BF16 with a per-GPU batch size of one (effective batch
four on four GPUs); the smoke test must establish whether this fits the allocated
hardware. GPU execution and checkpoint reload still require validation on the cluster.

Run `lerobot-rollout --agent_config=examples/rebot_agent/session.json` on the host
that will train or control the robot. Training can start before any checkpoint exists.
Before a physical rollout, replace the arm/camera placeholders and register a trained
checkpoint. The checked-in file does not contain installation calibration or credentials.

[astra_goal.md](astra_goal.md) is the reusable system/goal prompt for the complete
physical improvement loop. It also directs Astra to adapt Steerable Policies to our
data using the ReBot URDF, forward kinematics, and multiple aligned semantic, motion,
gripper, and calibrated visual command streams. This is an implementation objective;
the prompt itself does not generate those annotations. Paste it into an external
agent's goal, or configure the built-in live supervisor with:

```json
"supervisor": {
  "model": "gpt-6-astra",
  "prompt_path": "examples/rebot_agent/astra_goal.md",
  "timeout_s": 30
}
```

The built-in supervisor polls active robot sessions. An external experiment agent
uses the same prompt and tool API to drive training and improvement between sessions.
