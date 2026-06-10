#!/usr/bin/env python
"""A/B launcher: current pipeline vs. contact-sheet + causal-boundary changes.

Both jobs install the SAME experiment branch
(``experiment/contact-sheets-causal-boundaries``) and use an identical config;
the only difference is that the experiment job sets the two new flags
(``--plan.frame_input_mode=contact_sheet`` and ``--plan.causal_boundary_rules=true``).
With the flags off the branch behaves byte-identically to the current pipeline,
so the baseline is a faithful "current pipeline" run and the two new ideas are
the only variable.

Dataset: imstevenpmwork/super_poulain_draft (50 eps, omx_follower, 30 fps).
Generates: subtask + plan + memory (plan module) + VQA. Camera: front.
"""

import os

from huggingface_hub import get_token, run_job

token = os.environ.get("HF_TOKEN") or get_token()
if not token:
    raise RuntimeError("No HF token. Run `hf auth login` or `export HF_TOKEN=hf_...`")

BRANCH = "experiment/contact-sheets-causal-boundaries"
SRC_REPO = "imstevenpmwork/super_poulain_draft"

# Shared command. {extra_flags} is where the experiment job injects its flags;
# {new_repo_id} is the per-job output dataset.
CMD_TEMPLATE = (
    "apt-get update -qq && apt-get install -y -qq git ffmpeg && "
    "pip install --no-deps "
    f"'lerobot @ git+https://github.com/huggingface/lerobot.git@{BRANCH}' && "
    "pip install --upgrade-strategy only-if-needed "
    # av pinned to <16: PyAV 17.x dropped ``av.option`` which lerobot's
    # pyav_utils imports at module load. draccus pinned to lerobot's required
    # 0.10.0 (the CLI config parser). Both must be pinned because lerobot is
    # installed with --no-deps, so its own constraints aren't enforced.
    "datasets pyarrow 'av>=15.0.0,<16.0.0' jsonlines 'draccus==0.10.0' gymnasium torchcodec "
    "mergedeep pyyaml-include toml typing-inspect "
    "openai && "
    "export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 && "
    "export VLLM_VIDEO_BACKEND=pyav && "
    "lerobot-annotate "
    f"--repo_id={SRC_REPO} "
    "--new_repo_id={new_repo_id} "
    "--push_to_hub=true "
    "--vlm.backend=openai "
    "--vlm.model_id=Qwen/Qwen3.6-27B "
    "--vlm.parallel_servers=1 "
    "--vlm.num_gpus=1 "
    '--vlm.serve_command="vllm serve Qwen/Qwen3.6-27B '
    "--tensor-parallel-size 1 --max-model-len 32768 "
    '--gpu-memory-utilization 0.8 --uvicorn-log-level warning --port {{port}}" '
    "--vlm.serve_ready_timeout_s=1800 "
    "--vlm.client_concurrency=128 "
    "--vlm.max_new_tokens=512 "
    "--vlm.temperature=0.7 "
    "--executor.episode_parallelism=16 "
    "--vlm.chat_template_kwargs='{{\"enable_thinking\": false}}' "
    "--vlm.camera_key=observation.images.front "
    # ---- plan module: subtasks + plan + memory ----
    "--plan.use_video_url=false "
    "--plan.frames_per_second=1.0 "
    "--plan.max_video_frames=32 "
    "--plan.subtask_window_seconds=0 "
    "--plan.derive_task_from_video=off "
    "--plan.n_task_rephrasings=0 "
    "--plan.plan_max_steps=8 "
    "--plan.emit_plan=true "
    "--plan.emit_memory=true "
    # ---- interjections + speech (drives plan refreshes) ----
    "--interjections.max_interjections_per_episode=3 "
    # ---- general VQA ----
    "--vqa.enabled=true "
    "{extra_flags}"
)

JOBS = [
    {
        "name": "super-poulain-baseline",
        "new_repo_id": "pepijn223/super_poulain_draft_baseline",
        "extra_flags": "",
    },
    {
        "name": "super-poulain-contactsheet-causal",
        "new_repo_id": "pepijn223/super_poulain_draft_contactsheet_causal",
        "extra_flags": (
            "--plan.frame_input_mode=contact_sheet "
            "--plan.causal_boundary_rules=true"
        ),
    },
]

for spec in JOBS:
    cmd = CMD_TEMPLATE.format(new_repo_id=spec["new_repo_id"], extra_flags=spec["extra_flags"])
    job = run_job(
        image="vllm/vllm-openai:latest",
        command=["bash", "-c", cmd],
        flavor="h200",
        secrets={"HF_TOKEN": token},
        timeout="2h",
    )
    print(f"[{spec['name']}] -> {spec['new_repo_id']}")
    print(f"  Job URL: {job.url}")
    print(f"  Job ID:  {job.id}")
