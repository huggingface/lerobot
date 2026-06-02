#!/usr/bin/env python
"""Launch ``lerobot-annotate`` on a Hugging Face job (vllm + Qwen3.6 MoE).

Spawns one ``h200x2`` job that:

  1. installs this branch of ``lerobot`` plus the annotation extras,
  2. boots two vllm servers (one per GPU) with Qwen3.6-35B-A3B-FP8,
  3. runs the plan / interjections / vqa modules across the dataset
     in free-form mode (each episode generates its own subtasks +
     memory),
  4. uploads the annotated dataset to ``--dest_repo_id`` (when set)
     or back to ``--repo_id``.

Usage:

    HF_TOKEN=hf_... uv run python examples/annotations/run_hf_job.py

Adjust ``CMD`` below to point at your own dataset / target hub repo.
"""

import os

from huggingface_hub import get_token, run_job

token = os.environ.get("HF_TOKEN") or get_token()
if not token:
    raise RuntimeError("No HF token. Run `huggingface-cli login` or `export HF_TOKEN=hf_...`")

CMD = (
    "apt-get update -qq && apt-get install -y -qq git ffmpeg && "
    "pip install --no-deps "
    "'lerobot @ git+https://github.com/huggingface/lerobot.git@feat/language-annotation-pipeline' && "
    "pip install --upgrade-strategy only-if-needed "
    "datasets pyarrow av jsonlines draccus gymnasium torchcodec mergedeep pyyaml-include toml typing-inspect "
    "openai && "
    "export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 && "
    "export VLLM_VIDEO_BACKEND=pyav && "
    "lerobot-annotate "
    "--repo_id=pepijn223/robocasa_smoke_2atomic_v3 "
    "--dest_repo_id=pepijn223/robocasa_smoke_2atomic_v3_ann "
    "--push_to_hub=true "
    "--vlm.backend=openai "
    "--vlm.model_id=Qwen/Qwen3.6-35B-A3B-FP8 "
    "--vlm.parallel_servers=2 "
    "--vlm.num_gpus=2 "
    '--vlm.serve_command="vllm serve Qwen/Qwen3.6-35B-A3B-FP8 '
    "--tensor-parallel-size 1 --max-model-len 32768 "
    '--gpu-memory-utilization 0.8 --uvicorn-log-level warning --port {port}" '
    "--vlm.serve_ready_timeout_s=1800 "
    "--vlm.client_concurrency=128 "
    "--vlm.max_new_tokens=512 "
    "--vlm.temperature=0.7 "
    "--executor.episode_parallelism=16 "
    "--vlm.chat_template_kwargs='{\"enable_thinking\": false}' "
    "--vlm.camera_key=observation.images.robot0_agentview_right "
    # Phase 1 — plan module (subtasks + plan + memory).
    # Embed decoded frames directly (use_video_url=false) rather than
    # handing the server a file:// clip. The embedded path is more
    # reliable: if clip extraction ever fails, the video_url path would
    # silently send NO video and the VLM would hallucinate subtasks from
    # the task text alone. 2 fps gives dense visual grounding so the VLM
    # labels what actually happens.
    "--plan.frames_per_second=2.0 "
    "--plan.use_video_url=false "
    # IMPORTANT for RoboCasa: the dataset's task string ("Navigate to the
    # stove", "Pick the mug...") is authoritative and is what eval uses.
    # ``derive_task_from_video=off`` keeps that canonical task driving
    # subtask generation. Do NOT use ``always`` here — it throws the real
    # task away, asks the VLM "what is this video about?" with no hint,
    # and the hallucinated task then poisons every subtask + plan row.
    "--plan.derive_task_from_video=off "
    # NO task augmentation for RoboCasa: eval conditions on the exact task
    # strings, so synthetic rephrasings are unused at best and (when they
    # drift, e.g. "wander around the kitchen") harmful. 0 rephrasings +
    # axes disabled = the policy only ever sees the canonical task.
    "--plan.n_task_rephrasings=0 "
    # action_records OFF: the structured {verb,object,arm,grasp,dest}
    # schema is a manipulation schema; RoboCasa navigation / atomic tasks
    # don't fit it and the VLM hallucinates. When on, records are purely
    # additive (emitted as style="action_record" rows) and never touch
    # the subtask text — useful only for long composite manipulation
    # tasks. Leave off for RoboCasa atomic / navigation.
    # Keep subtask decomposition tight for atomic tasks:
    "--plan.plan_max_steps=6 "
    # Phase 2 — interjections + speech.
    "--interjections.max_interjections_per_episode=6 "
    # Phase 4 — general VQA.
    # Ground VQA on the SAME single camera as plan/interjections
    # (--vlm.camera_key) instead of iterating every camera. The whole
    # pipeline then focuses on one view, e.g. observation.images.base.
    "--vqa.restrict_to_default_camera=true "
    "--vqa.K=1 "
    "--vqa.vqa_emission_hz=1.0"
)

job = run_job(
    image="vllm/vllm-openai:latest",
    command=["bash", "-c", CMD],
    flavor="h200x2",
    secrets={"HF_TOKEN": token},
    timeout="2h",
)
print(f"Job URL: {job.url}")
print(f"Job ID:  {job.id}")
