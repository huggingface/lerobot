#!/usr/bin/env python
"""Launch ``lerobot-annotate`` on a Hugging Face job (vllm + Qwen3.6-27B VLM).

Spawns one ``h200x4`` job that:

  1. installs this branch of ``lerobot`` plus the annotation extras,
  2. boots four vllm servers (one per GPU) with Qwen3.6-27B (dense VLM),
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
    "--repo_id=pepijn223/robocasa_pretrain_human300_v4 "
    "--dest_repo_id=pepijn223/robocasa_pretrain_human300_v4_annotated5 "
    "--push_to_hub=true "
    "--vlm.backend=openai "
    "--vlm.model_id=Qwen/Qwen3.6-27B "
    "--vlm.parallel_servers=4 "
    "--vlm.num_gpus=4 "
    '--vlm.serve_command="vllm serve Qwen/Qwen3.6-27B '
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
    # the task text alone.
    #
    # CONTEXT BUDGET: with embedded frames, each frame is ~250-320 vision
    # tokens. The model's context is 32768 (see --max-model-len). 32
    # frames sampled uniformly across the episode (~8-10k tokens) fits
    # comfortably alongside the prompt and the describe pass.
    # Do NOT raise max_video_frames toward 128 with embedded frames — that
    # is ~33-39k tokens and overflows the context (BadRequestError 400,
    # "Input length exceeds maximum context length").
    "--plan.use_video_url=false "
    "--plan.frames_per_second=1.0 "
    "--plan.max_video_frames=32 "
    # Constant 1 fps density via windowing: episodes longer than 32s are
    # split into 32-second windows (each 32 frames @ 1 fps, fits context),
    # so long episodes get MORE subtasks instead of a sparser whole-episode
    # view. describe->segment runs per window; spans are merged +
    # stitched to a contiguous whole-episode cover. 0 disables.
    "--plan.subtask_window_seconds=32 "
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
    "--plan.plan_max_steps=10 "
    # Only annotate subtasks + memory — skip the numbered "plan" rows
    # (and their per-boundary VLM call). Flip to true to re-enable plan.
    "--plan.emit_plan=false "
    # NOTE: the grounding pass (describe -> segment, +1 VLM call/episode)
    # is ON BY DEFAULT. Pass --plan.subtask_describe_first=false to disable
    # on datasets you've verified are easy and want fewer calls.
    # Phase 2 — interjections + speech.
    "--interjections.max_interjections_per_episode=6 "
    # Phase 4 — general VQA: DISABLED for this run.
    "--vqa.enabled=false"
)

job = run_job(
    image="vllm/vllm-openai:latest",
    command=["bash", "-c", CMD],
    flavor="h200x4",
    secrets={"HF_TOKEN": token},
    timeout="2h",
)
print(f"Job URL: {job.url}")
print(f"Job ID:  {job.id}")
