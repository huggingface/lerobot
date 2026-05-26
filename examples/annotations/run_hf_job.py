#!/usr/bin/env python
"""Launch ``lerobot-annotate`` on a Hugging Face job (vllm + Qwen3.6 MoE).

Spawns one ``h200x4`` job that:

  1. installs this branch of ``lerobot`` plus the annotation extras,
  2. boots four vllm servers (one per H200) with Qwen3.6-35B-A3B-FP8,
  3. runs the plan + vqa modules across the dataset in free-form
     mode — phase 0 (canonical vocabulary discovery) is disabled so
     every episode's subtasks + memory are generated independently;
     interjections is also disabled, which short-circuits the
     plan_update phase that depends on it,
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
    "--dest_repo_id=pepijn223/robocasa_smoke_2atomic_v3_annotated "
    "--push_to_hub=true "
    "--vlm.backend=openai "
    "--vlm.model_id=Qwen/Qwen3.6-35B-A3B-FP8 "
    "--vlm.parallel_servers=4 "
    "--vlm.num_gpus=4 "
    '--vlm.serve_command="vllm serve Qwen/Qwen3.6-35B-A3B-FP8 '
    # 4× the context (32768 → 131072) so long episodes at 1 Hz fit even
    # at full Qwen vision resolution: 90 frames @ ~700 vision tokens/frame
    # ≈ 63 k tokens, comfortably under 131 k. On 1× H200 (144 GB) the
    # 35B-FP8 model leaves plenty of room for the bigger KV cache.
    "--tensor-parallel-size 1 --max-model-len 131072 "
    '--gpu-memory-utilization 0.85 --uvicorn-log-level warning --port {port}" '
    "--vlm.serve_ready_timeout_s=1800 "
    "--vlm.client_concurrency=256 "
    "--vlm.max_new_tokens=512 "
    "--vlm.temperature=0.7 "
    "--executor.episode_parallelism=64 "
    "--vlm.chat_template_kwargs='{\"enable_thinking\": false}' "
    # Whole-scene agentview is the right choice for subtask reasoning +
    # VQA on robocasa: the wrist (``robot0_eye_in_hand``) usually only
    # sees the gripper + nearby object, which hurts "what is happening
    # in this episode" decomposition. Override per-dataset if your
    # cameras are named differently (inspect ``meta/info.json``).
    "--vlm.camera_key=observation.images.robot0_agentview_left "
    # Phase 0 — canonical vocabulary discovery DISABLED. This dataset's
    # episodes span heterogeneous tasks/scenes, so a single shared
    # subtask + memory vocabulary would be too narrow — each episode
    # generates its subtasks + memory free-form instead.
    "--vocabulary.enabled=false "
    # Phase 1 — plan module (subtasks + plan + memory + task_aug).
    "--plan.enabled=true "
    "--plan.frames_per_second=1.0 "
    "--plan.use_video_url=true "
    "--plan.use_video_url_fps=1.0 "
    # Force coarse, composite subtasks (``pick up X`` = approach + grasp
    # + lift in one span, not three). 3 s is large enough to host a
    # full grasp-or-place composite at typical 20 fps robocasa speeds;
    # any candidate span shorter than this gets merged into a neighbour
    # by the prompt's authoring rules (see module_1_subtasks.txt).
    "--plan.min_subtask_seconds=3.0 "
    # Cap so the VLM can't drift into micro-segmentation. Combined with
    # the composite-action rules in the prompt, this targets ~3-6
    # meaningful spans per episode for typical pick-and-place demos.
    "--plan.plan_max_steps=9 "
    # ``off`` keeps the dataset's canonical ``record.episode_task`` as-is
    # — no per-episode VLM "what is this video about" call. Switch to
    # ``if_short`` (default) only if some episodes have placeholder /
    # missing canonical tasks; ``always`` overrides every episode's task.
    "--plan.derive_task_from_video=off "
    # 0 disables the task_aug pass entirely (see PlanConfig.n_task_rephrasings
    # docstring) — no per-episode paraphrase generation, no task_aug rows.
    "--plan.n_task_rephrasings=0 "
    # Phase 2 — interjections OFF (also skips phase 3 plan_update,
    # see executor.py:_run_plan_update_phase guard).
    "--interjections.enabled=false "
    # Phase 4 — general VQA. K=1 keeps each VQA answer on its own
    # emission frame (no temporal smear); see VqaConfig.K docstring.
    # 3 Hz cadence: at 20 fps source, that's a VQA tick every ~7 frames.
    # NOTE: VQA emits per-camera, so for robocasa (3 cameras) each tick
    # produces 3 (user, assistant) row pairs — total call volume ~= 3 *
    # 3 Hz * mean_episode_seconds * n_episodes.
    "--vqa.enabled=true "
    "--vqa.K=1 "
    "--vqa.vqa_emission_hz=3.0"
)

job = run_job(
    image="vllm/vllm-openai:latest",
    command=["bash", "-c", CMD],
    flavor="h200x4",
    secrets={"HF_TOKEN": token},
    timeout="24h",
)
print(f"Job URL: {job.url}")
print(f"Job ID:  {job.id}")
