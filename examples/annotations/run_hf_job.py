#!/usr/bin/env python
"""Launch ``lerobot-annotate`` on a Hugging Face job (vllm + Qwen3.6 MoE).

Spawns one ``h200x2`` job that:

  1. installs this branch of ``lerobot`` plus the annotation extras,
  2. boots two vllm servers (one per GPU) with Qwen3.6-35B-A3B-FP8,
  3. discovers the dataset's canonical subtask + memory vocabulary
     from the first 3 sample episodes (phase 0),
  4. runs the plan / interjections / vqa modules across the dataset
     (subtasks + memory are constrained to the canonical vocabulary),
  5. uploads the annotated dataset to ``--dest_repo_id`` (when set)
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
    "--repo_id=imstevenpmwork/super_poulain_draft "
    "--dest_repo_id=pepijn223/super_poulain_vocab "
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
    "--vlm.camera_key=observation.images.wrist "
<<<<<<< HEAD:examples/annotation/run_hf_job.py
    "--module_1.frames_per_second=1.0 "
    "--module_1.use_video_url=true "
    "--module_1.use_video_url_fps=1.0 "
    "--module_1.derive_task_from_video=always "
    "--module_1.n_task_rephrasings=30 "
    "--module_2.max_interjections_per_episode=6 "
    "--module_3.K=3 "
    "--module_3.vqa_emission_hz=1.0 "
    "--push_to_hub=pepijn223/super_poulain_full_tool3"
=======
    # Phase 0 — canonical vocabulary discovery from the first N sample
    # episodes. The VLM picks the right number of subtask + memory
    # entries itself from what it sees; the resulting
    # meta/canonical_vocabulary.json constrains every subtask + memory
    # string to a small repeatable target distribution.
    "--vocabulary.sample_episodes=3 "
    # Phase 1 — plan module (subtasks + plan + memory + task_aug).
    "--plan.frames_per_second=1.0 "
    "--plan.use_video_url=true "
    "--plan.use_video_url_fps=1.0 "
    "--plan.derive_task_from_video=always "
    "--plan.n_task_rephrasings=30 "
    # Phase 2 — interjections + speech.
    "--interjections.max_interjections_per_episode=6 "
    # Phase 4 — general VQA.
    "--vqa.K=3 "
    "--vqa.vqa_emission_hz=1.0"
>>>>>>> origin/feat/language-annotation-pipeline:examples/annotations/run_hf_job.py
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
