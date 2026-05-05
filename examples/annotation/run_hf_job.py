#!/usr/bin/env python
"""Launch ``lerobot-annotate`` on a Hugging Face job (vllm + Qwen3.6 MoE).

Spawns one ``h200x2`` job that:

  1. installs this branch of ``lerobot`` plus the annotation extras,
  2. boots two vllm servers (one per GPU) with Qwen3.6-35B-A3B-FP8,
  3. runs Module 1/2/3 across the dataset (per-camera VQA via PR 3471),
  4. uploads the annotated dataset to ``--push_to_hub``.

Usage:

    HF_TOKEN=hf_... uv run python examples/annotation/run_hf_job.py

Adjust ``CMD`` below to point at your own dataset / target hub repo.
"""

import os

from huggingface_hub import get_token, run_job

token = os.environ.get("HF_TOKEN") or get_token()
if not token:
    raise RuntimeError("No HF token. Run `huggingface-cli login` or `export HF_TOKEN=hf_...`")

# --- Diversity knobs (Pi0.7-style prompt expansion) -----------------------
# Bumped roughly 3x across the board to fight memorization on small datasets.
# A single dataset trained for many epochs with deterministic atom wording
# converges to perfect recall on training prompts but produces JSON-token
# garbage at inference for any wording that drifts slightly. More atom
# variants per episode + higher sampling temperature widens the training
# distribution so the model has to actually use its language head, not
# just memorize.
#
# Pushes to a *new* hub repo (``_tool3``) so the previous annotation pass
# (``_tool2``) stays intact — re-train from scratch on the new dataset and
# compare loss-curve shapes to verify the diversity bump is doing something.
CMD = (
    "apt-get update -qq && apt-get install -y -qq git ffmpeg && "
    "pip install --no-deps "
    "'lerobot @ git+https://github.com/huggingface/lerobot.git@feat/language-annotation-pipeline' && "
    "pip install --upgrade-strategy only-if-needed "
    "datasets pyarrow av jsonlines draccus gymnasium torchcodec mergedeep pyyaml-include toml typing-inspect && "
    "export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 && "
    "export VLLM_VIDEO_BACKEND=pyav && "
    "lerobot-annotate "
    "--repo_id=imstevenpmwork/super_poulain_draft "
    "--vlm.backend=openai "
    "--vlm.model_id=Qwen/Qwen3.6-35B-A3B-FP8 "
    "--vlm.parallel_servers=2 "
    "--vlm.num_gpus=2 "
    '--vlm.serve_command="vllm serve Qwen/Qwen3.6-35B-A3B-FP8 '
    "--tensor-parallel-size 1 --max-model-len 32768 "
    '--gpu-memory-utilization 0.8 --uvicorn-log-level warning --port {port}" '
    "--vlm.serve_ready_timeout_s=1800 "
    "--vlm.client_concurrency=256 "
    "--vlm.max_new_tokens=512 "
    "--vlm.temperature=0.7 "
    "--executor.episode_parallelism=32 "
    "--vlm.chat_template_kwargs='{\"enable_thinking\": false}' "
    "--vlm.camera_key=observation.images.wrist "
    "--module_1.frames_per_second=1.0 "
    "--module_1.use_video_url=true "
    "--module_1.use_video_url_fps=1.0 "
    "--module_1.derive_task_from_video=always "
    "--module_1.n_task_rephrasings=30 "
    "--module_2.max_interjections_per_episode=9 "
    "--module_3.K=3 "
    "--module_3.vqa_emission_hz=2.0 "
    "--push_to_hub=pepijn223/super_poulain_full_tool3"
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
