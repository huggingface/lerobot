#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Launch ``lerobot-annotate`` on a Hugging Face job (vllm + Qwen3.6 MoE).

Spawns one ``h200x2`` job that:

  1. installs this branch of ``lerobot`` plus the annotation extras,
  2. boots two vllm servers (one per GPU) with Qwen3.6-35B-A3B-FP8,
  3. runs the plan / interjections / vqa modules across the dataset,
  4. uploads the annotated dataset back to ``--repo_id``.

``--repo_id`` is both the download source and, with ``--push_to_hub=true``,
the upload destination — the job annotates the dataset in place.

Usage:

    HF_TOKEN=hf_... uv run python examples/annotations/run_hf_job.py

Adjust ``CMD`` below to point at your own dataset.
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
    # Mirror lerobot's [annotations] runtime deps. ``openai`` is required
    # because ``VlmConfig.backend`` defaults to ``"openai"`` (which talks
    # to a vllm/transformers/ktransformers OpenAI-compatible server).
    "datasets pyarrow av jsonlines draccus gymnasium torchcodec mergedeep pyyaml-include "
    "toml typing-inspect openai && "
    "export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 && "
    "export VLLM_VIDEO_BACKEND=pyav && "
    "lerobot-annotate "
    # The dataset to annotate; also the push destination (annotate in place).
    "--repo_id=<your-org>/<your-dataset> "
    "--push_to_hub=true "
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
    "--executor.episode_parallelism=32 "
    "--vlm.chat_template_kwargs='{enable_thinking: false}' "
    "--vlm.camera_key=observation.images.wrist "
    "--plan.frames_per_second=1.0 "
    "--plan.use_video_url=true "
    "--plan.use_video_url_fps=1.0 "
    "--vqa.K=1 --vqa.vqa_emission_hz=0.2"
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
