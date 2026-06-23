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
"""Launch ``lerobot-annotate`` on a Hugging Face job (vllm + Qwen3.6-27B VLM).

Spawns one single-GPU ``h200`` job that:

  1. installs ``lerobot`` from ``main`` plus the annotation extras,
  2. boots one vllm server with Qwen3.6-27B (dense VLM),
  3. runs the plan / interjections / vqa modules across the dataset
     in free-form mode (each episode generates its own subtasks +
     memory),
  4. uploads the annotated dataset to ``--new_repo_id`` (when set)
     or back to ``--repo_id``.

Usage:

    HF_TOKEN=hf_... uv run python examples/annotations/run_hf_job.py

Adjust ``CMD`` (dataset, model, hub repo) and ``flavor`` below for your
run. For larger datasets, scale to ``h200x4`` and raise
``--vlm.parallel_servers`` / ``--vlm.num_gpus`` to match.
"""

import os

from huggingface_hub import get_token, run_job

token = os.environ.get("HF_TOKEN") or get_token()
if not token:
    raise RuntimeError("No HF token. Run `huggingface-cli login` or `export HF_TOKEN=hf_...`")

CMD = (
    "apt-get update -qq && apt-get install -y -qq git ffmpeg && "
    "pip install --no-deps "
    "'lerobot @ git+https://github.com/huggingface/lerobot.git@main' && "
    "pip install --upgrade-strategy only-if-needed "
    "datasets pyarrow av jsonlines draccus gymnasium torchcodec mergedeep pyyaml-include toml typing-inspect "
    "openai && "
    "export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 && "
    "export VLLM_VIDEO_BACKEND=pyav && "
    "lerobot-annotate "
    "--repo_id=pepijn223/robocasa_pretrain_human300_v4 "
    "--new_repo_id=pepijn223/robocasa_pretrain_human300_v4_annotated "
    "--push_to_hub=true "
    "--vlm.backend=openai "
    "--vlm.model_id=Qwen/Qwen3.6-27B "
    "--vlm.num_gpus=1 "
    '--vlm.serve_command="vllm serve Qwen/Qwen3.6-27B '
    "--tensor-parallel-size 1 --max-model-len 32768 "
    '--gpu-memory-utilization 0.8 --uvicorn-log-level warning --port {port}" '
    "--vlm.serve_ready_timeout_s=1800 "
    # Qwen3.6 ships with thinking on; annotation wants plain JSON answers.
    "--vlm.chat_template_kwargs='{\"enable_thinking\": false}'"
)

job = run_job(
    image="vllm/vllm-openai:latest",
    command=["bash", "-c", CMD],
    flavor="h200",
    secrets={"HF_TOKEN": token},
    timeout="2h",
)
print(f"Job URL: {job.url}")
print(f"Job ID:  {job.id}")
