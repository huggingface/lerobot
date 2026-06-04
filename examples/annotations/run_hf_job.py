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
    "--new_repo_id=pepijn223/robocasa_pretrain_human300_v4_annotated5 "
    "--push_to_hub=true "
    "--vlm.backend=openai "
    "--vlm.model_id=Qwen/Qwen3.6-27B "
    "--vlm.parallel_servers=1 "
    "--vlm.num_gpus=1 "
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
    # Phase 1 — plan module (subtasks + memory).
    # Embed decoded frames (not a file:// clip): if clip extraction fails,
    # the video_url path silently sends no video and the VLM hallucinates.
    "--plan.use_video_url=false "
    "--plan.frames_per_second=1.0 "
    # 32 frames ≈ 8-10k vision tokens, fits the 32768 context. Don't push
    # toward 128 — that overflows the context (BadRequestError 400).
    "--plan.max_video_frames=32 "
    # Window long episodes into 32s chunks (constant 1 fps density) so they
    # get more subtasks; per-window spans are merged + stitched. 0 disables.
    "--plan.subtask_window_seconds=32 "
    # RoboCasa: the dataset task string is authoritative (eval uses it), so
    # keep it driving subtasks. ``always`` would throw it away and hallucinate.
    "--plan.derive_task_from_video=off "
    # No task augmentation: eval conditions on the exact task strings, so
    # rephrasings are unused at best and harmful when they drift.
    "--plan.n_task_rephrasings=0 "
    # Keep subtask decomposition tight for atomic tasks.
    "--plan.plan_max_steps=10 "
    # Only subtasks + memory — skip the numbered "plan" rows. true re-enables.
    "--plan.emit_plan=false "
    # The describe->segment grounding pass (+1 VLM call/episode) is ON by
    # default; pass --plan.subtask_describe_first=false to skip it.
    # Phase 2 — interjections + speech.
    "--interjections.max_interjections_per_episode=6 "
    # Phase 4 — general VQA: disabled for this run.
    "--vqa.enabled=false"
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
