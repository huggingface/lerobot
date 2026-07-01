#!/usr/bin/env bash

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

# LingBot-VLA one-line smoke test.
# Loads the policy from a pretrained checkpoint, feeds a synthetic observation
# (one camera view + real-dim state + a task string) through the processor
# pipeline, runs select_action, and asserts a final action vector. Verifies the
# full install -> load -> infer path end to end.
#
# Usage:
#   bash examples/lingbot_vla/run_lingbot_vla.sh
#
# Env overrides:
#   LINGBOT_CKPT   pretrained path or HF repo id (default: robbyant/lingbot-vla-4b)
#   LINGBOT_DEVICE cuda | cpu (default: cuda)
set -euo pipefail

CKPT="${LINGBOT_CKPT:-robbyant/lingbot-vla-4b}"
DEVICE="${LINGBOT_DEVICE:-cuda}"

python examples/lingbot_vla/smoke_test.py \
    --pretrained_path "${CKPT}" \
    --device "${DEVICE}"
