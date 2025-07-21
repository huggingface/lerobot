# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

.PHONY: tests

PYTHON_PATH := $(shell which python)

# If uv is installed and a virtual environment exists, use it
UV_CHECK := $(shell command -v uv)
ifneq ($(UV_CHECK),)
	PYTHON_PATH := $(shell .venv/bin/python)
endif

export PATH := $(dir $(PYTHON_PATH)):$(PATH)

DEVICE ?= cpu

build-cpu:
	docker build -t lerobot:latest -f docker/lerobot-cpu/Dockerfile .

build-gpu:
	docker build -t lerobot:latest -f docker/lerobot-gpu/Dockerfile .

test-end-to-end:
	${MAKE} DEVICE=$(DEVICE) test-act-ete-train
	${MAKE} DEVICE=$(DEVICE) test-act-ete-train-resume
	${MAKE} DEVICE=$(DEVICE) test-act-ete-eval
	${MAKE} DEVICE=$(DEVICE) test-diffusion-ete-train
	${MAKE} DEVICE=$(DEVICE) test-diffusion-ete-eval
	${MAKE} DEVICE=$(DEVICE) test-tdmpc-ete-train
	${MAKE} DEVICE=$(DEVICE) test-tdmpc-ete-eval
	${MAKE} DEVICE=$(DEVICE) test-smolvla-ete-train
	${MAKE} DEVICE=$(DEVICE) test-smolvla-ete-eval

test-act-ete-train:
	python -m lerobot.scripts.train \
		--policy.type=act \
		--policy.dim_model=64 \
		--policy.n_action_steps=20 \
		--policy.chunk_size=20 \
		--policy.device=$(DEVICE) \
		--policy.push_to_hub=false \
		--env.type=aloha \
		--env.episode_length=5 \
		--dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
		--dataset.image_transforms.enable=true \
		--dataset.episodes="[0]" \
		--batch_size=2 \
		--steps=4 \
		--eval_freq=2 \
		--eval.n_episodes=1 \
		--eval.batch_size=1 \
		--save_freq=2 \
		--save_checkpoint=true \
		--log_freq=1 \
		--wandb.enable=false \
		--output_dir=tests/outputs/act/

test-act-ete-train-resume:
	python -m lerobot.scripts.train \
		--config_path=tests/outputs/act/checkpoints/000002/pretrained_model/train_config.json \
		--resume=true

test-act-ete-eval:
	python -m lerobot.scripts.eval \
		--policy.path=tests/outputs/act/checkpoints/000004/pretrained_model \
		--policy.device=$(DEVICE) \
		--env.type=aloha \
		--env.episode_length=5 \
		--eval.n_episodes=1 \
		--eval.batch_size=1

test-diffusion-ete-train:
	python -m lerobot.scripts.train \
		--policy.type=diffusion \
		--policy.down_dims='[64,128,256]' \
		--policy.diffusion_step_embed_dim=32 \
		--policy.num_inference_steps=10 \
		--policy.device=$(DEVICE) \
		--policy.push_to_hub=false \
		--env.type=pusht \
		--env.episode_length=5 \
		--dataset.repo_id=lerobot/pusht \
		--dataset.image_transforms.enable=true \
		--dataset.episodes="[0]" \
		--batch_size=2 \
		--steps=2 \
		--eval_freq=2 \
		--eval.n_episodes=1 \
		--eval.batch_size=1 \
		--save_checkpoint=true \
		--save_freq=2 \
		--log_freq=1 \
		--wandb.enable=false \
		--output_dir=tests/outputs/diffusion/

test-diffusion-ete-eval:
	python -m lerobot.scripts.eval \
		--policy.path=tests/outputs/diffusion/checkpoints/000002/pretrained_model \
		--policy.device=$(DEVICE) \
		--env.type=pusht \
		--env.episode_length=5 \
		--eval.n_episodes=1 \
		--eval.batch_size=1

test-tdmpc-ete-train:
	python -m lerobot.scripts.train \
		--policy.type=tdmpc \
		--policy.device=$(DEVICE) \
		--policy.push_to_hub=false \
		--env.type=xarm \
		--env.task=XarmLift-v0 \
		--env.episode_length=5 \
		--dataset.repo_id=lerobot/xarm_lift_medium \
		--dataset.image_transforms.enable=true \
		--dataset.episodes="[0]" \
		--batch_size=2 \
		--steps=2 \
		--eval_freq=2 \
		--eval.n_episodes=1 \
		--eval.batch_size=1 \
		--save_checkpoint=true \
		--save_freq=2 \
		--log_freq=1 \
		--wandb.enable=false \
		--output_dir=tests/outputs/tdmpc/

test-tdmpc-ete-eval:
	python -m lerobot.scripts.eval \
		--policy.path=tests/outputs/tdmpc/checkpoints/000002/pretrained_model \
		--policy.device=$(DEVICE) \
		--env.type=xarm \
		--env.episode_length=5 \
		--env.task=XarmLift-v0 \
		--eval.n_episodes=1 \
		--eval.batch_size=1


test-smolvla-ete-train:
	python -m lerobot.scripts.train \
		--policy.type=smolvla \
		--policy.n_action_steps=20 \
		--policy.chunk_size=20 \
		--policy.device=$(DEVICE) \
		--policy.push_to_hub=false \
		--env.type=aloha \
		--env.episode_length=5 \
		--dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
		--dataset.image_transforms.enable=true \
		--dataset.episodes="[0]" \
		--batch_size=2 \
		--steps=4 \
		--eval_freq=2 \
		--eval.n_episodes=1 \
		--eval.batch_size=1 \
		--save_freq=2 \
		--save_checkpoint=true \
		--log_freq=1 \
		--wandb.enable=false \
		--output_dir=tests/outputs/smolvla/

test-smolvla-ete-eval:
	python -m lerobot.scripts.eval \
		--policy.path=tests/outputs/smolvla/checkpoints/000004/pretrained_model \
		--policy.device=$(DEVICE) \
		--env.type=aloha \
		--env.episode_length=5 \
		--eval.n_episodes=1 \
		--eval.batch_size=1
