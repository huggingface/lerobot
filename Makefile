.PHONY: tests

PYTHON_PATH := $(shell which python)

# If Poetry is installed, redefine PYTHON_PATH to use the Poetry-managed Python
POETRY_CHECK := $(shell command -v poetry)
ifneq ($(POETRY_CHECK),)
	PYTHON_PATH := $(shell poetry run which python)
endif

export PATH := $(dir $(PYTHON_PATH)):$(PATH)

DEVICE ?= cpu

build-cpu:
	docker build -t lerobot:latest -f docker/lerobot-cpu/Dockerfile .

build-gpu:
	docker build -t lerobot:latest -f docker/lerobot-gpu/Dockerfile .

test-end-to-end:
	${MAKE} DEVICE=$(DEVICE) test-act-ete-train
	${MAKE} DEVICE=$(DEVICE) test-act-ete-eval
	${MAKE} DEVICE=$(DEVICE) test-act-ete-train-amp
	${MAKE} DEVICE=$(DEVICE) test-act-ete-eval-amp
	${MAKE} DEVICE=$(DEVICE) test-diffusion-ete-train
	${MAKE} DEVICE=$(DEVICE) test-diffusion-ete-eval
	${MAKE} DEVICE=$(DEVICE) test-tdmpc-ete-train
	${MAKE} DEVICE=$(DEVICE) test-tdmpc-ete-train-with-online
	${MAKE} DEVICE=$(DEVICE) test-tdmpc-ete-eval
	${MAKE} DEVICE=$(DEVICE) test-default-ete-eval
	${MAKE} DEVICE=$(DEVICE) test-act-pusht-tutorial

test-act-ete-train:
	python lerobot/scripts/train.py \
		policy=act \
		policy.dim_model=64 \
		env=aloha \
		wandb.enable=False \
		training.offline_steps=2 \
		training.online_steps=0 \
		eval.n_episodes=1 \
		eval.batch_size=1 \
		device=$(DEVICE) \
		training.save_checkpoint=true \
		training.save_freq=2 \
		policy.n_action_steps=20 \
		policy.chunk_size=20 \
		training.batch_size=2 \
		training.image_transforms.enable=true \
		hydra.run.dir=tests/outputs/act/

test-act-ete-eval:
	python lerobot/scripts/eval.py \
		-p tests/outputs/act/checkpoints/000002/pretrained_model \
		eval.n_episodes=1 \
		eval.batch_size=1 \
		env.episode_length=8 \
		device=$(DEVICE) \

test-act-ete-train-amp:
	python lerobot/scripts/train.py \
		policy=act \
		policy.dim_model=64 \
		env=aloha \
		wandb.enable=False \
		training.offline_steps=2 \
		training.online_steps=0 \
		eval.n_episodes=1 \
		eval.batch_size=1 \
		device=$(DEVICE) \
		training.save_checkpoint=true \
		training.save_freq=2 \
		policy.n_action_steps=20 \
		policy.chunk_size=20 \
		training.batch_size=2 \
		hydra.run.dir=tests/outputs/act_amp/ \
		training.image_transforms.enable=true \
		use_amp=true

test-act-ete-eval-amp:
	python lerobot/scripts/eval.py \
		-p tests/outputs/act_amp/checkpoints/000002/pretrained_model \
		eval.n_episodes=1 \
		eval.batch_size=1 \
		env.episode_length=8 \
		device=$(DEVICE) \
		use_amp=true

test-diffusion-ete-train:
	python lerobot/scripts/train.py \
		policy=diffusion \
		policy.down_dims=\[64,128,256\] \
		policy.diffusion_step_embed_dim=32 \
		policy.num_inference_steps=10 \
		env=pusht \
		wandb.enable=False \
		training.offline_steps=2 \
		training.online_steps=0 \
		eval.n_episodes=1 \
		eval.batch_size=1 \
		device=$(DEVICE) \
		training.save_checkpoint=true \
		training.save_freq=2 \
		training.batch_size=2 \
		training.image_transforms.enable=true \
		hydra.run.dir=tests/outputs/diffusion/

test-diffusion-ete-eval:
	python lerobot/scripts/eval.py \
		-p tests/outputs/diffusion/checkpoints/000002/pretrained_model \
		eval.n_episodes=1 \
		eval.batch_size=1 \
		env.episode_length=8 \
		device=$(DEVICE) \

test-tdmpc-ete-train:
	python lerobot/scripts/train.py \
		policy=tdmpc \
		env=xarm \
		env.task=XarmLift-v0 \
		dataset_repo_id=lerobot/xarm_lift_medium \
		wandb.enable=False \
		training.offline_steps=2 \
		training.online_steps=0 \
		eval.n_episodes=1 \
		eval.batch_size=1 \
		env.episode_length=2 \
		device=$(DEVICE) \
		training.save_checkpoint=true \
		training.save_freq=2 \
		training.batch_size=2 \
		training.image_transforms.enable=true \
		hydra.run.dir=tests/outputs/tdmpc/

test-tdmpc-ete-train-with-online:
	python lerobot/scripts/train.py \
		env=pusht \
		env.gym.obs_type=environment_state_agent_pos \
		policy=tdmpc_pusht_keypoints \
		eval.n_episodes=1 \
		eval.batch_size=1 \
		env.episode_length=10 \
		device=$(DEVICE) \
		training.offline_steps=2 \
		training.online_steps=20 \
		training.save_checkpoint=false \
		training.save_freq=10 \
		training.batch_size=2 \
		training.online_rollout_n_episodes=2 \
		training.online_rollout_batch_size=2 \
		training.online_steps_between_rollouts=10 \
		training.online_buffer_capacity=15 \
		eval.use_async_envs=true \
		hydra.run.dir=tests/outputs/tdmpc_online/


test-tdmpc-ete-eval:
	python lerobot/scripts/eval.py \
		-p tests/outputs/tdmpc/checkpoints/000002/pretrained_model \
		eval.n_episodes=1 \
		eval.batch_size=1 \
		env.episode_length=8 \
		device=$(DEVICE) \

test-default-ete-eval:
	python lerobot/scripts/eval.py \
		--config lerobot/configs/default.yaml \
		eval.n_episodes=1 \
		eval.batch_size=1 \
		env.episode_length=8 \
		device=$(DEVICE) \

test-act-pusht-tutorial:
	cp examples/advanced/1_train_act_pusht/act_pusht.yaml lerobot/configs/policy/created_by_Makefile.yaml
	python lerobot/scripts/train.py \
		policy=created_by_Makefile.yaml \
		env=pusht \
		wandb.enable=False \
		training.offline_steps=2 \
		eval.n_episodes=1 \
		eval.batch_size=1 \
		env.episode_length=2 \
		device=$(DEVICE) \
		training.save_model=true \
		training.save_freq=2 \
		training.batch_size=2 \
		training.image_transforms.enable=true \
		hydra.run.dir=tests/outputs/act_pusht/
	rm lerobot/configs/policy/created_by_Makefile.yaml
