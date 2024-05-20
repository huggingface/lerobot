.PHONY: tests

PYTHON_PATH := $(shell which python)

# If Poetry is installed, redefine PYTHON_PATH to use the Poetry-managed Python
POETRY_CHECK := $(shell command -v poetry)
ifneq ($(POETRY_CHECK),)
    PYTHON_PATH := $(shell poetry run which python)
endif

export PATH := $(dir $(PYTHON_PATH)):$(PATH)


build-cpu:
	docker build -t lerobot:latest -f docker/lerobot-cpu/Dockerfile .

build-gpu:
	docker build -t lerobot:latest -f docker/lerobot-gpu/Dockerfile .

test-end-to-end:
	${MAKE} test-act-ete-train
	${MAKE} test-act-ete-eval
	${MAKE} test-diffusion-ete-train
	${MAKE} test-diffusion-ete-eval
	${MAKE} test-tdmpc-ete-train
	${MAKE} test-tdmpc-ete-eval
	${MAKE} test-default-ete-eval

test-act-ete-train:
	python lerobot/scripts/train.py \
		policy=act \
		env=aloha \
		wandb.enable=False \
		training.offline_steps=2 \
		training.online_steps=0 \
		eval.n_episodes=1 \
		eval.batch_size=1 \
		device=cpu \
		training.save_model=true \
		training.save_freq=2 \
		policy.n_action_steps=20 \
		policy.chunk_size=20 \
		training.batch_size=2 \
		hydra.run.dir=tests/outputs/act/

test-act-ete-eval:
	python lerobot/scripts/eval.py \
		-p tests/outputs/act/checkpoints/000002 \
		eval.n_episodes=1 \
		eval.batch_size=1 \
		env.episode_length=8 \
		device=cpu \

test-diffusion-ete-train:
	python lerobot/scripts/train.py \
		policy=diffusion \
		env=pusht \
		wandb.enable=False \
		training.offline_steps=2 \
		training.online_steps=0 \
		eval.n_episodes=1 \
		eval.batch_size=1 \
		device=cpu \
		training.save_model=true \
		training.save_freq=2 \
		training.batch_size=2 \
		hydra.run.dir=tests/outputs/diffusion/

test-diffusion-ete-eval:
	python lerobot/scripts/eval.py \
		-p tests/outputs/diffusion/checkpoints/000002 \
		eval.n_episodes=1 \
		eval.batch_size=1 \
		env.episode_length=8 \
		device=cpu \

# TODO(alexander-soare): Restore online_steps to 2 when it is reinstated.
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
		device=cpu \
		training.save_model=true \
		training.save_freq=2 \
		training.batch_size=2 \
		hydra.run.dir=tests/outputs/tdmpc/

test-tdmpc-ete-eval:
	python lerobot/scripts/eval.py \
		-p tests/outputs/tdmpc/checkpoints/000002 \
		eval.n_episodes=1 \
		eval.batch_size=1 \
		env.episode_length=8 \
		device=cpu \


test-default-ete-eval:
	python lerobot/scripts/eval.py \
		--config lerobot/configs/default.yaml \
		eval.n_episodes=1 \
		eval.batch_size=1 \
		env.episode_length=8 \
		device=cpu \
