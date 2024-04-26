.PHONY: tests

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

test-act-ete-train:
	python lerobot/scripts/train.py \
		policy=act \
		env=aloha \
		wandb.enable=False \
		offline_steps=2 \
		online_steps=0 \
		eval_episodes=1 \
		device=cpu \
		save_model=true \
		save_freq=2 \
		policy.n_action_steps=20 \
		policy.chunk_size=20 \
		policy.batch_size=2 \
		hydra.run.dir=tests/outputs/act/

test-act-ete-eval:
	python lerobot/scripts/eval.py \
		--config tests/outputs/act/.hydra/config.yaml \
		eval_episodes=1 \
		env.episode_length=8 \
		device=cpu \
		policy.pretrained_model_path=tests/outputs/act/models/2.pt

test-diffusion-ete-train:
	python lerobot/scripts/train.py \
		policy=diffusion \
		env=pusht \
		wandb.enable=False \
		offline_steps=2 \
		online_steps=0 \
		eval_episodes=1 \
		device=cpu \
		save_model=true \
		save_freq=2 \
		policy.batch_size=2 \
		hydra.run.dir=tests/outputs/diffusion/

test-diffusion-ete-eval:
	python lerobot/scripts/eval.py \
		--config tests/outputs/diffusion/.hydra/config.yaml \
		eval_episodes=1 \
		env.episode_length=8 \
		device=cpu \
		policy.pretrained_model_path=tests/outputs/diffusion/models/2.pt

test-tdmpc-ete-train:
	python lerobot/scripts/train.py \
		policy=tdmpc \
		env=xarm \
		wandb.enable=False \
		offline_steps=1 \
		online_steps=2 \
		eval_episodes=1 \
		env.episode_length=2 \
		device=cpu \
		save_model=true \
		save_freq=2 \
		policy.batch_size=2 \
		hydra.run.dir=tests/outputs/tdmpc/

test-tdmpc-ete-eval:
	python lerobot/scripts/eval.py \
		--config tests/outputs/tdmpc/.hydra/config.yaml \
		eval_episodes=1 \
		env.episode_length=8 \
		device=cpu \
		policy.pretrained_model_path=tests/outputs/tdmpc/models/2.pt
