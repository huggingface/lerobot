python lerobot/scripts/train.py \
    hydra.job.name=act_pusht \
    hydra.run.dir=outputs/train/act_pusht \
    env=aloha \
    env.task=AlohaInsertion-v0 \
    dataset_repo_id=lerobot/pusht \
    policy=act \
    policy.use_vae=true \
    training.eval_freq=10000 \
    training.log_freq=250 \
    training.offline_steps=100000 \
    training.save_model=true \
    training.save_freq=25000 \
    eval.n_episodes=50 \
    eval.batch_size=50 \
    wandb.enable=false \
    device=cuda \
