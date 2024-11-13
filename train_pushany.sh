DATA_DIR='/data' python lerobot/scripts/train.py \
    policy=diffusion \
    env=pushany \
    dataset_repo_id=pushany_demos_lerobot \
    wandb.enable=true \
    wandb.project=lerobot \
    wandb.notes="pushany" 

