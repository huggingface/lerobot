DATA_DIR='/data' python lerobot/scripts/train.py \
    policy=act_xarm_holi_real \
    env=xarm_real \
    seed=1 \
    dataset_repo_id=xarm_holi_demos_lerobot_state_as_action \
    wandb.enable=true \
    wandb.project=lerobot \
    wandb.notes="xarmholi"

