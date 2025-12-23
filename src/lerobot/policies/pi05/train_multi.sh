accelerate launch --multi_gpu --num_processes=2 \
    $(which lerobot-train) \
    --dataset.repo_id=lerobot/libero \
    --output_dir=/fsx/jade_choghari/outputs/libero_training_fast \
    --job_name=libero_training_fast \
    --policy.repo_id=jade_choghari/pi05-fast-libero \
    --policy.path=/fsx/jade_choghari/models/libero-pi-fast \
    --policy.dtype=bfloat16 \
    --steps=200000 \
    --save_freq=30000 \
    --batch_size=16 \
    --policy.device=cuda \
    --policy.fast_only=true \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=pi05-libero-training \
