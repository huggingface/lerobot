export CUDA_LAUNCH_BLOCKING=1 
lerobot-train \
    --dataset.repo_id=local \
    --dataset.root=/fsx/jade_choghari/data/libero \
    --output_dir=/fsx/jade_choghari/outputs/libero_training_fast_5 \
    --job_name=libero_training_fast \
    --policy.repo_id=jade_choghari/pi05-fast-libero \
    --policy.path=/fsx/jade_choghari/models/libero-pi-fast \
    --policy.dtype=bfloat16 \
    --steps=100000 \
    --save_freq=20000 \
    --batch_size=4 \
    --policy.device=cuda \
    --policy.fast_only=true \
    --policy.scheduler_warmup_steps=1000 \
    --policy.scheduler_decay_steps=30000 \
    --policy.scheduler_decay_lr=1e-5 \
    --policy.gradient_checkpointing=true \
    # --wandb.enable=true \
    # --wandb.disable_artifact=true \
    # --wandb.project=pi05-libero-training \
# /fsx/jade_choghari/.cache/huggingface/lerobot/jadechoghari/collect-data