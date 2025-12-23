export CUDA_LAUNCH_BLOCKING=1 
lerobot-train \
    --dataset.repo_id=local \
    --dataset.root=/fsx/jade_choghari/data/libero \
    --output_dir=/fsx/jade_choghari/outputs/libero_training_fast_1 \
    --job_name=libero_training_fast \
    --policy.repo_id=jade_choghari/pi05-fast-libero \
    --policy.path=/fsx/jade_choghari/models/libero-pi-fast \
    --policy.dtype=bfloat16 \
    --steps=200000 \
    --save_freq=30000 \
    --batch_size=16 \
    --policy.device=cuda \
    --policy.fast_only=true \
    --policy.gradient_checkpointing=true \
    # --wandb.enable=true \
    # --wandb.disable_artifact=true \
    # --wandb.project=pi05-libero-training \
# /fsx/jade_choghari/.cache/huggingface/lerobot/jadechoghari/collect-data