export CUDA_LAUNCH_BLOCKING=1 
lerobot-train \
    --dataset.repo_id=local \
    --dataset.root=/fsx/jade_choghari/outputs/collect-data-pgen \
    --output_dir=/fsx/jade_choghari/outputs/pi0_fast_fruit2 \
    --job_name=pi0_training \
    --policy.repo_id=jade_choghari/pi0-base1 \
    --policy.path=lerobot/pi05_base \
    --policy.dtype=bfloat16 \
    --steps=200000 \
    --save_freq=5000 \
    --rename_map='{
        "observation.images.base": "observation.images.base_0_rgb",
        "observation.images.left_wrist": "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist": "observation.images.right_wrist_0_rgb",
        }' \
    --batch_size=16 \
    --policy.device=cuda \
    --policy.fast_only=true \
    # --wandb.enable=true \
    # --wandb.disable_artifact=true \
    # --wandb.project=pi05hi-training \
# /fsx/jade_choghari/.cache/huggingface/lerobot/jadechoghari/collect-data