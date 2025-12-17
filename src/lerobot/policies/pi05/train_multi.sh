rm -rf /fsx/jade_choghari/outputs/pi0_multi_training
accelerate launch --multi_gpu --num_processes=2 \
    $(which lerobot-train) \
    --dataset.repo_id=local \
    --dataset.root=/fsx/jade_choghari/outputs/collect-data-pgen \
    --output_dir=/fsx/jade_choghari/outputs/pi0_multi_training \
    --job_name=pi0_multi_training \
    --policy.repo_id=jadechoghari/pi0-base1 \
    --policy.path=lerobot/pi05_base \
    --policy.dtype=bfloat16 \
    --steps=50000 \
    --save_freq=5000 \
    --rename_map='{
        "observation.images.base": "observation.images.base_0_rgb",
        "observation.images.left_wrist": "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist": "observation.images.right_wrist_0_rgb",
        }' \
    --policy.gradient_checkpointing=true \
    --batch_size=1 \
    --policy.device=cpu
    # --wandb.enable=true \
    # --wandb.disable_artifact=true \
    # --wandb.project=pi05hi-training \
