lerobot-train \
    --dataset.repo_id=lerobot \
    --dataset.root=/fsx/jade_choghari/outputs/collect-data-pgen \
    --output_dir=/fsx/jade_choghari/outputs/pi0test1 \
    --job_name=pi0_training \
    --policy.repo_id=jade_choghari/pi0-base \
    --policy.path=/fsx/jade_choghari/outputs/pi0_fast_fruit1/checkpoints/last/pretrained_model \
    --policy.dtype=bfloat16 \
    --steps=3000 \
    --save_freq=1000 \
    --rename_map='{
        "observation.images.base": "observation.images.base_0_rgb",
        "observation.images.left_wrist": "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist": "observation.images.right_wrist_0_rgb",
        }' \
    --batch_size=4 \
    --policy.device=cuda \
    # --wandb.enable=true \
    # --wandb.disable_artifact=true \
    # --wandb.project=pi05hi-training \

