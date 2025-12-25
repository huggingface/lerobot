rm -rf /fsx/jade_choghari/outputs/pi0_multi_training
lerobot-train \
    --dataset.repo_id=local\
    --dataset.root=/fsx/jade_choghari/data/libero \
    --output_dir=/fsx/jade_choghari/outputs/pi0_multi_training \
    --job_name=pi0_multi_training \
    --policy.repo_id=jadechoghari/pi0-base1 \
    --policy.path=/fsx/jade_choghari/outputs/libero_training_fast_4/checkpoints/last/pretrained_model/ \
    --policy.dtype=bfloat16 \
    --steps=50000 \
    --save_freq=5000 \
    --batch_size=4 \
    --policy.device=cuda \
