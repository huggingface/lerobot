python src/lerobot/policies/pi05/train_fast_tokenizer.py \
    --repo_id "lerobot/libero" \
    --action_horizon 50 \
    --encoded_dims "0:6" \
    --vocab_size 1024 \
    --output_dir "/fsx/jade_choghari/outputs/fast_tokenizer"
