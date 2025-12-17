python src/lerobot/policies/pi05/train_fast_tokenizer.py \
    --repo_id "local" \
    --root "/fsx/jade_choghari/outputs/collect-data-pgen" \
    --action_horizon 16 \
    --encoded_dims "0:15" \
    --action_horizon 50 \
    --vocab_size 1024 \
    --scale 10.0 \
    --output_dir "/fsx/jade_choghari/outputs/fast_tokenizer"
