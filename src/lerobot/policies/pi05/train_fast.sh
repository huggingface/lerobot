python src/lerobot/policies/pi05/train_fast_tokenizer.py \
    --repo_id "local" \
    --root /fsx/jade_choghari/data/libero \
    --action_horizon 10 \
    --encoded_dims "0:7" \
    --vocab_size 1024 \
    --push_to_hub \
    --hub_repo_id jadechoghari/fast-libero-tokenizer-mean-std \
    --normalization_mode MEAN_STD \


# python train_fast_tokenizer.py --repo_id my_dataset