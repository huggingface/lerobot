for episode_index in $(seq 0 19); do
  python -m lerobot.scripts.lerobot_dataset_viz \
    --repo-id yihao-brain-bot/xlerobot-get-musketeers \
    --root /home/yihao/.cache/huggingface/lerobot/yihao-brain-bot/xlerobot-get-musketeers \
    --episode-index "${episode_index}"
done
