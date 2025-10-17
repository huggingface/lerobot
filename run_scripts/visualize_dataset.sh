for episode_index in $(seq 0 49); do
  python -m lerobot.scripts.lerobot_dataset_viz \
    --repo-id yihao-brain-bot/xlerobot-get-water \
    --root /home/yihao/.cache/huggingface/lerobot/yihao-brain-bot/xlerobot-get-water \
    --episode-index "${episode_index}"
done
