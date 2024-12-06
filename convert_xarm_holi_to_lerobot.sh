python lerobot/scripts/push_dataset_to_hub.py \
    --raw-dir /data/xarm_holi_demos \
    --local-dir /data/xarm_holi_demos_lerobot_state_as_action \
    --raw-format xarm_holi_json \
    --repo-id jhseon-holiday/xarm_holi_demos_lerobot_state_as_action \
    --fps 30 \
    --force-override 1