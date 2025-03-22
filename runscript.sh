rm -r $(dirname "$0")/aloha_bear
python lerobot/scripts/control_robot.py \
    --robot.type=aloha \
    --control.type=record \
    --control.single_task="Grasp the teddy bear and move it to the side." \
    --control.repo_id=shalin/aloha_bear \
    --control.fps=30 \
    --control.num_episodes=1 \
    --control.warmup_time_s=5 \
    --control.episode_time_s=5 \
    --control.reset_time_s=5 \
    --control.push_to_hub=false \
    --control.play_sounds=false \
    --control.display_cameras=false
mv ~/.cache/huggingface/lerobot/shalin/aloha_bear $(dirname "$0")/
