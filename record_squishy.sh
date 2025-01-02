HF_USER=$(huggingface-cli whoami | head -n 1)
python lerobot/scripts/control_robot.py record \
    --robot-path lerobot/configs/robot/so100.yaml \
    --fps 30 \
    --repo-id ${HF_USER}/so100_squishy \
    --tags so100 tutorial \
    --warmup-time-s 5 \
    --episode-time-s 30 \
    --reset-time-s 1 \
    --num-episodes 50 \
    --push-to-hub 1 --single-task 'pick a squishy and put it in the bin'
