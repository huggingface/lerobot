HF_USER=$(huggingface-cli whoami | head -n 1)
python lerobot/scripts/control_robot.py record \
    --robot-path lerobot/configs/robot/so100.yaml \
    --fps 30 \
    --repo-id ${HF_USER}/so100_carrot_5 \
    --tags so100 tutorial \
    --warmup-time-s 5 \
    --episode-time-s 20 \
    --reset-time-s 1 \
    --num-episodes 40 \
    --local-files-only 1 \
    --resume 1 \
    --push-to-hub 1 --single-task 'pick a carrot and put it in the bin'
