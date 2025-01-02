HF_USER=$(huggingface-cli whoami | head -n 1)
rm -rf /Users/ssaroha/.cache/huggingface/lerobot/pranavsaroha/eval_act_so100_squishy100_3
python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/so100.yaml \
  --fps 30 \
  --repo-id ${HF_USER}/eval_act_so100_squishy100_3 \
  --tags so100 tutorial eval \
  --warmup-time-s 5 \
  --episode-time-s 180 \
  --reset-time-s 10 \
  --num-episodes 5 \
  --policy-overrides device=mps \
  -p ${HF_USER}/act_so100_squishy100 \
  --single-task 'pick a squishy and put it in the bin'

