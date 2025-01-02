HF_USER=$(huggingface-cli whoami | head -n 1)
rm -rf /Users/ssaroha/.cache/huggingface/lerobot/pranavsaroha/eval_openvla
python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/so100.yaml \
  --fps 30 \
  --repo-id ${HF_USER}/eval_openvla \
  --num-episodes 2 \
  -p openvla/openvla-7b \
  --policy-overrides instruction="pick up the blue object" \
  --single-task 'pick up the blue object' \
  --policy-overrides device="mps"
