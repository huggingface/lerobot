eval script
```bash
lerobot-record   --robot.type=bi_so100_follower   --robot.left_arm_port=/dev/ttyACM2   --robot.right_arm_port=/dev/ttyACM1   --robot.id=bimanual_follower   --robot.cameras='{
    left_gripper: {"type": "opencv", "index_or_path": "/dev/video4", "width": 640, "height": 480, "fps": 30},
    top: {"type": "opencv", "index_or_path": "/dev/video0", "width": 640, "height": 480, "fps": 30, "rotation": ROTATE_180},
    right_gripper: {"type": "opencv", "index_or_path": "/dev/video2", "width": 640, "height": 480, "fps": 30}
  }'   --teleop.type=bi_so100_leader   --teleop.left_arm_port=/dev/ttyACM3   --teleop.right_arm_port=/dev/ttyACM0   --teleop.id=bimanual_leader   --display_data=true   --dataset.repo_id=YieumYoon/eval_groot-bimanual-so100-crlbasket-004   --dataset.num_episodes=10   --dataset.single_task="Grab the red cube and put it in a red basket"   --dataset.episode_time_s=60   --dataset.reset_time_s=15   --policy.path=YieumYoon/groot-bimanual-so100-crlbasket-004
```
recode dataset
```bash
lerobot-record   --robot.type=bi_so100_follower \
    --robot.left_arm_port=/dev/ttyACM1 \
    --robot.right_arm_port=/dev/ttyACM0 \
    --robot.id=bimanual_follower \
    --robot.cameras='{
        left_gripper: {"type": "opencv", "index_or_path": "/dev/video4", "width": 640, "height": 480, "fps": 30},
        top: {"type": "opencv", "index_or_path": "/dev/video0", "width": 640, "height": 480, "fps": 30, "rotation": ROTATE_180},
        right_gripper: {"type": "opencv", "index_or_path": "/dev/video2", "width": 640, "height": 480, "fps": 30}
        }' \
    --teleop.type=bi_so100_leader \
    --teleop.left_arm_port=/dev/ttyACM2 \
    --teleop.right_arm_port=/dev/ttyACM3 \
    --teleop.id=bimanual_leader \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/bimanual-right-basket-right-rblock \
    --dataset.num_episodes=30 \
    --dataset.single_task="Grab the red cube and put it in a red basket" \
    --resume=true
  ```
train
```bash
accelerate launch --mixed_precision=bf16 $(which lerobot-train) \
  --dataset.repo_id=YieumYoon/recode-bimanual-red-block-basket-right-arm \
  --policy.type=groot \
  --policy.repo_id=YieumYoon/groot-bimanual-so100-right-arm-001 \
  --policy.push_to_hub=true \
  --batch_size=32 \
  --num_workers=8 \
  --steps=20000 \
  --save_checkpoint=true \
  --save_freq=1000 \
  --log_freq=10 \
  --wandb.enable=true \
  --wandb.project=lerobot-bimanual-so100-brev \
  --wandb.notes='L40S-Brev-right-arm-001' \
  --policy.tune_diffusion_model=false \
  --output_dir=outputs/train/groot-bimanual-so100-right-arm-001
```


# Merge train and validation splits back into one dataset
lerobot-edit-dataset \
    --repo_id YieumYoon/bimanual-center-basket-rblock-rlmerged-test00/ \
    --operation.type merge \
    --operation.repo_ids "['YieumYoon/bimanual-center-basket-right-rblock', 'YieumYoon/bimanual-center-basket-left-rblock']"


sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1 /dev/ttyACM2 /dev/ttyACM3

Gr00t inference server
```bash
python -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=8080
```

Groot Client server
```bash
python -m lerobot.async_inference.robot_client \
    --robot.type=bi_so100_follower \
    --robot.left_arm_port=/dev/ttyACM2 \
    --robot.right_arm_port=/dev/ttyACM1 \
    --robot.id=bimanual_follower \
    --robot.cameras='{
        left_gripper: {"type": "opencv", "index_or_path": "/dev/video5", "width": 640, "height": 480, "fps": 30},
        top: {"type": "opencv", "index_or_path": "/dev/video0", "width": 640, "height": 480, "fps": 30, "rotation": "ROTATE_180"},
        right_gripper: {"type": "opencv", "index_or_path": "/dev/video2", "width": 640, "height": 480, "fps": 30}
    }' \
    --task="Grab the red cube and put it in a red basket" \
    --server_address=127.0.0.1:8080 \
    --policy_type=groot \
    --pretrained_name_or_path=YieumYoon/groot-bimanual-so100-crlbasket-004 \
    --policy_device=cuda \
    --actions_per_chunk=16 \
    --embodiment_tag=new_embodiment
  ```