# Just teleop
python lerobot/scripts/control_robot.py --robot.type=aloha --robot.cameras='{}' --control.type=teleoperate

python lerobot/scripts/control_robot.py --robot.type=aloha --robot.cameras='{"cam_kinect": {"type": "opencv", "camera_index": 0, "fps": 30, "width": 1280, "height": 720}}' --control.type=teleoperate --control.display_data=true


# Record one episode locally
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Grasp mug and place it on the table." --control.repo_id=$USER/aloha_test --control.num_episodes=1 --robot.cameras='{}' --control.push_to_hub=false --control.fps=200
# with camera
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Grasp mug and place it on the table." --control.repo_id=$USER/aloha_test --control.num_episodes=1 --robot.cameras='{"cam_kinect": {"type": "opencv", "camera_index": 0, "fps": 30, "width": 1280, "height": 720}}' --control.push_to_hub=false --control.fps=200

# Visualize 
python lerobot/scripts/visualize_dataset.py  --repo-id ${USER}/aloha_test   --episode-index 0
# Replay
python lerobot/scripts/control_robot.py --robot.type=aloha --robot.cameras='{"cam_kinect": {"type": "opencv", "camera_index": 0, "fps": 30, "width": 1280, "height": 720}}' --control.type=replay --control.fps=60 --control.repo_id=${USER}/aloha_test --control.episode=0 

# Record dataset
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Grasp mug and place it on the table." --control.repo_id=$USER/aloha_mug --control.num_episodes=100 --robot.cameras='{"cam_kinect": {"type": "opencv", "camera_index": 0, "fps": 30, "width": 1280, "height": 720}}' --control.push_to_hub=false --control.fps=60 --control.reset_time_s=5 --control.warmup_time_s=3

# Train
python lerobot/scripts/train.py --dataset.repo_id=${USER}/aloha_test --policy.type=diffusion --output_dir=outputs/train/diffPo_aloha_test --job_name=diifPo_aloha_test --policy.device=cuda --wandb.enable=true

# Rollout
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=60 --control.single_task="Grasp mug and place it on the table." --control.repo_id=$USER/eval_aloha_mug --control.num_episodes=1 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_kinect": {"type": "opencv", "camera_index": 0, "fps": 30, "width": 1280, "height": 720}}' --control.push_to_hub=false --control.policy.path=outputs/train/diffPo_aloha_mug/checkpoints/last/pretrained_model/
