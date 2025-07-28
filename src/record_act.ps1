python -m lerobot.record `
  --robot.type=so101_follower `
  --robot.port=COM3 `
  --robot.id=follower `
  --teleop.type=so101_leader `
  --teleop.port=COM5 `
  --teleop.id=leader `
  --dataset.single_task="Grab a piece of tissue." `
  --dataset.repo_id=aiden-li/so101-grabtisue `
  --resume=true `
  --push_to_hub=false `
  --display_data=true `
  --dataset.num_episodes=1

