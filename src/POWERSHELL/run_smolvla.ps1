
# "Pick up the pink tape and place it in the blue box." `
# "Pick up the Lego and place it on the blue mark" `
python -m lerobot.record `
  --robot.type=so101_follower `
  --robot.port=COM6 `
  --robot.id=follower `
  --robot.cameras '{                                 
    "front": { "type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30 },
    "ego":   { "type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30 }
  }' `
  --policy.path=aiden-li/so101-smolvla-picktape `
  --policy.device=cuda `
  --dataset.repo_id=aiden-li/eval_so101-smolvla-picktape `
  --dataset.single_task="Pick up the pink tape and put it in the blue box" `
  --display_data=true `
  --dataset.push_to_hub=false
