
# データセットパス
# repo_id="real_data/green_paint_red_plate"
# job_name="green_paint_red_plate"
# repo_id="real_data/pen_black_plate"
# job_name="pen_black_plate"
# repo_id="real_data/box_black_plate"
# job_name="box_black_plate"
repo_id="real_data/multi_task"
job_name="multi_task"

# 50episode ⇒ 20000step 相当

seed_value=2000 # シード値
steps=60000 # 学習ステップ数

# 実行（事前学習あり）
lerobot-train \
  --policy.path=/home/masuoka/lerobot/smolvla_base \
  --dataset.repo_id=${repo_id} \
  --rename_map='{
    "observation.images.front": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2"
  }' \
  --dataset.video_backend=pyav \
  --output_dir=./outputs/pretrained/seed_${seed_value}/${job_name} \
  --policy.push_to_hub=false \
  --steps=${steps} \
  --job_name=${job_name}_pretrained_${seed_value} \
  --batch_size=64 \
  --policy.device=cuda \
  --seed=${seed_value} \
  --wandb.enable=true