
# マルチタスク設定
multi_task='["real_data/green_paint_red_plate","real_data/pen_black_plate","real_data/box_black_plate"]'

# データセットパス
repo_id="real_data/multi_task"

# データセットのマージ
lerobot-edit-dataset \
  --repo_id ${repo_id} \
  --operation.type merge \
  --operation.repo_ids ${multi_task}