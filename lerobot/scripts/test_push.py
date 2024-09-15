from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.push_dataset_to_hub import push_meta_data_to_hub, push_videos_to_hub, create_branch
from pathlib import Path

repo_id = 'bchandaka/acm_move_square'
root = 'data'

dataset = LeRobotDataset(repo_id, root=root)
dataset.hf_dataset.push_to_hub(repo_id, revision="main")
local_dir = Path('./data/bchandaka/acm_move_square')
meta_data_dir = local_dir / "meta_data"
videos_dir = local_dir / "videos"
push_meta_data_to_hub(repo_id, meta_data_dir, revision="main")
push_videos_to_hub(repo_id, videos_dir, revision="main")
create_branch(repo_id, repo_type="dataset", branch='v1.6')

