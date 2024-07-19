from lerobot.scripts.train import train
from lerobot.scripts.eval import get_pretrained_policy_path
from lerobot.common.utils.utils import init_hydra_config
import shutil
import hydra
from pathlib import Path


policy_repo_id = "lerobot/diffusion_pusht"
output_dir = "outputs/test_mps_operations"


def prepare_checkpoint_dir(pretrained_policy_name_or_path, output_folder):
    pretrained_policy_path = get_pretrained_policy_path(pretrained_policy_name_or_path)
    last_checkpoint_dir = Path(output_folder) / "last"
    last_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for item in pretrained_policy_path.iterdir():
        dest = last_checkpoint_dir / item.name
        if item.is_file():
            shutil.copy(item, dest)
        else:
            shutil.copytree(item, dest)
    
    return last_checkpoint_dir

overrides = ["device=mps", "training.offline_steps=110000", "resume=true"]

hydra_cfg = init_hydra_config(prepare_checkpoint_dir(policy_repo_id, output_dir) / "config.yaml", overrides=overrides)
