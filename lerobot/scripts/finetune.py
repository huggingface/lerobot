from lerobot.scripts.train import train
from lerobot.scripts.eval import get_pretrained_policy_path
from lerobot.common.utils.utils import init_hydra_config
import shutil
from pathlib import Path
import os
import yaml


policy_repo_id = "m1b/red_box_training_state"
output_dir = "outputs/test_mps_operations/tmp"

def load_pretrained_model(repo_id, output_dir):
    pretrained_policy_path = get_pretrained_policy_path(repo_id)
    model_dir = Path(output_dir) / "checkpoints" / "000000"
    shutil.copytree(pretrained_policy_path, model_dir, dirs_exist_ok=True)
    symlink_path = Path(output_dir) / "checkpoints" / "last"
    if os.path.islink(symlink_path):
        os.remove(symlink_path)
    os.symlink(os.path.abspath(str(model_dir)), os.path.abspath(str(symlink_path)))
    print(f"Symlink created pointing to '{model_dir}'.")
    return model_dir

def modify_config(model_dir, overrides):
    config_path = model_dir / "pretrained_model" / "config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    for override in overrides:
        key, value = override.split('=')
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file)

def main():
    overrides = ["device=mps", "training.offline_steps=80000", "resume=true"]
    model_dir = load_pretrained_model(policy_repo_id, output_dir)
    modify_config(model_dir, overrides)
    
    config_path = model_dir / "pretrained_model" / "config.yaml"
    hydra_cfg = init_hydra_config(config_path)
    train(hydra_cfg, output_dir, job_name="mps_test")

if __name__ == "__main__":
    main()