"""
This scripts demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run examples/3_train_policy.py first.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

from lerobot.common.utils.utils import init_hydra_config
from lerobot.scripts.eval import eval

# Get a pretrained policy from the hub.
# TODO(alexander-soare): This no longer works until we upload a new model that uses the current configs.
hub_id = "lerobot/diffusion_policy_pusht_image"
folder = Path(snapshot_download(hub_id))
# OR uncomment the following to evaluate a policy from the local outputs/train folder.
# folder = Path("outputs/train/example_pusht_diffusion")

config_path = folder / "config.yaml"
weights_path = folder / "model.pt"
stats_path = folder / "stats.pth"  # normalization stats

# Override some config parameters to do with evaluation.
overrides = [
    f"policy.pretrained_model_path={weights_path}",
    "eval_episodes=10",
    "rollout_batch_size=10",
    "device=cuda",
]

# Create a Hydra config.
cfg = init_hydra_config(config_path, overrides)

# Evaluate the policy and save the outputs including metrics and videos.
eval(
    cfg,
    out_dir=f"outputs/eval/example_{cfg.env.name}_{cfg.policy.name}",
    stats_path=stats_path,
)
