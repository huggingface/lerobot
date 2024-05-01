"""
This scripts demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run examples/3_train_policy.py first.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

from lerobot.scripts.eval import eval

# Get a pretrained policy from the hub.
pretrained_policy_name = "lerobot/diffusion_policy_pusht_image"
pretrained_policy_path = Path(snapshot_download(pretrained_policy_name))
# OR uncomment the following to evaluate a policy from the local outputs/train folder.
# pretrained_policy_path = Path("outputs/train/example_pusht_diffusion")

# Override some config parameters to do with evaluation.
overrides = [
    "eval.n_episodes=10",
    "eval.batch_size=10",
    "device=cuda",
]

# Evaluate the policy and save the outputs including metrics and videos.
# TODO(rcadene, alexander-soare): dont call eval, but add the minimal code snippet to rollout
eval(pretrained_policy_path=pretrained_policy_path)
