import logging
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from tensordict.nn import TensorDictModule

from lerobot.common.datasets.factory import make_offline_buffer
from lerobot.common.envs.factory import make_env
from lerobot.common.logger import log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils import get_safe_torch_device, init_logging, set_seed
from lerobot.scripts.eval import eval_policy

folder = Path(snapshot_download("lerobot/diffusion_policy_pusht_image", revision="v1.0"))
cfg = OmegaConf.load(folder / "config.yaml")
cfg.policy.pretrained_model_path = folder / "model.pt"
cfg.eval_episodes = 1
cfg.episode_length = 50
# cfg.device = "cpu"

out_dir = "tmp/"

if out_dir is None:
    raise NotImplementedError()

init_logging()

# Check device is available
get_safe_torch_device(cfg.device, log=True)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
set_seed(cfg.seed)

log_output_dir(out_dir)

logging.info("make_offline_buffer")
offline_buffer = make_offline_buffer(cfg)

logging.info("make_env")
env = make_env(cfg, transform=offline_buffer.transform)

if cfg.policy.pretrained_model_path:
    policy = make_policy(cfg)
    policy = TensorDictModule(
        policy,
        in_keys=["observation", "step_count"],
        out_keys=["action"],
    )
else:
    # when policy is None, rollout a random policy
    policy = None

metrics = eval_policy(
    env,
    policy=policy,
    save_video=True,
    video_dir=Path(out_dir) / "eval",
    fps=cfg.env.fps,
    max_steps=cfg.env.episode_length,
    num_episodes=cfg.eval_episodes,
)
print(metrics)

logging.info("End of eval")
