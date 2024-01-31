import hydra
import torch
from termcolor import colored

from lerobot.common.envs.factory import make_env
from lerobot.common.tdmpc import TDMPC

from ..common.utils import set_seed


@hydra.main(version_base=None, config_name="default", config_path="../configs")
def train(cfg: dict):
    assert torch.cuda.is_available()
    set_seed(cfg.seed)
    print(colored("Work dir:", "yellow", attrs=["bold"]), cfg.log_dir)

    env = make_env(cfg)
    agent = TDMPC(cfg)
    # ckpt_path = "/home/rcadene/code/fowm/logs/xarm_lift/all/default/2/models/offline.pt"
    ckpt_path = "/home/rcadene/code/fowm/logs/xarm_lift/all/default/2/models/final.pt"
    agent.load(ckpt_path)

    # online training

    eval_metrics = train_agent(
        env,
        agent,
        num_episodes=10,
        save_video=True,
        video_dir=Path("tmp/2023_01_29_xarm_lift_final"),
    )

    print(eval_metrics)


if __name__ == "__main__":
    train()
