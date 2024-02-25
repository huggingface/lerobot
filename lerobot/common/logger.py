import datetime
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from termcolor import colored

CONSOLE_FORMAT = [
    ("episode", "E", "int"),
    ("env_step", "S", "int"),
    ("avg_sum_reward", "RS", "float"),
    ("avg_max_reward", "RM", "float"),
    ("pc_success", "S", "float"),
    ("total_time", "T", "time"),
]
AGENT_METRICS = [
    "consistency_loss",
    "reward_loss",
    "value_loss",
    "total_loss",
    "weighted_loss",
    "pi_loss",
    "grad_norm",
]


def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return dir_path


def print_run(cfg, reward=None):
    """Pretty-printing of run information. Call at start of training."""
    prefix, color, attrs = "  ", "green", ["bold"]

    def limstr(s, maxlen=32):
        return str(s[:maxlen]) + "..." if len(str(s)) > maxlen else s

    def pprint(k, v):
        print(
            prefix + colored(f'{k.capitalize() + ":":<16}', color, attrs=attrs),
            limstr(v),
        )

    kvs = [
        ("task", cfg.env.task),
        ("train steps", f"{int(cfg.train_steps * cfg.env.action_repeat):,}"),
        # ('observations', 'x'.join([str(s) for s in cfg.obs_shape])),
        # ('actions', cfg.action_dim),
        # ('experiment', cfg.exp_name),
    ]
    if reward is not None:
        kvs.append(
            ("episode reward", colored(str(int(reward)), "white", attrs=["bold"]))
        )
    w = np.max([len(limstr(str(kv[1]))) for kv in kvs]) + 21
    div = "-" * w
    print(div)
    for k, v in kvs:
        pprint(k, v)
    print(div)


def cfg_to_group(cfg, return_list=False):
    """Return a wandb-safe group name for logging. Optionally returns group name as list."""
    # lst = [cfg.task, cfg.modality, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
    lst = [
        f"env:{cfg.env.name}",
        f"seed:{cfg.seed}",
    ]
    return lst if return_list else "-".join(lst)


class VideoRecorder:
    """Utility class for logging evaluation videos."""

    def __init__(self, root_dir, wandb, render_size=384, fps=15):
        self.save_dir = (root_dir / "eval_video") if root_dir else None
        self._wandb = wandb
        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.enabled = False
        self.camera_id = 0

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir and self._wandb and enabled
        try:
            env_name = env.unwrapped.spec.id
        except:
            env_name = ""
        if "maze2d" in env_name:
            self.camera_id = -1
        elif "quadruped" in env_name:
            self.camera_id = 2
        self.record(env)

    def record(self, env):
        if self.enabled:
            frame = env.render(
                mode="rgb_array",
                height=self.render_size,
                width=self.render_size,
                camera_id=self.camera_id,
            )
            self.frames.append(frame)

    def save(self, step):
        if self.enabled:
            frames = np.stack(self.frames).transpose(0, 3, 1, 2)
            self._wandb.log(
                {
                    "eval_video": self._wandb.Video(
                        frames, fps=self.env.fps, format="mp4"
                    )
                },
                step=step,
            )


class Logger(object):
    """Primary logger object. Logs either locally or using wandb."""

    def __init__(self, log_dir, job_name, cfg):
        self._log_dir = make_dir(Path(log_dir))
        self._job_name = job_name
        self._model_dir = make_dir(self._log_dir / "models")
        self._buffer_dir = make_dir(self._log_dir / "buffers")
        self._save_model = cfg.save_model
        self._save_buffer = cfg.save_buffer
        self._group = cfg_to_group(cfg)
        self._seed = cfg.seed
        self._cfg = cfg
        self._eval = []
        print_run(cfg)
        project = cfg.get("wandb", {}).get("project")
        entity = cfg.get("wandb", {}).get("entity")
        enable_wandb = cfg.get("wandb", {}).get("enable", False)
        run_offline = not enable_wandb or not project or not entity
        if run_offline:
            print(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
            self._wandb = None
        else:
            os.environ["WANDB_SILENT"] = "true"
            import wandb

            wandb.init(
                project=project,
                entity=entity,
                name=job_name,
                notes=cfg.get("wandb", {}).get("notes"),
                # group=self._group,
                tags=cfg_to_group(cfg, return_list=True),
                dir=self._log_dir,
                config=OmegaConf.to_container(cfg, resolve=True),
                # TODO(rcadene): try set to True
                save_code=False,
                # TODO(rcadene): split train and eval, and run async eval with job_type="eval"
                job_type="train_eval",
                # TODO(rcadene): add resume option
                resume=None,
            )
            print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
            self._wandb = wandb
        self._video = (
            VideoRecorder(self._log_dir, self._wandb)
            if self._wandb and cfg.save_video
            else None
        )

    @property
    def video(self):
        return self._video

    def save_model(self, agent, identifier):
        if self._save_model:
            fp = self._model_dir / f"{str(identifier)}.pt"
            agent.save(fp)
            if self._wandb:
                artifact = self._wandb.Artifact(
                    self._group + "-" + str(self._seed) + "-" + str(identifier),
                    type="model",
                )
                artifact.add_file(fp)
                self._wandb.log_artifact(artifact)

    def save_buffer(self, buffer, identifier):
        fp = self._buffer_dir / f"{str(identifier)}.pkl"
        buffer.save(fp)
        if self._wandb:
            artifact = self._wandb.Artifact(
                self._group + "-" + str(self._seed) + "-" + str(identifier),
                type="buffer",
            )
            artifact.add_file(fp)
            self._wandb.log_artifact(artifact)

    def finish(self, agent, buffer):
        if self._save_model:
            self.save_model(agent, identifier="final")
        if self._save_buffer:
            self.save_buffer(buffer, identifier="buffer")
        if self._wandb:
            self._wandb.finish()
        print_run(self._cfg, self._eval[-1][-1])

    def _format(self, key, value, ty):
        if ty == "int":
            return f'{colored(key + ":", "grey")} {int(value):,}'
        elif ty == "float":
            return f'{colored(key + ":", "grey")} {value:.01f}'
        elif ty == "time":
            value = str(datetime.timedelta(seconds=int(value)))
            return f'{colored(key + ":", "grey")} {value}'
        else:
            raise f"invalid log format type: {ty}"

    def _print(self, d, category):
        category = colored(category, "blue" if category == "train" else "green")
        pieces = [f" {category:<14}"]
        for k, disp_k, ty in CONSOLE_FORMAT:
            pieces.append(f"{self._format(disp_k, d.get(k, 0), ty):<26}")
        print("   ".join(pieces))

    def log(self, d, category="train"):
        assert category in {"train", "eval"}
        if self._wandb is not None:
            for k, v in d.items():
                self._wandb.log({category + "/" + k: v}, step=d["env_step"])
        if category == "eval":
            # keys = ['env_step', 'avg_reward']
            keys = ["env_step", "avg_sum_reward", "avg_max_reward", "pc_success"]
            self._eval.append(np.array([d[key] for key in keys]))
            pd.DataFrame(np.array(self._eval)).to_csv(
                self._log_dir / "eval.log", header=keys, index=None
            )
        self._print(d, category)
