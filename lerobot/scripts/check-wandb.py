import wandb
from lerobot.common.logger import Logger
from omegaconf import OmegaConf


class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        for k, v in kwargs.items():
            if isinstance(v, dict):
                self.__dict__[k] = Config(**v)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

cfg = OmegaConf.create({
    "dataset_repo_id": "my_dataset",
    "log_dir": "logs",
    "wandb_job_name": "my_job",
    "wandb": {
        "enable": True,
        "project": "my_project",
        "tags": ["tag1", "tag2"],
        "group": "my_group",
    },
    "policy": {
        "policy1": 1,
        "name": "policy1",
    },
    "env": {
        "name": "env1",
    },
    "seed": 42,
    "resume": False,
})
log = Logger(cfg=cfg, log_dir = 'outputs/wandb_log_dir', wandb_job_name='wtf111' )
for i in range(10):
    # training step
    loss = 0.1  # example loss
    log.log_dict({"loss": loss}, step=i)
    print(f"Step {i}: loss = {loss}")