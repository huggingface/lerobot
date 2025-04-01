from contextlib import nullcontext
from pprint import pformat
from termcolor import colored
import dataclasses
import logging
from typing import Iterator

import wandb
import numpy as np
import torch

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.utils import cycle
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import (
    init_logging,
    get_safe_torch_device,
)
from lerobot.common.utils.random_utils import set_seed
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig

class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: lerobot_dataset.LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)

@parser.wrap()
def main(cfg: TrainPipelineConfig):
    init_logging()
    logging.info(pformat(dataclasses.asdict(cfg)))

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    wandb.init(
        name=cfg.job_name,
        config=dataclasses.asdict(cfg),
        project="lerobot",
    )
    
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making dataset.")
    dataset = make_dataset(cfg)
    dataset.meta.info['features']['observation.state']['shape'] = (13,)
    dataset.meta.info['features']['observation.state']['names'] = ['left_arm_1', 'left_arm_2', 'left_arm_3', 'left_arm_4', 'left_arm_5', 'left_arm_6', 
                                                               'right_arm_1', 'right_arm_2', 'right_arm_3', 'right_arm_4', 'right_arm_5', 'right_arm_6', 'vacuum']
    dataset.meta.info['features']['action']['shape'] = (13,)
    dataset.meta.info['features']['action']['names'] = ['left_arm_exp_1', 'left_arm_exp_2', 'left_arm_exp_3', 'left_arm_exp_4', 'left_arm_exp_5', 'left_arm_exp_6', 
                                                    'right_arm_exp_1', 'right_arm_exp_2', 'right_arm_exp_3', 'right_arm_exp_4', 'right_arm_exp_5', 'right_arm_exp_6', 'vacuum_exp']
    
    for key, episode_stats in dataset.meta.episodes_stats.items():
                for feature in ['observation.state', 'action']:
                    for sub_feature in ['min', 'max', 'mean', 'std']:
                        episode_stats[feature][sub_feature] = np.delete(episode_stats[feature][sub_feature], [6,13], axis=0)
                        dataset.meta.episodes_stats[key] = episode_stats
    for feature in ['min', 'max', 'mean', 'std']:
        dataset.meta.stats['observation.state'][feature] = np.delete(dataset.meta.stats['observation.state'][feature], [6,13], axis=0)
        dataset.meta.stats['action'][feature] = np.delete(dataset.meta.stats['action'][feature], [6,13], axis=0)
    

    logging.info("Making policy.")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )
    
    # TODO: 支持多条，从配置读取
    episode_index = 0
    
    sampler = EpisodeSampler(
        dataset,  episode_index
    )
    data_len = len(sampler)
    print("entire data len:", data_len)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        sampler=sampler,
    )
    
    policy.eval()
    
    # aloha
    # dim_names = [
    #     "arm_left_0", "arm_left_1", "arm_left_2", "arm_left_3", "arm_left_4", "arm_left_5",
    #     "arm_left_gripper_0", "arm_left_gripper_1",
    #     "arm_right_0", "arm_right_1", "arm_right_2", "arm_right_3", "arm_right_4", "arm_right_5",
    #     "arm_right_gripper_0", "arm_right_gripper_1",
    # ]
    
    # galaxea vacumn - 13 dim
    dim_names = [
        "arm_left_0", "arm_left_1", "arm_left_2", "arm_left_3", "arm_left_4", "arm_left_5",
        "arm_right_0", "arm_right_1", "arm_right_2", "arm_right_3", "arm_right_4", "arm_right_5",
        "arm_right_vacumn",
    ]
    # galaxea vacumn - 15 dim
    # dim_names = [
    #     "arm_left_0", "arm_left_1", "arm_left_2", "arm_left_3", "arm_left_4", "arm_left_5", "arm_left_gripper",
    #     "arm_right_0", "arm_right_1", "arm_right_2", "arm_right_3", "arm_right_4", "arm_right_5","arm_right_gripper",
    #     "arm_right_vacumn",
    # ]
    
    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        # infer_action = policy.forward(input_raw_data)
        dl_iter = cycle(dataloader)
        
        # skip the first few frames
        for i in range(50):
            a = next(dl_iter)
        
        
        for step in range(200):
            # 准备数据
            input_raw_data = next(dl_iter)
            
            # shape记录 galaxea
            # observation.images.front: torch.Size([batch, 3, 360, 640])
            # observation.images.wrist_right: torch.Size([batch, 3, 480, 640])
            # observation.state torch.Size([batch, 15])
            if 'observation.images.depth' in input_raw_data:
                del input_raw_data['observation.images.depth']
                
            if input_raw_data["observation.state"].shape[-1] == 15:
                keep_cols = [i for i in range(15) if i not in {6, 13}]
                # 切片操作删除指定列
                input_raw_data["observation.state"] = input_raw_data["observation.state"][..., keep_cols]
                print("state shape", input_raw_data["observation.state"].shape)
                
            if input_raw_data["action"].shape[-1] == 15:
                keep_cols = [i for i in range(15) if i not in {6, 13}]
                input_raw_data["action"] = input_raw_data["action"][..., keep_cols]
                print("action shape", input_raw_data["action"].shape)
            
            for key in input_raw_data:
                if isinstance(input_raw_data[key], torch.Tensor):
                    input_raw_data[key] = input_raw_data[key].to(device, non_blocking=True)
                    
            # the front camera resolution is 640x360, pad it to 640x480 which is the resolution of the wrist camera
            input_raw_data['observation.images.front'] = torch.nn.functional.pad(
                input_raw_data['observation.images.front'], 
                pad=(0, 0, 60, 60),  # 左右不填充，上下各填充60
                mode='constant', 
                value=0  # black
            )
            logging.warning("pad the front images to ", input_raw_data['observation.images.front'].shape)
            
            # 推理
            infer_action = policy.select_action(input_raw_data)
            infer_action = infer_action[0].cpu().numpy()
            
            
            origin_action = input_raw_data["action"].cpu().numpy()[0][0]
            
            # 记录每个action dim的差异
            diff = infer_action - origin_action
            log_dict = {}
            for dim in range(cfg.policy.output_features["action"].shape[0]):
                key = f"{dim_names[dim]}_diff"
                log_dict[key] = diff[dim]
            
            wandb.log(log_dict)


if __name__ == "__main__":
    main()

