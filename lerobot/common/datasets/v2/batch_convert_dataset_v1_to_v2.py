#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Note: Since the original Aloha datasets don't use shadow motors, you need to comment those out in
lerobot/configs/robot/aloha.yaml before running this script.
"""

import traceback
from pathlib import Path
from textwrap import dedent

from lerobot.common.datasets.v2.convert_dataset_v1_to_v2 import convert_dataset, parse_robot_config

LOCAL_DIR = Path("data/")

ALOHA_CONFIG = Path("lerobot/configs/robot/aloha.yaml")
ALOHA_MOBILE_INFO = {
    "robot_config": parse_robot_config(ALOHA_CONFIG),
    "license": "mit",
    "url": "https://mobile-aloha.github.io/",
    "paper": "https://arxiv.org/abs/2401.02117",
    "citation_bibtex": dedent("""
        @inproceedings{fu2024mobile,
            author    = {Fu, Zipeng and Zhao, Tony Z. and Finn, Chelsea},
            title     = {Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation},
            booktitle = {arXiv},
            year      = {2024},
        }""").lstrip(),
}
ALOHA_STATIC_INFO = {
    "robot_config": parse_robot_config(ALOHA_CONFIG),
    "license": "mit",
    "url": "https://tonyzhaozh.github.io/aloha/",
    "paper": "https://arxiv.org/abs/2304.13705",
    "citation_bibtex": dedent("""
        @article{Zhao2023LearningFB,
            title={Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
            author={Tony Zhao and Vikash Kumar and Sergey Levine and Chelsea Finn},
            journal={RSS},
            year={2023},
            volume={abs/2304.13705},
            url={https://arxiv.org/abs/2304.13705}
        }""").lstrip(),
}
PUSHT_INFO = {
    "license": "mit",
    "url": "https://diffusion-policy.cs.columbia.edu/",
    "paper": "https://arxiv.org/abs/2303.04137v5",
    "citation_bibtex": dedent("""
        @article{chi2024diffusionpolicy,
            author = {Cheng Chi and Zhenjia Xu and Siyuan Feng and Eric Cousineau and Yilun Du and Benjamin Burchfiel and Russ Tedrake and Shuran Song},
            title ={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
            journal = {The International Journal of Robotics Research},
            year = {2024},
        }""").lstrip(),
}
XARM_INFO = {
    "license": "mit",
    "url": "https://www.nicklashansen.com/td-mpc/",
    "paper": "https://arxiv.org/abs/2203.04955",
    "citation_bibtex": dedent("""
        @inproceedings{Hansen2022tdmpc,
            title={Temporal Difference Learning for Model Predictive Control},
            author={Nicklas Hansen and Xiaolong Wang and Hao Su},
            booktitle={ICML},
            year={2022}
        }
    """),
}
UNITREEH_INFO = {
    "license": "apache-2.0",
}


DATASETS = {
    "aloha_mobile_cabinet": {
        "single_task": "Open the top cabinet, store the pot inside it then close the cabinet.",
        **ALOHA_MOBILE_INFO,
    },
    "aloha_mobile_chair": {
        "single_task": "Push the chairs in front of the desk to place them against it.",
        **ALOHA_MOBILE_INFO,
    },
    "aloha_mobile_elevator": {
        "single_task": "Take the elevator to the 1st floor.",
        **ALOHA_MOBILE_INFO,
    },
    "aloha_mobile_shrimp": {
        "single_task": "Sauté the raw shrimp on both sides, then serve it in the bowl.",
        **ALOHA_MOBILE_INFO,
    },
    "aloha_mobile_wash_pan": {
        "single_task": "Pick up the pan, rinse it in the sink and then place it in the drying rack.",
        **ALOHA_MOBILE_INFO,
    },
    "aloha_mobile_wipe_wine": {
        "single_task": "Pick up the wet cloth on the faucet and use it to clean the spilled wine on the table and underneath the glass.",
        **ALOHA_MOBILE_INFO,
    },
    "aloha_static_battery": {
        "single_task": "Place the battery into the slot of the remote controller.",
        **ALOHA_STATIC_INFO,
    },
    "aloha_static_candy": {"single_task": "Pick up the candy and unwrap it.", **ALOHA_STATIC_INFO},
    "aloha_static_coffee": {
        "single_task": "Place the coffee capsule inside the capsule container, then place the cup onto the center of the cup tray, then push the 'Hot Water' and 'Travel Mug' buttons.",
        **ALOHA_STATIC_INFO,
    },
    "aloha_static_coffee_new": {
        "single_task": "Place the coffee capsule inside the capsule container, then place the cup onto the center of the cup tray.",
        **ALOHA_STATIC_INFO,
    },
    "aloha_static_cups_open": {
        "single_task": "Pick up the plastic cup and open its lid.",
        **ALOHA_STATIC_INFO,
    },
    "aloha_static_fork_pick_up": {
        "single_task": "Pick up the fork and place it on the plate.",
        **ALOHA_STATIC_INFO,
    },
    "aloha_static_pingpong_test": {
        "single_task": "Transfer one of the two balls in the right glass into the left glass, then transfer it back to the right glass.",
        **ALOHA_STATIC_INFO,
    },
    "aloha_static_pro_pencil": {
        "single_task": "Pick up the pencil with the right arm, hand it over to the left arm then place it back onto the table.",
        **ALOHA_STATIC_INFO,
    },
    "aloha_static_screw_driver": {
        "single_task": "Pick up the screwdriver with the right arm, hand it over to the left arm then place it into the cup.",
        **ALOHA_STATIC_INFO,
    },
    "aloha_static_tape": {
        "single_task": "Cut a small piece of tape from the tape dispenser then place it on the cardboard box's edge.",
        **ALOHA_STATIC_INFO,
    },
    "aloha_static_thread_velcro": {
        "single_task": "Pick up the velcro cable tie with the left arm, then insert the end of the velcro tie into the other end's loop with the right arm.",
        **ALOHA_STATIC_INFO,
    },
    "aloha_static_towel": {
        "single_task": "Pick up a piece of paper towel and place it on the spilled liquid.",
        **ALOHA_STATIC_INFO,
    },
    "aloha_static_vinh_cup": {
        "single_task": "Pick up the platic cup with the right arm, then pop its lid open with the left arm.",
        **ALOHA_STATIC_INFO,
    },
    "aloha_static_vinh_cup_left": {
        "single_task": "Pick up the platic cup with the left arm, then pop its lid open with the right arm.",
        **ALOHA_STATIC_INFO,
    },
    "aloha_static_ziploc_slide": {"single_task": "Slide open the ziploc bag.", **ALOHA_STATIC_INFO},
    "aloha_sim_insertion_scripted": {"single_task": "Insert the peg into the socket.", **ALOHA_STATIC_INFO},
    "aloha_sim_insertion_scripted_image": {
        "single_task": "Insert the peg into the socket.",
        **ALOHA_STATIC_INFO,
    },
    "aloha_sim_insertion_human": {"single_task": "Insert the peg into the socket.", **ALOHA_STATIC_INFO},
    "aloha_sim_insertion_human_image": {
        "single_task": "Insert the peg into the socket.",
        **ALOHA_STATIC_INFO,
    },
    "aloha_sim_transfer_cube_scripted": {
        "single_task": "Pick up the cube with the right arm and transfer it to the left arm.",
        **ALOHA_STATIC_INFO,
    },
    "aloha_sim_transfer_cube_scripted_image": {
        "single_task": "Pick up the cube with the right arm and transfer it to the left arm.",
        **ALOHA_STATIC_INFO,
    },
    "aloha_sim_transfer_cube_human": {
        "single_task": "Pick up the cube with the right arm and transfer it to the left arm.",
        **ALOHA_STATIC_INFO,
    },
    "aloha_sim_transfer_cube_human_image": {
        "single_task": "Pick up the cube with the right arm and transfer it to the left arm.",
        **ALOHA_STATIC_INFO,
    },
    "pusht": {"single_task": "Push the T-shaped block onto the T-shaped target.", **PUSHT_INFO},
    "pusht_image": {"single_task": "Push the T-shaped block onto the T-shaped target.", **PUSHT_INFO},
    "unitreeh1_fold_clothes": {"single_task": "Fold the sweatshirt.", **UNITREEH_INFO},
    "unitreeh1_rearrange_objects": {"single_task": "Put the object into the bin.", **UNITREEH_INFO},
    "unitreeh1_two_robot_greeting": {
        "single_task": "Greet the other robot with a high five.",
        **UNITREEH_INFO,
    },
    "unitreeh1_warehouse": {
        "single_task": "Grab the spray paint on the shelf and place it in the bin on top of the robot dog.",
        **UNITREEH_INFO,
    },
    "xarm_lift_medium": {"single_task": "Pick up the cube and lift it.", **XARM_INFO},
    "xarm_lift_medium_image": {"single_task": "Pick up the cube and lift it.", **XARM_INFO},
    "xarm_lift_medium_replay": {"single_task": "Pick up the cube and lift it.", **XARM_INFO},
    "xarm_lift_medium_replay_image": {"single_task": "Pick up the cube and lift it.", **XARM_INFO},
    "xarm_push_medium": {"single_task": "Push the cube onto the target.", **XARM_INFO},
    "xarm_push_medium_image": {"single_task": "Push the cube onto the target.", **XARM_INFO},
    "xarm_push_medium_replay": {"single_task": "Push the cube onto the target.", **XARM_INFO},
    "xarm_push_medium_replay_image": {"single_task": "Push the cube onto the target.", **XARM_INFO},
    "umi_cup_in_the_wild": {
        "single_task": "Put the cup on the plate.",
        "license": "apache-2.0",
    },
}


def batch_convert():
    status = {}
    logfile = LOCAL_DIR / "conversion_log.txt"
    # assert set(DATASETS) == set(id_.split("/")[1] for id_ in available_datasets)
    for num, (name, kwargs) in enumerate(DATASETS.items()):
        repo_id = f"lerobot/{name}"
        print(f"\nConverting {repo_id} ({num}/{len(DATASETS)})")
        print("---------------------------------------------------------")
        try:
            convert_dataset(repo_id, LOCAL_DIR, **kwargs)
            status = f"{repo_id}: success."
            with open(logfile, "a") as file:
                file.write(status + "\n")
        except Exception:
            status = f"{repo_id}: failed\n    {traceback.format_exc()}"
            with open(logfile, "a") as file:
                file.write(status + "\n")
            continue


if __name__ == "__main__":
    batch_convert()
