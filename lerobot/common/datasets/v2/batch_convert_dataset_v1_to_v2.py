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
This script is for internal use to convert all datasets under the 'lerobot' hub user account to v2.

Note: Since the original Aloha datasets don't use shadow motors, you need to comment those out in
lerobot/configs/robot/aloha.yaml before running this script.
"""

import traceback
from pathlib import Path
from textwrap import dedent

from lerobot import available_datasets
from lerobot.common.datasets.v2.convert_dataset_v1_to_v2 import convert_dataset
from lerobot.common.robots.aloha.configuration_aloha import AlohaRobotConfig

LOCAL_DIR = Path("data/")

# spellchecker:off
ALOHA_MOBILE_INFO = {
    "robot_config": AlohaRobotConfig(),
    "license": "mit",
    "url": "https://mobile-aloha.github.io/",
    "paper": "https://arxiv.org/abs/2401.02117",
    "citation_bibtex": dedent(r"""
        @inproceedings{fu2024mobile,
            author    = {Fu, Zipeng and Zhao, Tony Z. and Finn, Chelsea},
            title     = {Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation},
            booktitle = {arXiv},
            year      = {2024},
        }""").lstrip(),
}
ALOHA_STATIC_INFO = {
    "robot_config": AlohaRobotConfig(),
    "license": "mit",
    "url": "https://tonyzhaozh.github.io/aloha/",
    "paper": "https://arxiv.org/abs/2304.13705",
    "citation_bibtex": dedent(r"""
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
    "citation_bibtex": dedent(r"""
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
    "citation_bibtex": dedent(r"""
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
        "single_task": "Pick up the plastic cup with the right arm, then pop its lid open with the left arm.",
        **ALOHA_STATIC_INFO,
    },
    "aloha_static_vinh_cup_left": {
        "single_task": "Pick up the plastic cup with the left arm, then pop its lid open with the right arm.",
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
    "asu_table_top": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "paper": "https://link.springer.com/article/10.1007/s10514-023-10129-1",
        "citation_bibtex": dedent(r"""
            @inproceedings{zhou2023modularity,
                title={Modularity through Attention: Efficient Training and Transfer of Language-Conditioned Policies for Robot Manipulation},
                author={Zhou, Yifan and Sonawani, Shubham and Phielipp, Mariano and Stepputtis, Simon and Amor, Heni},
                booktitle={Conference on Robot Learning},
                pages={1684--1695},
                year={2023},
                organization={PMLR}
            }
            @article{zhou2023learning,
                title={Learning modular language-conditioned robot policies through attention},
                author={Zhou, Yifan and Sonawani, Shubham and Phielipp, Mariano and Ben Amor, Heni and Stepputtis, Simon},
                journal={Autonomous Robots},
                pages={1--21},
                year={2023},
                publisher={Springer}
            }""").lstrip(),
    },
    "austin_buds_dataset": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://ut-austin-rpl.github.io/BUDS-website/",
        "paper": "https://arxiv.org/abs/2109.13841",
        "citation_bibtex": dedent(r"""
            @article{zhu2022bottom,
                title={Bottom-Up Skill Discovery From Unsegmented Demonstrations for Long-Horizon Robot Manipulation},
                author={Zhu, Yifeng and Stone, Peter and Zhu, Yuke},
                journal={IEEE Robotics and Automation Letters},
                volume={7},
                number={2},
                pages={4126--4133},
                year={2022},
                publisher={IEEE}
            }""").lstrip(),
    },
    "austin_sailor_dataset": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://ut-austin-rpl.github.io/sailor/",
        "paper": "https://arxiv.org/abs/2210.11435",
        "citation_bibtex": dedent(r"""
            @inproceedings{nasiriany2022sailor,
                title={Learning and Retrieval from Prior Data for Skill-based Imitation Learning},
                author={Soroush Nasiriany and Tian Gao and Ajay Mandlekar and Yuke Zhu},
                booktitle={Conference on Robot Learning (CoRL)},
                year={2022}
            }""").lstrip(),
    },
    "austin_sirius_dataset": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://ut-austin-rpl.github.io/sirius/",
        "paper": "https://arxiv.org/abs/2211.08416",
        "citation_bibtex": dedent(r"""
            @inproceedings{liu2022robot,
                title = {Robot Learning on the Job: Human-in-the-Loop Autonomy and Learning During Deployment},
                author = {Huihan Liu and Soroush Nasiriany and Lance Zhang and Zhiyao Bao and Yuke Zhu},
                booktitle = {Robotics: Science and Systems (RSS)},
                year = {2023}
            }""").lstrip(),
    },
    "berkeley_autolab_ur5": {
        "tasks_col": "language_instruction",
        "license": "cc-by-4.0",
        "url": "https://sites.google.com/view/berkeley-ur5/home",
        "citation_bibtex": dedent(r"""
            @misc{BerkeleyUR5Website,
                title = {Berkeley {UR5} Demonstration Dataset},
                author = {Lawrence Yunliang Chen and Simeon Adebola and Ken Goldberg},
                howpublished = {https://sites.google.com/view/berkeley-ur5/home},
            }""").lstrip(),
    },
    "berkeley_cable_routing": {
        "tasks_col": "language_instruction",
        "license": "cc-by-4.0",
        "url": "https://sites.google.com/view/cablerouting/home",
        "paper": "https://arxiv.org/abs/2307.08927",
        "citation_bibtex": dedent(r"""
            @article{luo2023multistage,
                author    = {Jianlan Luo and Charles Xu and Xinyang Geng and Gilbert Feng and Kuan Fang and Liam Tan and Stefan Schaal and Sergey Levine},
                title     = {Multi-Stage Cable Routing through Hierarchical Imitation Learning},
                journal   = {arXiv pre-print},
                year      = {2023},
                url       = {https://arxiv.org/abs/2307.08927},
            }""").lstrip(),
    },
    "berkeley_fanuc_manipulation": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://sites.google.com/berkeley.edu/fanuc-manipulation",
        "citation_bibtex": dedent(r"""
            @article{fanuc_manipulation2023,
                title={Fanuc Manipulation: A Dataset for Learning-based Manipulation with FANUC Mate 200iD Robot},
                author={Zhu, Xinghao and Tian, Ran and Xu, Chenfeng and Ding, Mingyu and Zhan, Wei and Tomizuka, Masayoshi},
                year={2023},
            }""").lstrip(),
    },
    "berkeley_gnm_cory_hall": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "paper": "https://arxiv.org/abs/1709.10489",
        "citation_bibtex": dedent(r"""
            @inproceedings{kahn2018self,
                title={Self-supervised deep reinforcement learning with generalized computation graphs for robot navigation},
                author={Kahn, Gregory and Villaflor, Adam and Ding, Bosen and Abbeel, Pieter and Levine, Sergey},
                booktitle={2018 IEEE international conference on robotics and automation (ICRA)},
                pages={5129--5136},
                year={2018},
                organization={IEEE}
            }""").lstrip(),
    },
    "berkeley_gnm_recon": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://sites.google.com/view/recon-robot",
        "paper": "https://arxiv.org/abs/2104.05859",
        "citation_bibtex": dedent(r"""
            @inproceedings{shah2021rapid,
                title={Rapid Exploration for Open-World Navigation with Latent Goal Models},
                author={Dhruv Shah and Benjamin Eysenbach and Nicholas Rhinehart and Sergey Levine},
                booktitle={5th Annual Conference on Robot Learning },
                year={2021},
                url={https://openreview.net/forum?id=d_SWJhyKfVw}
            }""").lstrip(),
    },
    "berkeley_gnm_sac_son": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://sites.google.com/view/SACSoN-review",
        "paper": "https://arxiv.org/abs/2306.01874",
        "citation_bibtex": dedent(r"""
            @article{hirose2023sacson,
                title={SACSoN: Scalable Autonomous Data Collection for Social Navigation},
                author={Hirose, Noriaki and Shah, Dhruv and Sridhar, Ajay and Levine, Sergey},
                journal={arXiv preprint arXiv:2306.01874},
                year={2023}
            }""").lstrip(),
    },
    "berkeley_mvp": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "paper": "https://arxiv.org/abs/2203.06173",
        "citation_bibtex": dedent(r"""
            @InProceedings{Radosavovic2022,
                title = {Real-World Robot Learning with Masked Visual Pre-training},
                author = {Ilija Radosavovic and Tete Xiao and Stephen James and Pieter Abbeel and Jitendra Malik and Trevor Darrell},
                booktitle = {CoRL},
                year = {2022}
            }""").lstrip(),
    },
    "berkeley_rpt": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "paper": "https://arxiv.org/abs/2306.10007",
        "citation_bibtex": dedent(r"""
            @article{Radosavovic2023,
                title={Robot Learning with Sensorimotor Pre-training},
                author={Ilija Radosavovic and Baifeng Shi and Letian Fu and Ken Goldberg and Trevor Darrell and Jitendra Malik},
                year={2023},
                journal={arXiv:2306.10007}
            }""").lstrip(),
    },
    "cmu_franka_exploration_dataset": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://human-world-model.github.io/",
        "paper": "https://arxiv.org/abs/2308.10901",
        "citation_bibtex": dedent(r"""
            @inproceedings{mendonca2023structured,
                title={Structured World Models from Human Videos},
                author={Mendonca, Russell  and Bahl, Shikhar and Pathak, Deepak},
                journal={RSS},
                year={2023}
            }""").lstrip(),
    },
    "cmu_play_fusion": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://play-fusion.github.io/",
        "paper": "https://arxiv.org/abs/2312.04549",
        "citation_bibtex": dedent(r"""
            @inproceedings{chen2023playfusion,
                title={PlayFusion: Skill Acquisition via Diffusion from Language-Annotated Play},
                author={Chen, Lili and Bahl, Shikhar and Pathak, Deepak},
                booktitle={CoRL},
                year={2023}
            }""").lstrip(),
    },
    "cmu_stretch": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://robo-affordances.github.io/",
        "paper": "https://arxiv.org/abs/2304.08488",
        "citation_bibtex": dedent(r"""
            @inproceedings{bahl2023affordances,
                title={Affordances from Human Videos as a Versatile Representation for Robotics},
                author={Bahl, Shikhar and Mendonca, Russell and Chen, Lili and Jain, Unnat and Pathak, Deepak},
                booktitle={CVPR},
                year={2023}
            }
                @article{mendonca2023structured,
                title={Structured World Models from Human Videos},
                author={Mendonca, Russell and Bahl, Shikhar and Pathak, Deepak},
                journal={CoRL},
                year={2023}
            }""").lstrip(),
    },
    "columbia_cairlab_pusht_real": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://diffusion-policy.cs.columbia.edu/",
        "paper": "https://arxiv.org/abs/2303.04137v5",
        "citation_bibtex": dedent(r"""
            @inproceedings{chi2023diffusionpolicy,
                title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
                author={Chi, Cheng and Feng, Siyuan and Du, Yilun and Xu, Zhenjia and Cousineau, Eric and Burchfiel, Benjamin and Song, Shuran},
                booktitle={Proceedings of Robotics: Science and Systems (RSS)},
                year={2023}
            }""").lstrip(),
    },
    "conq_hose_manipulation": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://sites.google.com/view/conq-hose-manipulation-dataset/home",
        "citation_bibtex": dedent(r"""
            @misc{ConqHoseManipData,
                author={Peter Mitrano and Dmitry Berenson},
                title={Conq Hose Manipulation Dataset, v1.15.0},
                year={2024},
                howpublished={https://sites.google.com/view/conq-hose-manipulation-dataset}
            }""").lstrip(),
    },
    "dlr_edan_shared_control": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "paper": "https://ieeexplore.ieee.org/document/9341156",
        "citation_bibtex": dedent(r"""
            @inproceedings{vogel_edan_2020,
                title = {EDAN - an EMG-Controlled Daily Assistant to Help People with Physical Disabilities},
                language = {en},
                booktitle = {2020 {IEEE}/{RSJ} {International} {Conference} on {Intelligent} {Robots} and {Systems} ({IROS})},
                author = {Vogel, Jörn and Hagengruber, Annette and Iskandar, Maged and Quere, Gabriel and Leipscher, Ulrike and Bustamante, Samuel and Dietrich, Alexander and Hoeppner, Hannes and Leidner, Daniel and Albu-Schäffer, Alin},
                year = {2020}
            }
            @inproceedings{quere_shared_2020,
                address = {Paris, France},
                title = {Shared {Control} {Templates} for {Assistive} {Robotics}},
                language = {en},
                booktitle = {2020 {IEEE} {International} {Conference} on {Robotics} and {Automation} ({ICRA})},
                author = {Quere, Gabriel and Hagengruber, Annette and Iskandar, Maged and Bustamante, Samuel and Leidner, Daniel and Stulp, Freek and Vogel, Joern},
                year = {2020},
                pages = {7},
            }""").lstrip(),
    },
    "dlr_sara_grid_clamp": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "paper": "https://www.researchsquare.com/article/rs-3289569/v1",
        "citation_bibtex": dedent(r"""
            @article{padalkar2023guided,
                title={A guided reinforcement learning approach using shared control templates for learning manipulation skills in the real world},
                author={Padalkar, Abhishek and Quere, Gabriel and Raffin, Antonin and Silv{\'e}rio, Jo{\~a}o and Stulp, Freek},
                journal={Research square preprint rs-3289569/v1},
                year={2023}
            }""").lstrip(),
    },
    "dlr_sara_pour": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "paper": "https://elib.dlr.de/193739/1/padalkar2023rlsct.pdf",
        "citation_bibtex": dedent(r"""
            @inproceedings{padalkar2023guiding,
                title={Guiding Reinforcement Learning with Shared Control Templates},
                author={Padalkar, Abhishek and Quere, Gabriel and Steinmetz, Franz and Raffin, Antonin and Nieuwenhuisen, Matthias and Silv{\'e}rio, Jo{\~a}o and Stulp, Freek},
                booktitle={40th IEEE International Conference on Robotics and Automation, ICRA 2023},
                year={2023},
                organization={IEEE}
            }""").lstrip(),
    },
    "droid_100": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://droid-dataset.github.io/",
        "paper": "https://arxiv.org/abs/2403.12945",
        "citation_bibtex": dedent(r"""
            @article{khazatsky2024droid,
                title   = {DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset},
                author  = {Alexander Khazatsky and Karl Pertsch and Suraj Nair and Ashwin Balakrishna and Sudeep Dasari and Siddharth Karamcheti and Soroush Nasiriany and Mohan Kumar Srirama and Lawrence Yunliang Chen and Kirsty Ellis and Peter David Fagan and Joey Hejna and Masha Itkina and Marion Lepert and Yecheng Jason Ma and Patrick Tree Miller and Jimmy Wu and Suneel Belkhale and Shivin Dass and Huy Ha and Arhan Jain and Abraham Lee and Youngwoon Lee and Marius Memmel and Sungjae Park and Ilija Radosavovic and Kaiyuan Wang and Albert Zhan and Kevin Black and Cheng Chi and Kyle Beltran Hatch and Shan Lin and Jingpei Lu and Jean Mercat and Abdul Rehman and Pannag R Sanketi and Archit Sharma and Cody Simpson and Quan Vuong and Homer Rich Walke and Blake Wulfe and Ted Xiao and Jonathan Heewon Yang and Arefeh Yavary and Tony Z. Zhao and Christopher Agia and Rohan Baijal and Mateo Guaman Castro and Daphne Chen and Qiuyu Chen and Trinity Chung and Jaimyn Drake and Ethan Paul Foster and Jensen Gao and David Antonio Herrera and Minho Heo and Kyle Hsu and Jiaheng Hu and Donovon Jackson and Charlotte Le and Yunshuang Li and Kevin Lin and Roy Lin and Zehan Ma and Abhiram Maddukuri and Suvir Mirchandani and Daniel Morton and Tony Nguyen and Abigail O'Neill and Rosario Scalise and Derick Seale and Victor Son and Stephen Tian and Emi Tran and Andrew E. Wang and Yilin Wu and Annie Xie and Jingyun Yang and Patrick Yin and Yunchu Zhang and Osbert Bastani and Glen Berseth and Jeannette Bohg and Ken Goldberg and Abhinav Gupta and Abhishek Gupta and Dinesh Jayaraman and Joseph J Lim and Jitendra Malik and Roberto Martín-Martín and Subramanian Ramamoorthy and Dorsa Sadigh and Shuran Song and Jiajun Wu and Michael C. Yip and Yuke Zhu and Thomas Kollar and Sergey Levine and Chelsea Finn},
                year    = {2024},
            }""").lstrip(),
    },
    "fmb": {
        "tasks_col": "language_instruction",
        "license": "cc-by-4.0",
        "url": "https://functional-manipulation-benchmark.github.io/",
        "paper": "https://arxiv.org/abs/2401.08553",
        "citation_bibtex": dedent(r"""
            @article{luo2024fmb,
                title={FMB: a Functional Manipulation Benchmark for Generalizable Robotic Learning},
                author={Luo, Jianlan and Xu, Charles and Liu, Fangchen and Tan, Liam and Lin, Zipeng and Wu, Jeffrey and Abbeel, Pieter and Levine, Sergey},
                journal={arXiv preprint arXiv:2401.08553},
                year={2024}
            }""").lstrip(),
    },
    "iamlab_cmu_pickup_insert": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://openreview.net/forum?id=WuBv9-IGDUA",
        "paper": "https://arxiv.org/abs/2401.14502",
        "citation_bibtex": dedent(r"""
            @inproceedings{saxena2023multiresolution,
                title={Multi-Resolution Sensing for Real-Time Control with Vision-Language Models},
                author={Saumya Saxena and Mohit Sharma and Oliver Kroemer},
                booktitle={7th Annual Conference on Robot Learning},
                year={2023},
                url={https://openreview.net/forum?id=WuBv9-IGDUA}
            }""").lstrip(),
    },
    "imperialcollege_sawyer_wrist_cam": {
        "tasks_col": "language_instruction",
        "license": "mit",
    },
    "jaco_play": {
        "tasks_col": "language_instruction",
        "license": "cc-by-4.0",
        "url": "https://github.com/clvrai/clvr_jaco_play_dataset",
        "citation_bibtex": dedent(r"""
            @software{dass2023jacoplay,
                author = {Dass, Shivin and Yapeter, Jullian and Zhang, Jesse and Zhang, Jiahui
                            and Pertsch, Karl and Nikolaidis, Stefanos and Lim, Joseph J.},
                title = {CLVR Jaco Play Dataset},
                url = {https://github.com/clvrai/clvr_jaco_play_dataset},
                version = {1.0.0},
                year = {2023}
            }""").lstrip(),
    },
    "kaist_nonprehensile": {
        "tasks_col": "language_instruction",
        "license": "cc-by-4.0",
        "url": "https://github.com/JaeHyung-Kim/rlds_dataset_builder",
        "citation_bibtex": dedent(r"""
            @article{kimpre,
                title={Pre-and post-contact policy decomposition for non-prehensile manipulation with zero-shot sim-to-real transfer},
                author={Kim, Minchan and Han, Junhyek and Kim, Jaehyung and Kim, Beomjoon},
                booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
                year={2023},
                organization={IEEE}
            }""").lstrip(),
    },
    "nyu_door_opening_surprising_effectiveness": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://jyopari.github.io/VINN/",
        "paper": "https://arxiv.org/abs/2112.01511",
        "citation_bibtex": dedent(r"""
            @misc{pari2021surprising,
                title={The Surprising Effectiveness of Representation Learning for Visual Imitation},
                author={Jyothish Pari and Nur Muhammad Shafiullah and Sridhar Pandian Arunachalam and Lerrel Pinto},
                year={2021},
                eprint={2112.01511},
                archivePrefix={arXiv},
                primaryClass={cs.RO}
            }""").lstrip(),
    },
    "nyu_franka_play_dataset": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://play-to-policy.github.io/",
        "paper": "https://arxiv.org/abs/2210.10047",
        "citation_bibtex": dedent(r"""
            @article{cui2022play,
                title   = {From Play to Policy: Conditional Behavior Generation from Uncurated Robot Data},
                author  = {Cui, Zichen Jeff and Wang, Yibin and Shafiullah, Nur Muhammad Mahi and Pinto, Lerrel},
                journal = {arXiv preprint arXiv:2210.10047},
                year    = {2022}
            }""").lstrip(),
    },
    "nyu_rot_dataset": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://rot-robot.github.io/",
        "paper": "https://arxiv.org/abs/2206.15469",
        "citation_bibtex": dedent(r"""
            @inproceedings{haldar2023watch,
                title={Watch and match: Supercharging imitation with regularized optimal transport},
                author={Haldar, Siddhant and Mathur, Vaibhav and Yarats, Denis and Pinto, Lerrel},
                booktitle={Conference on Robot Learning},
                pages={32--43},
                year={2023},
                organization={PMLR}
            }""").lstrip(),
    },
    "roboturk": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://roboturk.stanford.edu/dataset_real.html",
        "paper": "PAPER",
        "citation_bibtex": dedent(r"""
            @inproceedings{mandlekar2019scaling,
                title={Scaling robot supervision to hundreds of hours with roboturk: Robotic manipulation dataset through human reasoning and dexterity},
                author={Mandlekar, Ajay and Booher, Jonathan and Spero, Max and Tung, Albert and Gupta, Anchit and Zhu, Yuke and Garg, Animesh and Savarese, Silvio and Fei-Fei, Li},
                booktitle={2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
                pages={1048--1055},
                year={2019},
                organization={IEEE}
            }""").lstrip(),
    },
    "stanford_hydra_dataset": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://sites.google.com/view/hydra-il-2023",
        "paper": "https://arxiv.org/abs/2306.17237",
        "citation_bibtex": dedent(r"""
            @article{belkhale2023hydra,
                title={HYDRA: Hybrid Robot Actions for Imitation Learning},
                author={Belkhale, Suneel and Cui, Yuchen and Sadigh, Dorsa},
                journal={arxiv},
                year={2023}
            }""").lstrip(),
    },
    "stanford_kuka_multimodal_dataset": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://sites.google.com/view/visionandtouch",
        "paper": "https://arxiv.org/abs/1810.10191",
        "citation_bibtex": dedent(r"""
            @inproceedings{lee2019icra,
                title={Making sense of vision and touch: Self-supervised learning of multimodal representations for contact-rich tasks},
                author={Lee, Michelle A and Zhu, Yuke and Srinivasan, Krishnan and Shah, Parth and Savarese, Silvio and Fei-Fei, Li and  Garg, Animesh and Bohg, Jeannette},
                booktitle={2019 IEEE International Conference on Robotics and Automation (ICRA)},
                year={2019},
                url={https://arxiv.org/abs/1810.10191}
            }""").lstrip(),
    },
    "stanford_robocook": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://hshi74.github.io/robocook/",
        "paper": "https://arxiv.org/abs/2306.14447",
        "citation_bibtex": dedent(r"""
            @article{shi2023robocook,
                title={RoboCook: Long-Horizon Elasto-Plastic Object Manipulation with Diverse Tools},
                author={Shi, Haochen and Xu, Huazhe and Clarke, Samuel and Li, Yunzhu and Wu, Jiajun},
                journal={arXiv preprint arXiv:2306.14447},
                year={2023}
            }""").lstrip(),
    },
    "taco_play": {
        "tasks_col": "language_instruction",
        "license": "cc-by-4.0",
        "url": "https://www.kaggle.com/datasets/oiermees/taco-robot",
        "paper": "https://arxiv.org/abs/2209.08959, https://arxiv.org/abs/2210.01911",
        "citation_bibtex": dedent(r"""
            @inproceedings{rosete2022tacorl,
                author = {Erick Rosete-Beas and Oier Mees and Gabriel Kalweit and Joschka Boedecker and Wolfram Burgard},
                title = {Latent Plans for Task Agnostic Offline Reinforcement Learning},
                journal = {Proceedings of the 6th Conference on Robot Learning (CoRL)},
                year = {2022}
            }
            @inproceedings{mees23hulc2,
                title={Grounding  Language  with  Visual  Affordances  over  Unstructured  Data},
                author={Oier Mees and Jessica Borja-Diaz and Wolfram Burgard},
                booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
                year={2023},
                address = {London, UK}
            }""").lstrip(),
    },
    "tokyo_u_lsmo": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "URL",
        "paper": "https://arxiv.org/abs/2107.05842",
        "citation_bibtex": dedent(r"""
            @Article{Osa22,
                author  = {Takayuki Osa},
                journal = {The International Journal of Robotics Research},
                title   = {Motion Planning by Learning the Solution Manifold in Trajectory Optimization},
                year    = {2022},
                number  = {3},
                pages   = {291--311},
                volume  = {41},
            }""").lstrip(),
    },
    "toto": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://toto-benchmark.org/",
        "paper": "https://arxiv.org/abs/2306.00942",
        "citation_bibtex": dedent(r"""
            @inproceedings{zhou2023train,
                author={Zhou, Gaoyue and Dean, Victoria and Srirama, Mohan Kumar and Rajeswaran, Aravind and Pari, Jyothish and Hatch, Kyle and Jain, Aryan and Yu, Tianhe and Abbeel, Pieter and Pinto, Lerrel and Finn, Chelsea and Gupta, Abhinav},
                booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
                title={Train Offline, Test Online: A Real Robot Learning Benchmark},
                year={2023},
            }""").lstrip(),
    },
    "ucsd_kitchen_dataset": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "citation_bibtex": dedent(r"""
            @ARTICLE{ucsd_kitchens,
                author = {Ge Yan, Kris Wu, and Xiaolong Wang},
                title = {{ucsd kitchens Dataset}},
                year = {2023},
                month = {August}
            }""").lstrip(),
    },
    "ucsd_pick_and_place_dataset": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://owmcorl.github.io/#",
        "paper": "https://arxiv.org/abs/2310.16029",
        "citation_bibtex": dedent(r"""
            @preprint{Feng2023Finetuning,
                title={Finetuning Offline World Models in the Real World},
                author={Yunhai Feng, Nicklas Hansen, Ziyan Xiong, Chandramouli Rajagopalan, Xiaolong Wang},
                year={2023}
            }""").lstrip(),
    },
    "uiuc_d3field": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://robopil.github.io/d3fields/",
        "paper": "https://arxiv.org/abs/2309.16118",
        "citation_bibtex": dedent(r"""
            @article{wang2023d3field,
                title={D^3Field: Dynamic 3D Descriptor Fields for Generalizable Robotic Manipulation},
                author={Wang, Yixuan and Li, Zhuoran and Zhang, Mingtong and Driggs-Campbell, Katherine and Wu, Jiajun and Fei-Fei, Li and Li, Yunzhu},
                journal={arXiv preprint arXiv:},
                year={2023},
            }""").lstrip(),
    },
    "usc_cloth_sim": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://uscresl.github.io/dmfd/",
        "paper": "https://arxiv.org/abs/2207.10148",
        "citation_bibtex": dedent(r"""
            @article{salhotra2022dmfd,
                author={Salhotra, Gautam and Liu, I-Chun Arthur and Dominguez-Kuhne, Marcus and Sukhatme, Gaurav S.},
                journal={IEEE Robotics and Automation Letters},
                title={Learning Deformable Object Manipulation From Expert Demonstrations},
                year={2022},
                volume={7},
                number={4},
                pages={8775-8782},
                doi={10.1109/LRA.2022.3187843}
            }""").lstrip(),
    },
    "utaustin_mutex": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://ut-austin-rpl.github.io/MUTEX/",
        "paper": "https://arxiv.org/abs/2309.14320",
        "citation_bibtex": dedent(r"""
            @inproceedings{shah2023mutex,
                title={{MUTEX}: Learning Unified Policies from Multimodal Task Specifications},
                author={Rutav Shah and Roberto Mart{\'\i}n-Mart{\'\i}n and Yuke Zhu},
                booktitle={7th Annual Conference on Robot Learning},
                year={2023},
                url={https://openreview.net/forum?id=PwqiqaaEzJ}
            }""").lstrip(),
    },
    "utokyo_pr2_opening_fridge": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "citation_bibtex": dedent(r"""
            @misc{oh2023pr2utokyodatasets,
                author={Jihoon Oh and Naoaki Kanazawa and Kento Kawaharazuka},
                title={X-Embodiment U-Tokyo PR2 Datasets},
                year={2023},
                url={https://github.com/ojh6404/rlds_dataset_builder},
            }""").lstrip(),
    },
    "utokyo_pr2_tabletop_manipulation": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "citation_bibtex": dedent(r"""
            @misc{oh2023pr2utokyodatasets,
                author={Jihoon Oh and Naoaki Kanazawa and Kento Kawaharazuka},
                title={X-Embodiment U-Tokyo PR2 Datasets},
                year={2023},
                url={https://github.com/ojh6404/rlds_dataset_builder},
            }""").lstrip(),
    },
    "utokyo_saytap": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://saytap.github.io/",
        "paper": "https://arxiv.org/abs/2306.07580",
        "citation_bibtex": dedent(r"""
            @article{saytap2023,
                author = {Yujin Tang and Wenhao Yu and Jie Tan and Heiga Zen and Aleksandra Faust and
                Tatsuya Harada},
                title  = {SayTap: Language to Quadrupedal Locomotion},
                eprint = {arXiv:2306.07580},
                url    = {https://saytap.github.io},
                note   = {https://saytap.github.io},
                year   = {2023}
            }""").lstrip(),
    },
    "utokyo_xarm_bimanual": {
        "tasks_col": "language_instruction",
        "license": "cc-by-4.0",
        "citation_bibtex": dedent(r"""
            @misc{matsushima2023weblab,
                title={Weblab xArm Dataset},
                author={Tatsuya Matsushima and Hiroki Furuta and Yusuke Iwasawa and Yutaka Matsuo},
                year={2023},
            }""").lstrip(),
    },
    "utokyo_xarm_pick_and_place": {
        "tasks_col": "language_instruction",
        "license": "cc-by-4.0",
        "citation_bibtex": dedent(r"""
            @misc{matsushima2023weblab,
                title={Weblab xArm Dataset},
                author={Tatsuya Matsushima and Hiroki Furuta and Yusuke Iwasawa and Yutaka Matsuo},
                year={2023},
            }""").lstrip(),
    },
    "viola": {
        "tasks_col": "language_instruction",
        "license": "mit",
        "url": "https://ut-austin-rpl.github.io/VIOLA/",
        "paper": "https://arxiv.org/abs/2210.11339",
        "citation_bibtex": dedent(r"""
            @article{zhu2022viola,
                title={VIOLA: Imitation Learning for Vision-Based Manipulation with Object Proposal Priors},
                author={Zhu, Yifeng and Joshi, Abhishek and Stone, Peter and Zhu, Yuke},
                journal={6th Annual Conference on Robot Learning (CoRL)},
                year={2022}
            }""").lstrip(),
    },
}
# spellchecker:on


def batch_convert():
    status = {}
    logfile = LOCAL_DIR / "conversion_log.txt"
    assert set(DATASETS) == {id_.split("/")[1] for id_ in available_datasets}
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
