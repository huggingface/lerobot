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

import traceback
from pathlib import Path

from lerobot import available_datasets
from lerobot.common.datasets.v2.convert_dataset_v1_to_v2 import convert_dataset, parse_robot_config

LOCAL_DIR = Path("data/")
ALOHA_SINGLE_TASKS_REAL = {
    "aloha_mobile_cabinet": "Open the top cabinet, store the pot inside it then close the cabinet.",
    "aloha_mobile_chair": "Push the chairs in front of the desk to place them against it.",
    "aloha_mobile_elevator": "Take the elevator to the 1st floor.",
    "aloha_mobile_shrimp": "Sauté the raw shrimp on both sides, then serve it in the bowl.",
    "aloha_mobile_wash_pan": "Pick up the pan, rinse it in the sink and then place it in the drying rack.",
    "aloha_mobile_wipe_wine": "Pick up the wet cloth on the faucet and use it to clean the spilled wine on the table and underneath the glass.",
    "aloha_static_battery": "Place the battery into the slot of the remote controller.",
    "aloha_static_candy": "Pick up the candy and unwrap it.",
    "aloha_static_coffee": "Place the coffee capsule inside the capsule container, then place the cup onto the center of the cup tray, then push the 'Hot Water' and 'Travel Mug' buttons.",
    "aloha_static_coffee_new": "Place the coffee capsule inside the capsule container, then place the cup onto the center of the cup tray.",
    "aloha_static_cups_open": "Pick up the plastic cup and open its lid.",
    "aloha_static_fork_pick_up": "Pick up the fork and place it on the plate.",
    "aloha_static_pingpong_test": "Transfer one of the two balls in the right glass into the left glass, then transfer it back to the right glass.",
    "aloha_static_pro_pencil": "Pick up the pencil with the right arm, hand it over to the left arm then place it back onto the table.",
    "aloha_static_screw_driver": "Pick up the screwdriver with the right arm, hand it over to the left arm then place it into the cup.",
    "aloha_static_tape": "Cut a small piece of tape from the tape dispenser then place it on the cardboard box's edge.",
    "aloha_static_thread_velcro": "Pick up the velcro cable tie with the left arm, then insert the end of the velcro tie into the other end's loop with the right arm.",
    "aloha_static_towel": "Pick up a piece of paper towel and place it on the spilled liquid.",
    "aloha_static_vinh_cup": "Pick up the platic cup with the right arm, then pop its lid open with the left arm.",
    "aloha_static_vinh_cup_left": "Pick up the platic cup with the left arm, then pop its lid open with the right arm.",
    "aloha_static_ziploc_slide": "Slide open the ziploc bag.",
}
ALOHA_CONFIG = Path("lerobot/configs/robot/aloha.yaml")


def batch_convert():
    status = {}
    logfile = LOCAL_DIR / "conversion_log.txt"
    for num, repo_id in enumerate(available_datasets):
        print(f"\nConverting {repo_id} ({num}/{len(available_datasets)})")
        print("---------------------------------------------------------")
        name = repo_id.split("/")[1]
        single_task, tasks_col, robot_config = None, None, None

        if "aloha" in name:
            robot_config = parse_robot_config(ALOHA_CONFIG)
            if "sim_insertion" in name:
                single_task = "Insert the peg into the socket."
            elif "sim_transfer" in name:
                single_task = "Pick up the cube with the right arm and transfer it to the left arm."
            else:
                single_task = ALOHA_SINGLE_TASKS_REAL[name]
        elif "unitreeh1" in name:
            if "fold_clothes" in name:
                single_task = "Fold the sweatshirt."
            elif "rearrange_objects" in name or "rearrange_objects" in name:
                single_task = "Put the object into the bin."
            elif "two_robot_greeting" in name:
                single_task = "Greet the other robot with a high five."
            elif "warehouse" in name:
                single_task = (
                    "Grab the spray paint on the shelf and place it in the bin on top of the robot dog."
                )
        elif name != "columbia_cairlab_pusht_real" and "pusht" in name:
            single_task = "Push the T-shaped block onto the T-shaped target."
        elif "xarm_lift" in name or "xarm_push" in name:
            single_task = "Pick up the cube and lift it."
        elif name == "umi_cup_in_the_wild":
            single_task = "Put the cup on the plate."
        else:
            tasks_col = "language_instruction"

        try:
            convert_dataset(
                repo_id=repo_id,
                local_dir=LOCAL_DIR,
                single_task=single_task,
                tasks_col=tasks_col,
                robot_config=robot_config,
            )
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
