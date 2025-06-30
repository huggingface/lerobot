"""
Utils function by Mustafa to refactor
"""
import torch
import numpy as np
from lerobot.common.datasets.compute_stats import (
    aggregate_stats
)
from collections import defaultdict
OBS_IMAGE = "observation.image"
OBS_IMAGE_2 = "observation.image2"
OBS_IMAGE_3 = "observation.image3"

def reshape_features_to_max_dim(features: dict, reshape_dim: int = -1, keys_to_max_dim: dict = {}) -> dict:
    """Reshape features to have a maximum dimension of `max_dim`."""
    reshaped_features = {}
    for key in features:
        if key in keys_to_max_dim and keys_to_max_dim[key] is not None:
            reshaped_features[key] = features[key]
            shape = list(features[key]["shape"])
            if any([k in key for k in [OBS_IMAGE, OBS_IMAGE_2, OBS_IMAGE_3]]):  # Assume square images
                shape[-3] = keys_to_max_dim[key]
                shape[-2] = keys_to_max_dim[key]
            else:
                shape[reshape_dim] = keys_to_max_dim[key]
            reshaped_features[key]["shape"] = tuple(shape)
        else:
            reshaped_features[key] = features[key]
    return reshaped_features

def keep_datasets_with_valid_fps(
    ls_datasets: list, min_fps: int = 1, max_fps: int = 100
) -> list:
    print(f"Keeping datasets with fps between {min_fps} and {max_fps}. Considering {len(ls_datasets)} datasets.")
    for ds in ls_datasets:
        if ds.fps < min_fps or ds.fps > max_fps:
            print(f"Dataset {ds} has invalid fps: {ds.fps}. Removing it.")
            ls_datasets.remove(ds)
    print(f"Keeping {len(ls_datasets)} datasets with valid fps.")
    return ls_datasets

def keep_datasets_with_the_same_features_per_robot_type(
    ls_datasets: list
) -> list:
    """
    Filters datasets to only keep those with consistent feature shapes per robot type.

    Args:
        ls_datasets (List): List of datasets, each with a `meta.info['robot_type']`
            and `meta.episodes_stats` dictionary.

    Returns:
        List: Filtered list of datasets with consistent feature shapes.
    """
    robot_types = {ds.meta.info["robot_type"] for ds in ls_datasets}
    datasets_to_remove = set()

    for robot_type in robot_types:
        # Collect all stats dicts for this robot type
        stats_list = [
            ep_stats
            for ds in ls_datasets if ds.meta.info["robot_type"] == robot_type
            for ep_stats in ds.meta.episodes_stats.values()
        ]
        if not stats_list:
            continue

        # Determine the most common shape for each key
        all_keys = {key for stats in stats_list for key in stats}
        for ds in ls_datasets:
            if ds.meta.info["robot_type"] != robot_type:
                continue
            for key in all_keys:
                shape_counter = defaultdict(int)

                for stats in stats_list:
                    value = stats.get(key)
                    if value and "mean" in value and isinstance(value["mean"], (torch.Tensor, np.ndarray)): # FIXME(mshukor): check all stats; min, mean, max
                        shape_counter[value["mean"].shape] += 1
                if not shape_counter:
                    continue

                # Identify the most frequent shape
                main_shape = max(shape_counter, key=shape_counter.get)
                # Flag datasets that don't match the main shape
                # for ds in ls_datasets:
                first_ep_stats = next(iter(ds.meta.episodes_stats.values()), None)
                if not first_ep_stats:
                    continue
                value = first_ep_stats.get(key)
                if value and "mean" in value and isinstance(value["mean"], (torch.Tensor, np.ndarray)) and value["mean"].shape != main_shape:
                    datasets_to_remove.add(ds)
                    break

    # Filter out inconsistent datasets
    datasets_maks = [ds not in datasets_to_remove for ds in ls_datasets]
    filtered_datasets = [ds for ds in ls_datasets if ds not in datasets_to_remove]
    print(f"Keeping {len(filtered_datasets)} datasets. Removed {len(datasets_to_remove)} inconsistent ones. Inconsistent datasets:\n{datasets_to_remove}")
    return filtered_datasets, datasets_maks



def aggregate_stats_per_robot_type(ls_datasets) -> dict[str, dict[str, torch.Tensor]]:
    """Aggregate stats of multiple LeRobot datasets into multiple set of stats per robot type.

    The final stats will have the union of all data keys from each of the datasets.

    The final stats will have the union of all data keys from each of the datasets. For instance:
    - new_max = max(max_dataset_0, max_dataset_1, ...)
    - new_min = min(min_dataset_0, min_dataset_1, ...)
    - new_mean = (mean of all data)
    - new_std = (std of all data)
    """

    robot_types = {ds.meta.info["robot_type"] for ds in ls_datasets}
    stats = {robot_type: {} for robot_type in robot_types}
    for robot_type in robot_types:
        robot_type_datasets = []
        for ds in ls_datasets:
            if ds.meta.info["robot_type"] == robot_type:
                robot_type_datasets.extend(list(ds.meta.episodes_stats.values()))
        # robot_type_datasets = [list(ds.episodes_stats.values()) for ds in ls_datasets if ds.meta.info["robot_type"] == robot_type]
        stat = aggregate_stats(robot_type_datasets)
        stats[robot_type] = stat
    return stats

def str_to_torch_dtype(dtype_str):
    """Convert a dtype string to a torch dtype."""
    mapping = {
        "float32": torch.float32,
        "int64": torch.int64,
        "int16": torch.int16,
        "bool": torch.bool,
        "video": torch.float32,  # Assuming video is stored as uint8 images
    }
    return mapping.get(dtype_str, torch.float32)  # Default to float32

def create_padded_features(item: dict, features: dict = {}):
    for key, ft in features.items():
        if any([k in key for k in ["cam", "effort", "absolute"]]): # FIXME(mshukor): temporary hack
            continue
        shape = ft["shape"]
        if len(shape) == 3:  # images to torch format (C, H, W)
            shape = (shape[2], shape[0], shape[1])
        if len(shape) == 1 and shape[0] == 1:  # ft with shape are actually tensor(ele)
            shape = []
        if key not in item:
            dtype = str_to_torch_dtype(ft["dtype"])
            item[key] = torch.zeros(shape, dtype=dtype)
            item[f"{key}_padding_mask"] = torch.tensor(0, dtype=torch.int64)
            if "image" in key: # FIXME(mshukor): support other observations
                item[f"{key}_is_pad"] = torch.BoolTensor([False])
        else:
            item[f"{key}_padding_mask"] = torch.tensor(1, dtype=torch.int64)
    return item

ROBOT_TYPE_KEYS_MAPPING = {
    "lerobot/stanford_hydra_dataset": "static_single_arm",
    "lerobot/iamlab_cmu_pickup_insert": "static_single_arm",
    "lerobot/berkeley_fanuc_manipulation": "static_single_arm",
    "lerobot/toto": "static_single_arm",
    "lerobot/roboturk": "static_single_arm",
    "lerobot/jaco_play": "static_single_arm",
    "lerobot/taco_play": "static_single_arm_7statedim",
}

def pad_tensor(
    tensor: torch.Tensor, max_size: int, pad_dim: int = -1, pad_value: float = 0.0
) -> torch.Tensor:
    is_numpy = isinstance(tensor, np.ndarray)
    if is_numpy:
        tensor = torch.tensor(tensor)
    pad = max_size - tensor.shape[pad_dim]
    if pad > 0:
        pad_sizes = (0, pad)  # pad right
        tensor = torch.nn.functional.pad(tensor, pad_sizes, value=pad_value)
    return tensor.numpy() if is_numpy else tensor

def map_dict_keys(item: dict, feature_keys_mapping: dict, training_features: list = None, pad_key: str = "is_pad") -> dict:
    """Maps feature keys from the dataset to the keys used in the model."""
    if feature_keys_mapping is None:
        return item
    features = {}
    for key in item:
        if key in feature_keys_mapping:
            if feature_keys_mapping[key] is not None:
                if training_features is None or feature_keys_mapping[key] in training_features:
                    features[feature_keys_mapping[key]] = item[key]
        else:
            if training_features is None or key in training_features or pad_key in key:
                features[key] = item[key]
    return features


TASKS_KEYS_MAPPING = {
    "pranavsaroha/so100_legos4": {0: "Pick up the LEGO block and place it in the bowl of the same color as the LEGO block."},
    "pranavsaroha/so100_onelego2": {0: "Pick up the green LEGO block and place it in the green bowl."},
    "jpata/so100_pick_place_tangerine": {0: "Pick up the tangerine and place it."},
    "pranavsaroha/so100_onelego3": {0: "Pick up the green LEGO block and place it in the green bowl."},
    "pranavsaroha/so100_carrot_2": {0: "Pick up a carrot and put it in the bin."},
    "pranavsaroha/so100_carrot_5": {0: "Pick up a carrot and put it in the bin."},
    "pandaRQ/pick_med_1": {0: "Pick up the object and place it in the box."},
    "HITHY/so100_strawberry": {0: "Grasp a strawberry and put it in the bin."},
    "vladfatu/so100_above": {0: "Pick up red object and place it in the box."},
    "koenvanwijk/orange50-1": {0: "Pick up the orange object and but it in the LEGO box. "},
    "koenvanwijk/orange50-variation-2": {0: "Pick up the orange object and but it in the LEGO box. "},
    "FeiYjf/new_GtoR": {0: "Move along the line on the paper from start to end."},
    "CSCSXX/pick_place_cube_1.18": {0: "Pick up the cube and place it in the box."},
    "vladfatu/so100_office": {0: "Pick up the red object and place it in the box."},
    "dragon-95/so100_sorting": {0: "Pick up the object from box A and place it in box B."},
    "dragon-95/so100_sorting_1": {0: "Pick up the object from box A and place it in box B."},
    "nbaron99/so100_pick_and_place4": {0: "Pick up the triangular object and place it on a green sticker."},
    "Beegbrain/pick_place_green_block": {0: "Pick up the green block and place in the red cup."},
    "Ityl/so100_recording2": {0: "Pick up the red cube and place it on top of the blue cube."},
    "dragon-95/so100_sorting_2": {0: "Pick up the object from box A and place it in box B."},
    "dragon-95/so100_sorting_3": {0: "Pick up the object from box A and place it in box B."},
    "aractingi/push_cube_offline_data": {0: "Push the green cube to the yellow sticker."},
    "HITHY/so100_peach3": {0: "Grasp a peach and put it in the bin."},
    "HITHY/so100_peach4": {0: "Grasp a peach and put it on the plate."},
    "shreyasgite/so100_legocube_50": {0: "Grasp a lego block and put it in the bin."},
    "shreyasgite/so100_base_env": {0: "Grasp a lego block and put it in the bin."},
    "triton7777/so100_dataset_mix": {
        0: "Pick up the black tape and place it inside the white tape roll.",
        1: "Pick up the gift miniatures and place them in the black box.",
        2: "Sort the mixed objects into their appropriate categories.",
        3: "Place the pens into the pen holder.",
        4: "Place pens, bottles, and any suitable items into the pen holder, as appropriate.",
        5: "Place the oranges into the yellow basket.",
        6: "Stack the plates and place the cup on top."
    },
    "Deason11/Open_the_drawer_to_place_items": {0: "Put the objects in the open drawer."},
    "Deason11/PLACE_TAPE_PUSH_DRAWER": {0: "Place the tape in the drawer and close it. "},
    "NONHUMAN-RESEARCH/SOARM100_TASK_VENDA": {0: "Pick up the object and place it in the box."},
    "mikechambers/block_cup_14": {0: "Grasp a block and put it in a cup."},
    "samsam0510/tooth_extraction_3": {0: "Extract the tooth and put it somewhere."},
    "samsam0510/tooth_extraction_4": {0: "Extarct the molar and put it somewhere."},
    "samsam0510/cube_reorientation_2": {0: "Rotate the object so it aligns with the silhouette on the table."},
    "samsam0510/cube_reorientation_4": {0: "Rotate the object so it aligns with respect to the line on the table."},
    "samsam0510/glove_reorientation_1": {0: "Rotate the glove so the bottom part aligns with the line on the table."},
    "DorayakiLin/so100_pick_charger_on_tissue": {0: "Pick up the charger and put it on the white tissue."},
    "zijian2022/noticehuman3": {0: "Notice human."},
    "liuhuanjim013/so100_th": {0: "Grasp a lego figure and put it in the box."},
    "Bartm3/tape_to_bin": {0: "Grasp a tape and put it in the bin."},

    # Community dataset v2
    "Chojins/chess_game_009_white": {
        0: "Move the blue chess pieces to the highlighted squares."
    },
    "1g0rrr/sam_openpi03": {
        0: "Pick up the cube and place it in the box."
    },
    "sihyun77/suho_3_17_1": {
        0: "Grasp a lego block and put it in the bin."
    },
    "sihyun77/sihyun_3_17_2": {
        0: "Grasp a lego block and put it in the bin."
    },
    "sihyun77/suho_3_17_3": {
        0: "Grasp a lego block and put it in the bin."
    },
    "sihyun77/sihyun_3_17_5": {
        0: "Grasp a lego block and put it in the bin."
    },
    "Odog16/so100_cube_drop_pick_v1": {
        0: "Pick up the orange cube, release it, and then pick it up again."
    },
    "sihyun77/sihyun_main_2": {
        0: "Grasp a lego block and put it in the bin."
    },
    "sihyun77/suho_main_2": {
        0: "Grasp a lego block and put it in the bin."
    },
    "Bartm3/dice2": {
        0: "Grasp a dice and put it in the bin."
    },
    "sihyun77/sihyun_main_3": {
        0: "Grasp a lego block and put it in the bin."
    },
    "Loki0929/so100_duck": {
        0: "Grasp red, green, yellow ducks and put them in the box."
    },
    "pietroom/holdthis": {
        0: "Hold the object steadily without releasing it."
    },
    "pietroom/actualeasytask": {
        0: "Grasp the marker and put it in the plastic box."
    },
    "Beegbrain/pick_lemon_and_drop_in_bowl": {
        0: "Pick the yellow lemon and drop it in the red bowl."
    },
    "Beegbrain/sweep_tissue_cube": {
        0: "Sweep the red cubes to the right with the tissue."
    },
    "zijian2022/321": {
        0: "Grasp a lego block and put it in the bin."
    },
    "1g0rrr/sam_openpi_solder1": {
        0: "Bring contact to the pad on the board."
    },
    "1g0rrr/sam_openpi_solder2": {
        0: "Bring contact to the pad on the board."
    },
    "gxy1111/so100_pick_place": {
        0: "Grasp a toy panda and put it in the cup."
    },
    "Odog16/so100_cube_stacking_v1": {
        0: "Stack the cubes in the following order from bottom to top: black, blue, then orange."
    },
    "sihyun77/mond_1": {
        0: "Grasp a lego block and put it in the bin."
    },
    "andlyu/so100_indoor_1": {
        0: "Locate and grasp the blueberry."
    },
    "andlyu/so100_indoor_3": {
        0: "Locate and grasp the blueberry."
    },
    "frk2/so100large": {
        0: "Pick up roll of tape and put it in the bin."
    },
    "lirislab/sweep_tissue_cube": {
        0: "Sweep the red cubes to the right with the tissue bag."
    },
    "lirislab/lemon_into_bowl": {
        0: "Pick the yellow lemon and drop it in the red bowl"
    },
    "lirislab/red_cube_into_green_lego_block": {
        0: "Put the red cube on top of the yellow cube."
    },
    "lirislab/red_cube_into_blue_cube": {
        0: "Put the red cube on top of the blue cube."
    },
    "00ri/so100_battery": {
        0: "Grasp a battery and put it in the bin."
    },
    "frk2/so100largediffcam": {
        0: "Pick up roll of tape and put it in the bin"
    },
    "FsqZ/so100_1": {
        0: "Put the yellow cube inside the purple box."
    },
    "ZGGZZG/so100_drop0": {
        0: "Grasp a ball and put it in the hole."
    },
    "Chojins/chess_game_000_white_red": {
        0: "Move the red chess pieces to the highlighted squares."
    },
    "smanni/train_so100_fluffy_box": {
        0: "Grasp a small object and place it in the box."
    },
    "ganker5/so100_push_20250328": {
        0: "Grasp a lego block and put it in the bin."
    },
    "ganker5/so100_dataline_0328": {
        0: "Grasp a lego block and put it in the bin."
    },
    "ganker5/so100_color_0328": {
        0: "Grasp a lego block and put it in the bin."
    },
    "CrazyYhang/A1234-B-C_mvA2B": {
        0: "Move the top disk from the left column to the middle column."
    },
    "RasmusP/so100_Orange2Green": {
        0: "Grasp the orange block and drop it in the box."
    },
    "sixpigs1/so100_pick_cube_in_box": {
        0: "Pick up the red cube and put it in the box."
    },
    "ganker5/so100_push_20250331": {
        0: "Grasp a lego block and put it in the bin."
    },
    "ganker5/so100_dataline_20250331": {
        0: "Grasp a lego block and put it in the bin."
    },
    "lirislab/put_caps_into_teabox": {
        0: "Pick the coffee capsule and put it into the top drawer of the teabox"
    },
    "lirislab/close_top_drawer_teabox": {
        0: "Close the top drawer of the teabox"
    },
    "lirislab/open_top_drawer_teabox": {
        0: "Open the top drawer of the teabox"
    },
    "lirislab/unfold_bottom_right": {
        0: "Unfold the bag from bottom right corner"
    },
    "lirislab/push_cup_target": {
        0: "Push the red cup to the pink target"
    },
    "lirislab/put_banana_bowl": {
        0: "Put the banana into the red bowl"
    },
    "Chojins/chess_game_001_blue_stereo": {
        0: "Move the blue chess pieces to the highlighted squares"
    },
    "Chojins/chess_game_001_red_stereo": {
        0: "Move the red chess pieces to the highlighted squares"
    },
    "ganker5/so100_toy_20250402": {
        0: "Grasp a lego block and put it in the bin."
    },
    "Gano007/so100_medic": {
        0: "Grasp a medic box and put it in the bin."
    },
    "00ri/so100_battery_bin_center": {
        0: "Grasp a battery and put it in the bin."
    },
    "paszea/so100_whale_2": {
        0: "Grasp a whale and put it in the plate."
    },
    "lirislab/fold_bottom_right": {
        0: "Fold the bag from the bottom right corner."
    },
    "lirislab/put_coffee_cap_teabox": {
        0: "Put the coffee capsule into the top drawer of the teabox."
    },
    "therarelab/so100_pick_place_2": {
        0: "Pick a plaster roll and place it to the blue sticker."
    },
    "paszea/so100_whale_3": {
        0: "Grasp a whale and put it in the plate."
    },
    "paszea/so100_whale_4": {
        0: "Grasp a whale and put it in the plate."
    },
    "paszea/so100_lego": {
        0: "Grasp a lego and put it in the basket."
    },
    "LemonadeDai/so100_coca": {
        0: "Grasp the Coca-Cola can and orient it upright with the top facing up."
    },
    "zijian2022/backgrounda": {
        0: "Grasp a lego block and put it in the bin."
    },
    "zijian2022/backgroundb": {
        0: "Grasp a lego block and put it in the bin."
    },
    "356c/so100_nut_sort_1": {
        0: "Pick up the steel nuts and sort them by color."
    },
    "Mwuqiu/so100_0408_muti": {
        0: "Grasp a yellow duck and put it in the box."
    },
    "aimihat/so100_tape": {
        0: "Pick up the tape and put it in the bowl."
    },
    "lirislab/so100_demo": {
        0: "Put the banana into the red bowl."
    },
    "356c/so100_duck_reposition_1": {
        0: "Grasp the tool and use it to move the duck to the indicated position."
    },
    "zijian2022/sort1": {
        0: "Grasp a box and sort it by color: place grey boxes on the left and black boxes on the right."
    },
    "weiye11/so100_410_zwy": {
        0: "Pick up the cube and place it on the black circle."
    },
    "VoicAndrei/so100_banana_to_plate_only": {
        0: "Pick up the banana and place it on the plate."
    },
    "sixpigs1/so100_stack_cube_error": {
        0: "Pick up the red cube and stack it on the green cube with position offset when grasping.",
        1: "Pick up the red cube and stack it on the green cube with gripper error when grasping.",
        2: "Pick up the red cube and stack it on the green cube with position offset when stacking.",
        3: "Pick up the red cube and stack it on the green cube without errors",
    },
    "isadev/bougies3": {
        0: "Grab the candle wick by the aluminium plate and place it in the box."
    },
    "zijian2022/close3": {
        0: "Grasp a lego block and put it in the bin."
    },
    "bensprenger/left_arm_yellow_brick_in_box_v0": {
        0: "Grasp the yellow lego block and put it in the box."
    },
    "bensprenger/left_arm_yellow_brick_in_box_with_purple_noise_v0": {
        0: "Grasp a yellow lego block and put it in the bin."
    },
    "roboticshack/team16-can-stacking": {
        0: "Grasp the flipped cup and stack it on top of the midpoint between the two other cups to create a tower"
    },
    "bensprenger/right_arm_p_brick_in_box_with_y_noise_v0": {
        0: "Grasp the purple lego block and put it in the box."
    },
    "pierfabre/pig2": {
        0: "Pick the pig and place it to the right."
    },
    "zijian2022/insert2": {
        0: "Grasp a lego block and put it in the bin."
    },
    "roboticshack/team-7-right-arm-grasp-tape": {
        0: "Grasp the tape and put it in the box."
    },
    "pierfabre/pig3": {
        0: "Pick the pig and place it to the right."
    },
    "Jiangeng/so100_413": {
        0: "Pick up the cube and place it on top of the black circle."
    },
    "roboticshack/team9-pick_cube_place_static_plate": {
        0: "Pick up the green cube and place on orange plate."
    },
    "AndrejOrsula/lerobot_double_ball_stacking_random": {
        0: "Stack the balls on top of each other."
    },
    "roboticshack/left-arm-grasp-lego-brick": {
        0: "Grasp the lego brick and put it in the box."
    },
    "roboticshack/team-7-left-arm-grasp-motor": {
        0: "Grasp the black motor and put it in the box."
    },
    "pierfabre/cow2": {
        0: "Pick the cow and place it to the right."
    },
    "pierfabre/sheep": {
        0: "Pick the sheep and place it to the right."
    },
    "roboticshack/team9-pick_chicken_place_plate": {
        0: "Pick up the chicken and place on orange plate"
    },
    "roboticshack/team13-two-balls-stacking": {
        0: "Stack the balls on top of each other."
    },
    "tkc79/so100_lego_box_1": {
        0: "Grasp a lego block and put it in the box."
    },
    "pierfabre/rabbit": {
        0: "Pick the rabbit and place it to the right.",
        1: "Pick the rabbit and put it to the right"
    },
    "roboticshack/team13-three-balls-stacking": {
        0: "Stack the balls on top of each other."
    },
    "pierfabre/horse": {
        0: "Pick the horse and place it to the right."
    },
    "pierfabre/chicken": {
        0: "Pick the chicken and place it to the right."
    },
    "roboticshack/team16-water-pouring": {
        0: "Pouring water from one cup to another cup"
    },
    "ad330/cubePlace": {
        0: "Grasp white cube and place it in the bowl."
    },
    "paszea/so100_lego_2cam": {
        0: "Grap lego blocks and put them in the plate."
    },
    "bensprenger/chess_game_001_blue_stereo": {
        0: "Move the blue chess pieces to the highlighted squares."
    },
    "Mohamedal/put_banana": {
        0: "Put the banana in the red bowl."
    },
    "tkc79/so100_lego_box_2": {
        0: "Grasp a lego block and put it in the box."
    },
    "samanthalhy/so100_herding_1": {
        0: "Grasp a green tool and herd all the particles to the grey bin."
    },
    "jlesein/TestBoulon7": {
        0: "Pick up the bolt and put it on the plate."
    },
    # V3 with VLM (Qwen-VL-2.5-instruct) annotation
    "satvikahuja/mixer_on_off_new_1": {0: "Press the button on the blender."}, 
    "andy309/so100_0314_fold_cloths": {0: "fold the cloths, use two cameras, two arms."}, 
    "jchun/so100_pickplace_small_20250323_120056": {0: "Grasp items from white bowl and place in black tray"}, 
    "Ofiroz91/so_100_cube2bowl": {0: "placing cube inside a red bawl"}, 
    "ZCM5115/so100_1210": {0: "picks up the USB cable."}, 
    "francescocrivelli/carrot_eating": {0: "pick up carrot and bring to mouth"}, 
    "ZCM5115/so100_2Arm3cameras_movebox": {0: "Pick up the white box from the table."}, 
    "pranavsaroha/so100_carrot_1": {0: "pick a carrot and put it in the bin"}, 
    "pranavsaroha/so100_carrot_3": {0: "pick a carrot and put it in the bin"}, 
    "maximilienroberti/so100_lego_red_box": {0: "Placing the red Lego in the red box bin."}, "pranavsaroha/so100_squishy": {0: "pick a squishy and put it in the bin"}, "rabhishek100/so100_train_dataset": {0: "picks tape and places it in a cup."}, "pranavsaroha/so100_squishy100": {0: "pick a squishy and put it in the bin"}, "pandaRQ/pickmed": {0: "Place the green block on the table."}, "swarajgosavi/act_kikobot_pusht_real": {0: "picks up the red block."}, "pranavsaroha/so100_squishy2colors_1": {0: "pick the squishies and put them in the bins with their corresponding colors"}, "Chojins/chess_game_001_white": {0: "Move the blue chess pieces as indicated by the highlighted squares"}, "jmrog/so100_sweet_pick": {0: "Pick up the candy and place it in the bowl."}, "Chojins/chess_game_002_white": {0: "Move the blue chess pieces as indicated by the highlighted squares"}, "pranavsaroha/so100_squishy2colors_2_new": {0: "pick the squishies and put them in the bins with their corresponding colors"}, "Chojins/chess_game_003_white": {0: "Move the blue chess pieces as indicated by the highlighted squares"}, "Chojins/chess_game_004_white": {0: "Move the blue chess pieces as indicated by the highlighted squares"}, "Chojins/chess_game_005_white": {0: "Move the blue chess pieces as indicated by the highlighted squares"}, "Chojins/chess_game_006_white": {0: "Move the blue chess pieces as indicated by the highlighted squares"}, "Chojins/chess_game_007_white": {0: "Move the blue chess pieces as indicated by the highlighted squares"}, "koenvanwijk/blue52": {0: "places blue block on red LEGO piece."}, "jlitch/so100multicam7": {0: "pick up brick and put in bin"}, "vladfatu/so100_ds": {0: "Pick up the cube and place it in the box."}, "Chojins/chess_game_000_white": {0: "Move the blue chess pieces as indicated by the highlighted squares"}, "satvikahuja/orange_mixer_1": {0: "pick orange and place in mixer"}, "satvikahuja/mixer_on_off": {0: "switch the mixer on or off"}, "satvikahuja/orange_pick_place_new1": {0: "Pick up the orange and place it in the bowl."}, "satvikahuja/mixer_on_off_new": {0: "Adjust the s position."}, "FeiYjf/Makalu_push": {0: "Pick up the blue cube."}, "chmadran/so100_dataset04": {0: "picks the blue block and places it in the red cup."}, "FeiYjf/Maklu_dataset": {0: "Pick up the blue cube and place it on the paper."}, "FeiYjf/new_Dataset": {0: "Pick up the blue cube."}, "satvikahuja/mixer_on_off_new_4": {0: "Place the lid on the blender."}, "CSCSXX/pick_place_cube_1.17": {0: "Pick up the red block and place it in the box."}, "liyitenga/so100_pick_taffy3": {0: "Place the eraser in the container."}, "liyitenga/so100_pick_taffy6": {0: "Pick up the toy and place it in the purple cup."}, "yuz1wan/so100_pickplace": {0: "Pick the pink block and place it in the paper cup."}, "liyitenga/so100_pick_taffy7": {0: "Pick up the toy and place it in the box."}, "swarajgosavi/act_kikobot_block_real": {0: "Pick up the blue cube and place it in the box."}, "SeanLMH/so100_picknplace_v2": {0: "picks up blue cube and places it in yellow box."}, "DimiSch/so100_50ep_2": {0: "Place the yellow object in the bowl."}, "DimiSch/so100_50ep_3": {0: "Pick the yellow button from the table."}, "SeanLMH/so100_picknplace": {0: "Pick up the blue block and place it in the yellow box."}, "nbaron99/so100_pick_and_place2": {0: "picks up the white object."}, "chmadran/so100_dataset08": {0: "places blue block on paper."}, "Ityl/so100_recording1": {0: "Putting the red square onto the yellow piece"}, "ad330/so100_box_pickPlace": {0: "places jar in box."}, "carpit680/giraffe_task": {0: "Grasp a block and put it in the bin."}, "carpit680/giraffe_sock_demo_1": {0: "Grasp a sock off the floor."}, "DimiSch/so100_terra_50_2": {0: "Grasp a lego block and put it in the bin."}, "aractingi/push_cube_offline_data_cropped_resized": {0: "Push the green cube to the yellow sticker"}, "FeiYjf/Test_NNNN": {0: "Pick up the purple cube and move it to the right."}, "HITHY/so100_peach": {0: "Grasp a peach and put it in the bin."}, "zaringleb/so100_cube_4_binary": {0: "Grasp a lego block and put it in the bin."}, "FeiYjf/Grab_Pieces": {0: "places the black object on the table."}, "hegdearyandev/so100_eraser_cup_v1": {0: "picks up the red object."}, "jbraumann/so100_1902": {0: "picks up the yellow ball."}, "zaringleb/so100_cube_5_linear": {0: "Grasp a lego block and put it in the bin."}, "samsam0510/tape_insert_1": {0: "Grasp a red tape and put it on the box."}, "samsam0510/tape_insert_2": {0: "Grasp a red tape and put it in the yellow tape."}, "pengjunkun/so100_push_to_hole": {0: "Push the T into the hole."}, "Deason11/Random_Kitchen": {0: "Pick up the cup and place it on the table."}, "Loki0929/so100_100": {0: "Grasp a rubber duck and put it in the box."}, "speedyyoshi/so100_grasp_pink_block": {0: "Grasp a lego block and put it in the bin."}, "lirislab/green_lego_block_into_mug": {0: "pick the green block and place it in the red cup"}, "kevin510/lerobot-cat-toy-placement": {0: "Grasp the cat toy and put it in the cup."}, "NONHUMAN-RESEARCH/SOARM100_TASK_VENDA_BOX": {0: "Move the cube to the right side of the table."}, "zijian2022/noticehuman5": {0: "picks up the box."}, "zijian2022/noticehuman70": {0: "Stop movement when human encounter testbed.", 1: "Stop movement when human encounter testbed w/ trigger."}, "Bartm3/tape_to_bin": {0: "Grasp a tape and put it in the bin."}, "Pi-robot/barbecue_flip": {0: "Pick up the orange cone and place it on the table."}, "Pi-robot/barbecue_put": {0: "Pick up the stick and place it in the grill."}, "sshh11/so100_orange_50ep_1": {0: "Grasp an orange object and put it in the bin."}, "sshh11/so100_orange_50ep_2": {0: "Grasp an orange object and put it in the bin."}, "DorayakiLin/so100_pick_cube_in_box": {0: "Pick up the red cube and put it in the box."}, "Bartm3/tape_to_bin2": {0: "Grasp a tape and put it in the bin."}, "andy309/so100_0311_1152": {0: "Grasp and put it in the bin."}, "sihyun77/suho_so100": {0: "Grasp a lego block and put it in the bin."}, "sihyun77/si_so100": {0: "Grasp a lego block and put it in the bin."}, "shreyasgite/so100_base_left": {0: "Grasp a lego block and put it in the bin."}, "sihyun77/suho_red": {0: "Grasp a lego block and put it in the bin."}, "liuhuanjim013/so100_block": {0: "Grasp a lego block and put it in the bin."}, "joaoocruz00/so100_makeitD1": {0: "Grasp a lego block and put it in the bin."}, "sihyun77/suho_angel": {0: "Grasp a lego block and put it in the bin."}, "sihyun77/sihyun_king": {0: "Grasp a lego block and put it in the bin."}, "acrampette/third_arm_01": {0: "Pick up the circuit board from the table."}, "Winster/so100_cube": {0: "Grasp a lego block and put it in the bin."}, "1g0rrr/sam_openpi03": {0: "Grasp a blue cube and put it in the gray box."}, "thedevansh/mar16_1336": {0: "Grasp a lego block and put it in the bin."}, "hkphoooey/throw_stuffie": {0: "Grab stuffed animal and throw it on the dot."}, "acrampette/third_arm_02": {0: "Pick up the tie and place it in the box."}, "kumarhans/so100_tape_task": {0: "Grasp a roll of tape and put it over the candle case."}, "Odog16/so100_tea_towel_folding_v1": {0: "Fold tea towel into quarters"}, "pietroom/first_task_short": {0: "Pick up the marker from the box."}, "zijian2022/c0": {0: "Grasp a lego block and put it in the bin based on color.", 1: "Grasp a lego block and put it in the bin."}, "1g0rrr/sam_openpi_solder1": {0: "bring contact to the pad on the board."}, "1g0rrr/sam_openpi_solder2": {0: "bring contact to the pad on the board."}, "bnarin/so100_tic_tac_toe_we_do_it_live": {0: "move tic tac toe as player 2."}, "chmadran/so100_home_dataset": {0: "Grasp a lego block and put it in the bin."}, "baladhurgesh97/so100_final_picking_3": {0: "Grasp a carrot, plastic bottle and put it in respective bins."}, "zaringleb/so100_cube_6_2d": {0: "Grasp a lego block and put it in the bin."}, "ZGGZZG/so100_drop1": {0: "Grasp a cube and put it in the right place."}, "abhisb/so100_51_ep": {0: "Pick up the cube and place it in the box."}, "allenchienxxx/so100Test": {0: "Grasp a lego block and put it in the bin."}, "lizi178119985/so100_jia": {0: "Grasp a lego block and put it in the bin."}, "andrewcole712/so100_tape_bin_place": {0: "Place the tape in the wooden box."}, "Gano007/so100_doliprane": {0: "Grasp a medic box and put it in the bin."}, "XXRRSSRR/so100_v3_num_episodes_50": {0: "Grasp a box and put it in the side."}, "Gano007/so100_gano": {0: "Grasp a box and put it in the bin."}, "paszea/so100_whale_grab": {0: "Grasp a whale and put it in the plate."}, "Clementppr/lerobot_pick_and_place_dataset_world_model": {0: "Grasp a fruit and put it in the cup."}, "RasmusP/so100_dataset50ep": {0: "Grasp a square block and put it in the box."}, "Gano007/so100_second": {0: "Grasp a yellow box and put it in the bin."}, "zaringleb/so100_cude_linear_and_2d_comb": {0: "Grasp a lego block and put it in the bin."}, "zijian2022/digitalfix3": {0: "Grasp a lego block and put it in the bin."}, "sihyun77/mond_13": {0: "Grasp a lego block and put it in the bin."}, "356c/so100_rope_reposition_1": {0: "Grasp rope and reposition."}, "paszea/so100_lego_mix": {0: "Grasp lego blocks and put them in the plate."}, "jiajun001/eraser00_2": {0: "picks tissue paper from box."}, "VoicAndrei/so100_banana_to_plate_rebel_full": {0: "Pick up the banana and place it on the place"}, "isadev/bougies1": {0: "Put the candles in the box."}, "sixpigs1/so100_pick_cube_in_box_error": {0: "Pick up the red cube and put it in the box with position offset when grasping.", 1: "Pick up the red cube and put it in the box with gripper error when grasping.", 2: "Pick up the red cube and put it in the box with position offset when releasing.", 3: "Pick up the red cube and put it in the box without errors."}, "sixpigs1/so100_push_cube_error": {0: "Push the blue cube to the red and white target with position offset when reaching.", 1: "Push the blue cube to the red and white target with position offset when pushing.", 2: "Push the blue cube to the red and white target with gripper error when pushing.", 3: "Push the blue cube to the red and white target without errors."}, "sixpigs1/so100_pull_cube_error": {0: "Pull the yellow cube to the red and white target with position offset when reaching.", 1: "Pull the yellow cube to the red and white target with position offset when pulling.", 2: "Pull the yellow cube to the red and white target with gripper error when pulling.", 3: "Pull the yellow cube to the red and white target without errors."}, "isadev/bougies2": {0: "grab the candle wick and place it in the tray."}, "therarelab/med_dis_rare_6": {0: "places green object in box."}, "sixpigs1/so100_pull_cube_by_tool_error": {0: "Pick up the L-shaped tool and pull the purple cube by the tool with position offset when grasping.", 1: "Pick up the L-shaped tool and pull the purple cube by the tool with rotation offset when grasping.", 2: "Pick up the L-shaped tool and pull the purple cube by the tool with gripper error when grasping.", 3: "Pick up the L-shaped tool and pull the purple cube by the tool with position offset when lowering.", 4: "Pick up the L-shaped tool and pull the purple cube by the tool with rotation offset when lowering.", 5: "Pick up the L-shaped tool and pull the purple cube by the tool with position offset when pulling.", 6: "Pick up the L-shaped tool and pull the purple cube by the tool with gripper error when pulling.", 7: "Pick up the L-shaped tool and pull the purple cube by the tool without errors."}, "sixpigs1/so100_insert_cylinder_error": {0: "Pick up the cylinder, upright it, and insert it into the middle hole of the shelf with position offset when grasping.", 1: "Pick up the cylinder, upright it, and insert it into the middle hole of the shelf with gripper error when grasping.", 2: "Pick up the cylinder, upright it, and insert it into the middle hole of the shelf with rotation offset when uprighting.", 3: "Pick up the cylinder, upright it, and insert it into the middle hole of the shelf with position offset when inserting.", 4: "Pick up the cylinder, upright it, and insert it into the middle hole of the shelf with choice error when inserting.", 5: "Pick up the cylinder, upright it, and insert it into the middle hole of the shelf without errors."}, "lirislab/guess_who_no_cond": {0: "Place the card in the slot."}, "lirislab/guess_who_lighting": {0: "Pick up the card from the shelf."}, "nguyen-v/so100_press_red_button": {0: "The  places the cube in the box."}, "nguyen-v/so100_bimanual_grab_lemon_put_in_box2": {0: "Grab the lemon with the black arm, then give it to the green arm, then place the lemon in the cardboard box with the green arm."}, "nguyen-v/press_red_button_new": {0: "Press the red button with the black arm"}, "nguyen-v/so100_rotate_red_button": {0: "Rotate the red button clockwise  with the black arm"}, "roboticshack/team10-red-block": {0: "Pick a red lego block and move it to the right."}, "Cidoyi/so100_all_notes_1": {0: "Connect the cable to the device."}, "roboticshack/team11_pianobot": {0: "Point at the keyboard."}, "roboticshack/team2-guess_who_so100": {0: "Pick up the card from the shelf."}, "roboticshack/team2-guess_who_so100_light": {0: "Place the card in the slot."}, "roboticshack/team2-guess_who_less_ligth": {0: "Pick up the card and place it in the slot."}, "jiajun001/eraser00_3": {0: "Pick up the white object from the table."}, 
    "Setchii/so100_grab_ball": {0: "Grasp a ball and put it on a goblet."},
    # V4 with VLM annotation
    'ctbfl/sort_battery': {0: 'put the battery into battery_box'}, 'lerobot/aloha_static_screw_driver': {0: 'Pick up the screwdriver with the right arm, hand it over to the left arm then place it into the cup.'}, 'lerobot/aloha_static_candy': {0: 'Pick up the candy and unwrap it.'}, 'lerobot/aloha_mobile_wipe_wine': {0: 'Pick up the wet cloth on the faucet and use it to clean the spilled wine on the table and underneath the glass.'}, 'lerobot/aloha_static_coffee': {0: "Place the coffee capsule inside the capsule container, then place the cup onto the center of the cup tray, then push the 'Hot Water' and 'Travel Mug' buttons."}, 'lerobot/aloha_static_towel': {0: 'Pick up a piece of paper towel and place it on the spilled liquid.'}, 'lerobot/aloha_static_vinh_cup': {0: 'Pick up the platic cup with the right arm, then pop its lid open with the left arm.'}, 'lerobot/aloha_static_vinh_cup_left': {0: 'Pick up the platic cup with the left arm, then pop its lid open with the right arm.'}, 'lerobot/aloha_static_ziploc_slide': {0: 'Slide open the ziploc bag.'}, 'lerobot/aloha_static_coffee_new': {0: 'Place the coffee capsule inside the capsule container, then place the cup onto the center of the cup tray.'}, 'lerobot/aloha_static_cups_open': {0: 'Pick up the plastic cup and open its lid.'}, 'lerobot/aloha_static_pro_pencil': {0: 'Pick up the pencil with the right arm, hand it over to the left arm then place it back onto the table.'}, 'lerobot/aloha_mobile_wash_pan': {0: 'Pick up the pan, rinse it in the sink and then place it in the drying rack.'}, 'lerobot/aloha_mobile_cabinet': {0: 'Open the top cabinet, store the pot inside it then close the cabinet.'}, 'lerobot/aloha_mobile_chair': {0: 'Push the chairs in front of the desk to place them against it.'}, 'lerobot/aloha_mobile_elevator': {0: 'Take the elevator to the 1st floor.'}, 'aliberts/koch_tutorial': {0: 'Pick the Lego block and drop it in the box on the right.'}, 'underctrl/single-block_multi-color_pick-up_50': {0: 'Pick single block of multi-color  and drop it in the box on the right.'}, 'underctrl/single-block_blue-color_pick-up_80': {0: 'Pick single block of multi-color  and drop it in the box on the right.'}, 'underctrl/mutli-stacked-block_mutli-color_pick-up_80': {0: 'Pick single block of multi-color  and drop it in the box on the right.'}, 'underctrl/single-stacked-block_two-color_pick-up_80': {0: 'Pick single block of multi-color  and drop it in the box on the right.'}, 'underctrl/single-stacked-block_mutli-color_pick-up_80': {0: 'Pick single block of multi-color  and drop it in the box on the right.'}, 'underctrl/handcamera_single_blue': {0: 'Pick the Lego block and drop it in the box on the right.'}, 'cmcgartoll/cube_color_organizer': {0: 'Organize blue cube', 1: 'Organize red cube', 2: 'Organize yellow cube'}, 'T-K-233/koch_k1_pour_shot': {0: 'Place the glass on the table.'}, 'Beegbrain/stack_2_cubes': {0: 'picks up the red block.'}, 'seeingrain/pick_place_lego': {0: 'Place the cube in the basket.'}, 'seeingrain/pick_place_lego_wider_range_richard': {0: 'Pick up the blue cube and place it in the basket.'}, 'seeingrain/pick_place_lego_wider_range_dang': {0: 'Pick up the cube and place it in the basket.'}, 'seeingrain/pick_place_lego_wider_range_dong': {0: 'Pick up the blue object and place it in the basket.'}, 'seeingrain/pick_lego_to_hand': {0: 'places blue object on table.'}, 'seeingrain/pick_place_pink_lego': {0: 'Pick up the red cube and place it in the basket.'}, 'seeingrain/pick_place_pink_lego_few_samples': {0: 'pick_place_pink_lego_few_samples '}, 'seeingrain/one_shot_learning_18episodes': {0: 'Pick up the red block and place it in the basket.'}, 'helper2424/hil-serl-push-circle-classifier': {0: 'Push small circle object to the correct position'}, 'seeingrain/lego_3cameras': {0: 'Pick up the red block and place it in the basket.'}, 'Lugenbott/koch_1225': {0: 'Pick up the blue block and place it in the red box.'}, 'twerdster/koch_training_red': {0: 'Pick up the red block.'}, 'dboemer/koch_50-samples': {0: 'Pick up the red block and place it on top of the yellow box.'}, 'seeingrain/241228_pick_place_2cams': {0: 'Place the cube in the basket.'}, 'Eyas/grab_pink_lighter_10_per_loc': {0: 'Pick up the pink object from the table.'}, 'Eyas/grab_bouillon': {0: 'picks up the box and places it in the box.'}, 'twerdster/koch_new_training_red': {0: 'move red cube into cellotape circle'}, 'andabi/shoes_easy': {0: 'picks up the shoe.'}, 'andabi/D2': {0: 'Pick up the shoe and place it on the table.'}, 'Beegbrain/oc_stack_cubes': {0: 'stack the red cube on the blue cube'}, 'abougdira/cube_target': {0: 'put_the cube on the yellow target'}, 'andabi/D3': {0: 'picks up the shoe.'}, 'andabi/D4': {0: 'picks up the shoe.'}, 'andabi/D5': {0: 'places shoe on table.'}, 'andabi/D6': {0: 'picks up the shoe.'}, 'andabi/D7': {0: 'picks up the shoe.'}, 'jainamit/koch_realcube3': {0: 'pick up the cube real with keyboard input'}, 'jainamit/koch_pickcube': {0: 'Pick up the blue cube and place it in the box.'}, 'andabi/D8': {0: 'picks up the shoe.'}, 'andabi/D9': {0: 'The  picks up the paper and places it on the table.'}, 'andabi/D10': {0: 'picks up the shoe.'}, 'andabi/D11': {0: 'The  picks up the paper and places it on the table.'}, 'andabi/D12': {0: 'The  places the cube in the box.'}, 'andabi/D13': {0: 'picks up shoes.'}, 'andabi/D14': {0: 'picks up shoes.'}, 'andabi/D15': {0: 'picks up the shoe.'}, 'rgarreta/koch_pick_place_lego': {0: 'Pick the Lego block and drop it in the box on the right.'}, 'shin1107/koch_train_block': {0: 'Grasp a block and put it in the hole.'}, 'andabi/D16': {0: 'picks up the shoe.'}, 'TrossenRoboticsCommunity/aloha_fold_tshirt': {0: 'Fold the t-shirt.'}, 'rgarreta/koch_pick_place_lego_v2': {0: 'Pick the Lego block and drop it in the box on the right.'}, '1g0rrr/screw1': {0: 'Grasp a lego block and put it in the bin.'}, 'ncavallo/moss_train_grasp': {0: 'Grasp a lego block and put it in the bin.'}, 'andabi/D17': {0: 'picks up the shoe.'}, 'ma3oun/rpi_squares_1': {0: 'Raspberry Pi 5 squares recording 1'}, 'shin1107/koch_move_block_with_some_shapes': {0: 'Grasp a block and put it in the hole with some shapes.'}, 'jannick-st/classifier': {0: 'Move the blue object to the right side of the table.'}, 'rgarreta/koch_pick_place_lego_v3': {0: 'Pick the Lego block and drop it in the box on the right. Top and wrist cameras.'}, 'TrossenRoboticsCommunity/aloha_stationary_logo_assembly': {0: 'Assemble the Trossen Robotics Logo.'}, 'rgarreta/koch_pick_place_lego_v6': {0: 'Grasp a lego block and put it in the bin.'}, 'Beegbrain/put_green_into_blue_bin': {0: 'Put the green cube into the blue bin'}, 'Beegbrain/put_screwdriver_box': {0: 'Put the screwdriver into the box'}, 'Beegbrain/align_three_pens': {0: 'picks up a pen.'}, 'Beegbrain/stack_green_on_blue_cube': {0: 'Stack the blue cube on top of the green cube'}, 'Beegbrain/align_cubes_green_blue': {0: 'Put the green cubes on the left and the blue cube on the right'}, 'IPEC-COMMUNITY/ucsd_kitchen_dataset_lerobot': {0: 'Turn on the faucet', 1: 'Put the bowl inside the kitchen cabinet', 2: 'Open the oven door', 3: 'Place the teapot on the stove', 4: 'Put the white box into the sink', 5: 'Open the carbinet door', 6: 'Put the green box into the sink', 7: 'Put the canned spam into the sink'}, 'dkdltu1111/omx-bottle1': {0: 'Grasp a lego block and put it in the bin.'}, 'rgarreta/koch_pick_place_lego_v7': {0: 'Grasp a lego block and put it in the bin.'}, 'rgarreta/koch_pick_place_lego_v8': {0: 'Grasp a lego block and put it in the bin.'}, 'IPEC-COMMUNITY/berkeley_mvp_lerobot': {0: 'push wooden cube', 1: 'pick detergent from the sink', 2: 'reach red block', 3: 'pick yellow cube', 4: 'close fridge door', 5: 'pick fruit'}, 'BlobDieKatze/GrabBlocks': {0: 'Grasp a lego block and put it in the bin.'}, 'Yuanzhu/koch_bimanual_grasp_0': {0: 'places the yellow block on the mousepad.'}, 'Yuanzhu/koch_bimanual_grasp_3': {0: 'picks up a yellow block.'}, 'takuzennn/aloha-pick100': {0: 'arm pick pen and put it into the cup'}, 'abbyoneill/pusht': {0: 'Grasp a lego block and put it in the bin.'}, 'ncavallo/moss_train_grasp_new': {0: 'Grasp a lego block and put it in the bin.'}, 'mlfu7/pi0_conversion_no_pad_video': {0: 'pickplace deer greybowl', 1: 'stack red on green', 2: 'stack orange cup to yellow cup', 3: 'put orange cup into yellow cup', 4: 'put push red pen to blue pen', 5: 'put tiger to black bowl', 6: 'put potato in bot to black bowl', 7: 'pick up green triangle', 8: 'put push blue pen to red pen', 9: 'close drawer', 10: 'put push green block to red', 11: 'pickup potato', 12: 'open drawer', 13: 'put closing tongs', 14: 'poke block', 15: 'put push blue cube', 16: 'poke tiger', 17: 'pick red cube into black bowl', 18: 'pick blue cube stack on wood block', 19: 'pick blue cube into grey bowl', 20: 'put red ball in black bowl', 21: 'pick green triangle into pink bowl', 22: 'pick red ball into pink bowl', 23: 'poke green triangle', 24: 'poke grey bowl', 25: 'put blue cube pink bowl', 26: 'put red cube into black bowl', 27: 'put deer into gray bowl', 28: 'put red ball into pink bowl', 29: 'put green triangle into pink bowl', 30: 'put pour from yellow cup into black bowl', 31: 'put pour blue cup into pink bowl', 32: 'put brown cube into gray bowl'}, 'pepijn223/lekiwi_pen': {0: 'Fold the jeans.'}, 'TrossenRoboticsCommunity/aloha_baseline_dataset': {0: 'Pick up a blue block and put it in a green bowl. Baseline dataset for testing'}, 'KeWangRobotics/piper_rl_1': {0: 'Pick up the cube and place it in the box.'}, 'nduque/cam_setup2': {0: 'Grasp a green block  and put it in the bin.'}, 'KeWangRobotics/piper_rl_1_cropped_resized': {0: 'picks up the cube.'}, 'abbyoneill/new_dataset_pick_place': {0: 'Grasp a lego block and put it in the bin.'}, 'abbyoneill/thurs1120pickplace': {0: 'Grasp a lego block and put it in the bin.'}, 'abbyoneill/data_w_mug': {0: 'Grasp a lego block and put it in the bin.'}, 'pepijn223/lekiwi_drive_in_circle': {0: 'Pick up the red object and place it on the table.'}, 'pepijn223/lekiwi_block_cleanup2': {0: 'Put red block in black box'}, 'KeWangRobotics/piper_rl_2': {0: 'Move the cube to the right side of the table.'}, 'KeWangRobotics/piper_rl_2_cropped_resized': {0: 'Move the block to the right side of the table.'}, 'hannesill/koch_pnp_simple_50': {0: 'Grasp a small block with a specific orientation and put it in the bin with a specific position and orientation.'}, 'KeWangRobotics/piper_rl_3': {0: 'Pick up the wooden block.'}, 'KeWangRobotics/piper_rl_3_cropped_resized': {0: 'Place the block in the box.'}, 'hannesill/koch_pnp_2_blocks_2_bins_200': {0: 'Grasp the blue block first and put it in the first bin that has a specific position and orientation. Then grasp the white block and put it in the second bin that has a specific position and orientation.'}, 'ellen2imagine/pusht_green1': {0: 'Place the green block in the box.'}, 'imatrixlee/koch_place': {0: 'Pick up the white object and place it on the table.'}, 'ellen2imagine/pusht_green_same_init2': {0: 'Place the green block in the correct position.'}, 'KeWangRobotics/piper_rl_4': {0: 'Move the block slightly.'}, 'KeWangRobotics/piper_rl_4_cropped_resized': {0: 'picks the wooden block.'}, 'lalalala0620/koch_blue_paper_tape': {0: 'Grasp a blue paper tape and put it in the bin.'}, 'nduque/act_50_ep': {0: 'Grasp a green block and put it in the bin.'}, 'nduque/act_50_ep2': {0: 'Grasp a green block and put it in the bin.'}, 'HWJ658970/fat_fish': {0: 'Grasp a fat fish toy and put it in the bin.'}, 'Beegbrain/put_red_triangle_green_rect': {0: 'Put the red triangle on top of the green rectangle'}, 'ncavallo/moss_train_gc_block': {0: 'Grasp a lego block and put it in the bin.'}, 'takuzennn/square3': {0: 'Pick the cube from the table.'}, 'HWJ658970/lego_50': {0: 'Grasp a yellow lego block and put it in the bin.'}, 'Deason11/mobile_manipulator_0319': {0: 'Grasp a lego block and put it in the bin.'}, 'Gongsta/grasp_duck_in_cup': {0: 'Grasp the rubber duck and put it in the cup.'}, 'nduque/robustness_e2': {0: 'Grasp a green dice and put it in the bin.'}, 'ibru/bobo_trash_collector': {0: 'Bobo Trash collect and place it in a bin'}, 'Beegbrain/moss_open_drawer_teabox': {0: 'Open the top drawer of the teabox'}, 'Beegbrain/moss_put_cube_teabox': {0: 'Put the green cube in the top drawer of the teabox'}, 'Beegbrain/moss_close_drawer_teabox': {0: 'Close the top drawer of the teabox'}, 'Beegbrain/moss_stack_cubes': {0: 'Stack the green cube on top of the blue cube'}, 'HWJ658970/lego_50_camera_change': {0: 'Grasp a yellow lego block and put it in the bin.'}, 'HWJ658970/lego_100_class': {0: 'Separate yellow and white Lego blocks and place them into the bin.'}, 'nduque/robustness_e3': {0: 'Grasp a green dice and put it in the bin.'}, 'nimitvasavat/Gr00t_lerobot': {0: 'Place the cereal box on the shelf.'}, 'Deason11/mobile_manipulator_0326': {0: 'Grasp a lego block and put it in the bin.', 1: 'mobile_lekiwi.'}, 'KeWangRobotics/piper_push_cube_gamepad_1': {0: 'push the cube to the black area'}, 'KeWangRobotics/piper_push_cube_gamepad_1_cropped_resized': {0: 'push the cube to the black area'}, 'jannick-st/push-cube-classifier_cropped_resized': {0: 'Close the cabinet door.'}, 'nimitvasavat/Gr00t_lerobotV2': {0: 'Place the chocolate chip cookie dough box on the table.'}, 'Zhaoting123/koch_cleanDesk_': {0: 'Grasp a card and use it to clean the desk'}, 'arclabmit/koch_gear_and_bin': {0: 'Pick the gear and place it in the bin.'}, 'Allen-488/koch_dataset_50': {0: 'Grasp a block and put it in the bin.'}, 'dop0/koch_pick_terminal': {0: 'Pick up the terminal and place on the cover.'}, 'nduque/robustness_e4': {0: 'Grasp a green dice and put it in the bin.'}, 'arclabmit/Koch_twoarms': {0: 'official two arms recordings10'}, 'nimitvasavat/Gr00t_lerobot_state_action': {0: 'Place the chocolate chip cookie dough box on the table.'}, 'zliu157/i3r': {0: 'Grasp a lego block and put it in the bin.'}, 'hangwu/koch_pick_terminal': {0: 'Pick up the terminal and place on the cover.'}, 'nduque/robustness_e5': {0: 'Grasp a green dice and put it in the bin.'}, 'zliu157/i3r2': {0: 'Grasp a i3r logo and put it in the bin.'}, 'HuaihaiLyu/groceries': {0: 'Pick the brown long bread and Egg yolk pasry into package'}, 'zliu157/i3r3': {0: 'Grasp a i3r logo and put it in the bin.'}, 'hangwu/piper_pick_terminal_and_place': {0: 'Grasp a terminal and put it on the black box.'}, 'hangwu/piper_pick_terminal_2': {0: 'Grasp the white terminal and put it on the green lid.'}, 'engineer0002/pepper': {0: 'Place the bottle on the table.'}, 'theo-michel/lekiwi_v2': {0: 'Pick up the can on the ground'}, 'theo-michel/lekiwi_v5': {0: 'Pick up the can on the ground'}, 'roboticshack/sandee-kiwiv10': {0: 'Place the bottle on the table.'}, 'ibru/bob_jetson': {0: 'Drive forward pickup the object and put it in the red box and drive back.'}, 'ibru/bobo_jetson': {0: 'Drive forward pickup the object and put it in the red box and drive back.', 1: 'Driver forward'}, 'zliu157/i3r5': {0: 'Grasp a i3r logo and put it in the bin.'}, 'Dongkkka/cable_pick_and_place2': {0: 'Put a black charging cable in a black bowl and put a red charging cable in a green bowl'},


}
