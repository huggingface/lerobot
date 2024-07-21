"""
NOTE(YL): Adapted from:
    OpenVLA: https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/rlds/oxe/configs.py
    Octo: https://github.com/octo-models/octo/blob/main/octo/data/oxe/oxe_dataset_configs.py

TODO: implement the following:
    - Populate all `fps` for each dataset
    - Upload the dataset config to the Readme of each dataset on huggingface hub, for verbosity

configs.py

Defines per-dataset configuration (kwargs) for each dataset in Open-X Embodiment.

Configuration adopts the following structure:
    image_obs_keys:
        primary: primary external RGB
        secondary: secondary external RGB
        wrist: wrist RGB

    depth_obs_keys:
        primary: primary external depth
        secondary: secondary external depth
        wrist: wrist depth

    # Always 8-dim =>> changes based on `StateEncoding`
    state_obs_keys:
        StateEncoding.POS_EULER:    EEF XYZ (3) + Roll-Pitch-Yaw (3) + <PAD> (1) + Gripper Open/Close (1)
        StateEncoding.POS_QUAT:     EEF XYZ (3) + Quaternion (4) + Gripper Open/Close (1)
        StateEncoding.JOINT:        Joint Angles (7, <PAD> if fewer) + Gripper Open/Close (1)

    state_encoding: Type of `StateEncoding`
    action_encoding: Type of action encoding (e.g., EEF Position vs. Joint Position)
"""

from enum import IntEnum


# Defines Proprioceptive State Encoding Schemes
class StateEncoding(IntEnum):
    # fmt: off
    NONE = -1               # No Proprioceptive State
    POS_EULER = 1           # EEF XYZ (3) + Roll-Pitch-Yaw (3) + <PAD> (1) + Gripper Open/Close (1)
    POS_QUAT = 2            # EEF XYZ (3) + Quaternion (4) + Gripper Open/Close (1)
    JOINT = 3               # Joint Angles (7, <PAD> if fewer) + Gripper Open/Close (1)
    JOINT_BIMANUAL = 4      # Joint Angles (2 x [ Joint Angles (6) + Gripper Open/Close (1) ])
    # fmt: on


# Defines Action Encoding Schemes
class ActionEncoding(IntEnum):
    # fmt: off
    EEF_POS = 1             # EEF Delta XYZ (3) + Roll-Pitch-Yaw (3) + Gripper Open/Close (1)
    JOINT_POS = 2           # Joint Delta Position (7) + Gripper Open/Close (1)
    JOINT_POS_BIMANUAL = 3  # Joint Delta Position (2 x [ Joint Delta Position (6) + Gripper Open/Close (1) ])
    EEF_R6 = 4              # EEF Delta XYZ (3) + R6 (6) + Gripper Open/Close (1)
    EEF_VEL = 5
    # fmt: on


# === Individual Dataset Configs ===
OXE_DATASET_CONFIGS = {
    "fractal20220817_data": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["base_pose_tool_reached", "gripper_closed"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 3,
    },
    "kuka": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [
            "clip_function_input/base_pose_tool_reached",
            "gripper_closed",
        ],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 10
    },
    "bridge_oxe": {  # Version of Bridge V2 in Open X-Embodiment mixture
        "image_obs_keys": {"primary": "image", "secondary": "image_1", "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 5,
    },
    "bridge_orig": {  # Original version of Bridge V2 from project website
        "image_obs_keys": {"primary": "image_0", "secondary": "image_1", "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 5,
    },
    "bridge_dataset": {  # Original version of Bridge V2 from project website
        "image_obs_keys": {"primary": "image_0", "secondary": "image_1", "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 5,
    },
    "taco_play": {
        "image_obs_keys": {
            "primary": "rgb_static",
            "secondary": None,
            "wrist": "rgb_gripper",
        },
        "depth_obs_keys": {
            "primary": "depth_static",
            "secondary": None,
            "wrist": "depth_gripper",
        },
        "state_obs_keys": ["state_eef", None, "state_gripper"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 15,
    },
    "jaco_play": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "image_wrist",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state_eef", None, "state_gripper"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 10,
    },
    "berkeley_cable_routing": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": "top_image",
            "wrist": "wrist45_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["robot_state", None],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_VEL,
        "fps": 10,
    },
    "roboturk": {
        "image_obs_keys": {"primary": "front_rgb", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [None, None, None, None, None, None, None, None],
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 10,

    },
    "nyu_door_opening_surprising_effectiveness": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [None, None, None, None, None, None, None, None],
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 3,
    },
    "viola": {
        "image_obs_keys": {
            "primary": "agentview_rgb",
            "secondary": None,
            "wrist": "eye_in_hand_rgb",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["joint_states", "gripper_states"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 20,
    },
    "berkeley_autolab_ur5": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "hand_image",
        },
        "depth_obs_keys": {"primary": "depth", "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 5,
    },
    "toto": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.JOINT_POS,
        "fps": 30,
    },
    "language_table": {
        "image_obs_keys": {"primary": "rgb", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["effector_translation", None, None, None, None, None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 10,
    },
    "columbia_cairlab_pusht_real": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["robot_state", None, None, None, None, None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 10,
    },
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["ee_position", "ee_orientation", None],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 20,
    },
    "nyu_rot_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 3,
    },
    "stanford_hydra_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 10,
    },
    "austin_buds_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 20,
    },
    "nyu_franka_play_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": "image_additional_view",
            "wrist": None,
        },
        "depth_obs_keys": {
            "primary": "depth",
            "secondary": "depth_additional_view",
            "wrist": None,
        },
        "state_obs_keys": ["eef_state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_VEL,
        "fps": 3,
    },
    "maniskill_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {
            "primary": "depth",
            "secondary": None,
            "wrist": "wrist_depth",
        },
        "state_obs_keys": ["tcp_pose", "gripper_state"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 20,
    },
    "furniture_bench_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_VEL,
        "fps": 10,
    },
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "highres_image",
            "secondary": None,
            "wrist": None,
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [None, None, None, None, None, None, None, None],
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 10,
    },
    "ucsd_kitchen_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["joint_state", None],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 2,
    },
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_VEL,
        "fps": 3,
    },
    "austin_sailor_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_VEL,
        "fps": 20,
    },
    "austin_sirius_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_VEL,
        "fps": 20,
    },
    "bc_z": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [
            "present/xyz",
            "present/axis_angle",
            None,
            "present/sensed_close",
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 10,
    },
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 10,
    },
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 10,
    },
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": "image2",
            "wrist": "hand_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["end_effector_pose", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 10,
    },
    "utokyo_xarm_bimanual_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["pose_r", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 10,
    },
    "robo_net": {
        "image_obs_keys": {"primary": "image", "secondary": "image1", "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_VEL,
        "fps": 1,
    },
    "berkeley_mvp_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "hand_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["pose", "gripper"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.JOINT_POS,
        "fps": 5,
    },
    "berkeley_rpt_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "hand_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["joint_pos", "gripper"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.JOINT_POS,
        "fps": 30,
    },
    "kaist_nonprehensile_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 10,
    },
    "stanford_mask_vit_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "tokyo_u_lsmo_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_VEL,
        "fps": 10
    },
    "dlr_sara_pour_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_VEL,
        "fps": 10,
    },
    "dlr_sara_grid_clamp_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_VEL,
        "fps": 10,
    },
    "dlr_edan_shared_control_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_VEL,
        "fps": 5,
    },
    "asu_table_top_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 12.5,
    },
    "stanford_robocook_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image_1", "secondary": "image_2", "wrist": None},
        "depth_obs_keys": {"primary": "depth_1", "secondary": "depth_2", "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 5,
    },
    "imperialcollege_sawyer_wrist_cam": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [None, None, None, None, None, None, None, "state"],
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 10,
    },
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["joint_state", "gripper_state"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 20,
    },
    "uiuc_d3field": {
        "image_obs_keys": {"primary": "image_1", "secondary": "image_2", "wrist": None},
        "depth_obs_keys": {"primary": "depth_1", "secondary": "depth_2", "wrist": None},
        "state_obs_keys": [None, None, None, None, None, None, None, None],
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 1,
    },
    "utaustin_mutex": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 20,
    },
    "berkeley_fanuc_manipulation": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["joint_state", None, "gripper_state"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 10,
    },
    "cmu_playing_with_food": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "finger_vision_1",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 10,
    },
    "cmu_play_fusion": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 5,
    },
    "cmu_stretch": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 10,
    },
    "berkeley_gnm_recon": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_VEL,
        "fps": 3,
    },
    "berkeley_gnm_cory_hall": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_VEL,
        "fps": 5,
    },
    "berkeley_gnm_sac_son": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 10,
    },
    "droid": {
        "image_obs_keys": {
            "primary": "exterior_image_1_left",
            "secondary": "exterior_image_2_left",
            "wrist": "wrist_image_left",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 15,
    },
    "droid100": {  # For testing
        "image_obs_keys": {
            "primary": "exterior_image_1_left",
            "secondary": "exterior_image_2_left",
            "wrist": "wrist_image_left",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 15,
    },
    "fmb_dataset": {
        "image_obs_keys": {
            "primary": "image_side_1",
            "secondary": "image_side_2",
            "wrist": "image_wrist_1",
        },
        "depth_obs_keys": {
            "primary": "image_side_1_depth",
            "secondary": "image_side_2_depth",
            "wrist": "image_wrist_1_depth",
        },
        "state_obs_keys": ["proprio"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_VEL,
        "fps": 10,
    },
    "dobbe": {
        "image_obs_keys": {"primary": "wrist_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 3.75,
    },
    "roboset": {
        "image_obs_keys": {
            "primary": "image_left",
            "secondary": "image_right",
            "wrist": "image_wrist",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.JOINT_POS,
        "fps": 5,
    },
    "rh20t": {
        "image_obs_keys": {
            "primary": "image_front",
            "secondary": "image_side_right",
            "wrist": "image_wrist",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    ### T-DROID datasets
    "tdroid_carrot_in_bowl": {  # "put carrot in bowl" task, 50 demos @ 5 Hz control
        "image_obs_keys": {"primary": "static_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "static_depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 15,
    },
    "tdroid_pour_corn_in_pot": {  # "pour corn from red bowl into steel pot" task, 50 demos @ 5 Hz control
        "image_obs_keys": {"primary": "static_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "static_depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 15,
    },
    "tdroid_flip_pot_upright": {  # "flip pot upright" task, 10 demos @ 5 Hz control
        "image_obs_keys": {"primary": "static_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "static_depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 15,
    },
    "tdroid_move_object_onto_plate": {  # "move <object> onto plate" task, 150 demos @ 5 Hz control
        "image_obs_keys": {"primary": "static_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "static_depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 15,
    },
    "tdroid_knock_object_over": {  # "knock <object> over" task, 70 demos @ 5 Hz control
        "image_obs_keys": {"primary": "static_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "static_depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 15,
    },
    "tdroid_cover_object_with_towel": {  # "cover <object> with towel" task, 45 demos @ 5 Hz control
        "image_obs_keys": {"primary": "static_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "static_depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 15,
    },
    ### DROID Finetuning datasets
    "droid_wipe": {
        "image_obs_keys": {
            "primary": "exterior_image_2_left",
            "secondary": None,
            "wrist": "wrist_image_left",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "fps": 15,
    },
}
