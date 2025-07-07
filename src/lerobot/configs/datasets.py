from lerobot.common.constants import ACTION, OBS_IMAGE, OBS_IMAGE_2, OBS_IMAGE_3, OBS_STATE, OBS_IMAGE_4, TASK, ROBOT

IMAGES_ORDER = {
    OBS_IMAGE: 0,
    OBS_IMAGE_2: 1,
    OBS_IMAGE_3: 2,
    OBS_IMAGE_4: 3,
}
ROBOT_TYPE_KEYS_MAPPING = {
    "lerobot/stanford_hydra_dataset": "static_single_arm",
    "lerobot/iamlab_cmu_pickup_insert": "static_single_arm",
    "lerobot/berkeley_fanuc_manipulation": "static_single_arm",
    "lerobot/toto": "static_single_arm",
    "lerobot/roboturk": "static_single_arm",
    "lerobot/jaco_play": "static_single_arm",
    "lerobot/taco_play": "static_single_arm_7statedim",
}
TRAINING_FEATURES = {
    0: [ACTION, OBS_STATE, TASK, ROBOT, OBS_IMAGE],
    1: [ACTION, OBS_STATE, TASK, ROBOT, OBS_IMAGE, OBS_IMAGE_2],
    2: [ACTION, OBS_STATE, TASK, ROBOT, OBS_IMAGE, OBS_IMAGE_2, OBS_IMAGE_3],
}
# Map to "observation.state", "action", "observation.image", etc.
FEATURE_KEYS_MAPPING = {
    "lerobot/aloha_mobile_cabinet": {
        "observation.images.cam_high": OBS_IMAGE,
        "observation.images.cam_left_wrist": OBS_IMAGE_2,
        "observation.images.cam_right_wrist": OBS_IMAGE_3,
        "observation.effort": None,
        "observation.images.cam_low": None,
    },
    "lerobot/aloha_static_cups_open": {
        "observation.images.cam_high": OBS_IMAGE,
        "observation.images.cam_left_wrist": OBS_IMAGE_2,
        "observation.images.cam_right_wrist": OBS_IMAGE_3,
        "observation.images.cam_low": None,
        "observation.effort": None,
    },
    "lerobot/aloha_static_ziploc_slide": {
        "observation.images.cam_high": OBS_IMAGE,
        "observation.images.cam_left_wrist": OBS_IMAGE_2,
        "observation.images.cam_right_wrist": OBS_IMAGE_3,
        "observation.images.cam_low": None,
        "observation.effort": None,
    },
    "lerobot/aloha_static_vinh_cup": {
        "observation.images.cam_high": OBS_IMAGE,
        "observation.images.cam_left_wrist": OBS_IMAGE_2,
        "observation.images.cam_right_wrist": OBS_IMAGE_3,
        "observation.images.cam_low": None,
        "observation.effort": None,
    },
    "lerobot/aloha_static_vinh_cup_left": {
        "observation.images.cam_high": OBS_IMAGE,
        "observation.images.cam_left_wrist": OBS_IMAGE_2,
        "observation.images.cam_right_wrist": OBS_IMAGE_3,
        "observation.images.cam_low": None,
        "observation.effort": None,
    },
    "lerobot/aloha_static_coffee": {
        "observation.images.cam_high": OBS_IMAGE,
        "observation.images.cam_left_wrist": OBS_IMAGE_2,
        "observation.images.cam_right_wrist": OBS_IMAGE_3,
        "observation.images.cam_low": None,
        "observation.effort": None,
    },
    "lerobot/aloha_static_towel": {
        "observation.images.cam_high": OBS_IMAGE,
        "observation.images.cam_left_wrist": OBS_IMAGE_2,
        "observation.images.cam_right_wrist": OBS_IMAGE_3,
        "observation.images.cam_low": None,
        "observation.effort": None,
    },
    "lerobot/aloha_static_screw_driver": {
        "observation.images.cam_high": OBS_IMAGE,
        "observation.images.cam_left_wrist": OBS_IMAGE_2,
        "observation.images.cam_right_wrist": OBS_IMAGE_3,
        "observation.images.cam_low": None,
        "observation.effort": None,
    },
    "lerobot/aloha_mobile_wash_pan": {
        "observation.images.cam_high": OBS_IMAGE,
        "observation.images.cam_left_wrist": OBS_IMAGE_2,
        "observation.images.cam_right_wrist": OBS_IMAGE_3,
        "observation.images.cam_low": None,
        "observation.effort": None,
    },
    "lerobot/aloha_mobile_shrimp": {
        "observation.images.cam_high": OBS_IMAGE,
        "observation.images.cam_left_wrist": OBS_IMAGE_2,
        "observation.images.cam_right_wrist": OBS_IMAGE_3,
        "observation.images.cam_low": None,
        "observation.effort": None,
    },
    "lerobot/aloha_mobile_chair": {
        "observation.images.cam_high": OBS_IMAGE,
        "observation.images.cam_left_wrist": OBS_IMAGE_2,
        "observation.images.cam_right_wrist": OBS_IMAGE_3,
        "observation.images.cam_low": None,
        "observation.effort": None,
    },
    "lerobot/aloha_mobile_wipe_wine": {
        "observation.images.cam_high": OBS_IMAGE,
        "observation.images.cam_left_wrist": OBS_IMAGE_2,
        "observation.images.cam_right_wrist": OBS_IMAGE_3,
        "observation.images.cam_low": None,
        "observation.effort": None,
    },
    "lerobot/aloha_mobile_elevator": {
        "observation.images.cam_high": OBS_IMAGE,
        "observation.images.cam_left_wrist": OBS_IMAGE_2,
        "observation.images.cam_right_wrist": OBS_IMAGE_3,
        "observation.images.cam_low": None,
        "observation.effort": None,
    },
    "lerobot/aloha_sim_transfer_cube_human": {
        "observation.images.top": OBS_IMAGE,
    },
    "physical-intelligence/libero": {
        "image": OBS_IMAGE,
        "wrist_image": OBS_IMAGE_2,
        "state": OBS_STATE,
        "actions": ACTION,
    },
    "IPEC-COMMUNITY/libero_10_no_noops_image_lerobot": {
        "observation.images.image": OBS_IMAGE,
        # "observation.images.wrist_image": None,
        "observation.images.wrist_image": OBS_IMAGE_2,
    },
    "lerobot/libero_goal_image": {
        "observation.images.image": OBS_IMAGE,
        "observation.images.wrist_image": OBS_IMAGE_2,
    },
    "lerobot/libero_object_image": {
        "observation.images.image": OBS_IMAGE,
        "observation.images.wrist_image": OBS_IMAGE_2,
    },
    "lerobot/libero_spatial_image": {
        "observation.images.image": OBS_IMAGE,
        "observation.images.wrist_image": OBS_IMAGE_2,
    },
    "lerobot/libero_10_image": {
        "observation.images.image": OBS_IMAGE,
        "observation.images.wrist_image": OBS_IMAGE_2,
    },
    "lerobot/stanford_hydra_dataset": {
        "observation.images.image": OBS_IMAGE,
        "observation.images.wrist_image": OBS_IMAGE_2,
    },
    "lerobot/taco_play": {
        "observation.images.rgb_static": OBS_IMAGE,
        "observation.images.rgb_gripper": OBS_IMAGE_2,
    },
    "lerobot/jaco_play": {
        "observation.images.image": OBS_IMAGE,
        "observation.images.image_wrist": OBS_IMAGE_2,
    },
    "lerobot/roboturk": {
        "observation.images.front_rgb": OBS_IMAGE,
    },
    "lerobot/toto": {
        "observation.images.image": OBS_IMAGE,
    },
    "lerobot/iamlab_cmu_pickup_insert": {
        "observation.images.image": OBS_IMAGE,
        "observation.images.wrist_image": OBS_IMAGE_2,
    },
    "lerobot/berkeley_fanuc_manipulation": {
        "observation.images.image": OBS_IMAGE,
        "observation.images.wrist_image": OBS_IMAGE_2,
    },
    "cadene/droid_1.0.1": {
        "observation.images.exterior_1_left": OBS_IMAGE,
        "observation.images.wrist_left": OBS_IMAGE_2,
        "observation.images.exterior_2_left": OBS_IMAGE_3,
        "language_instruction": None,
        "language_instruction_2": None,
        "language_instruction_3": None,
        "observation.state.gripper_position": None,
        "observation.state.cartesian_position": None,
        "observation.state.joint_position": None,
        "action.gripper_position": None,
        "action.gripper_velocity":  None,
        "action.cartesian_position": None,
        "action.cartesian_velocity": None,
        "action.joint_position": None,
        "action.joint_velocity": None,
        "action.original": None,
        "discount": None,
        "camera_extrinsics.wrist_left": None, 
        "camera_extrinsics.exterior_1_left": None, 
        "camera_extrinsics.exterior_2_left": None, 
        "is_episode_successful": None,
        "task_category": None,
        "building": None,
        "collector_id": None,
        "date": None,
    },
    "danaaubakirova/svla_so100_task4_v3_multiple": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2,
    },
    "danaaubakirova/svla_so100_task1_v3": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2,
    },
    # Community datasets V1
    "pranavsaroha/so100_legos4": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE,
    },
    "pranavsaroha/so100_onelego2": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE,
    },
    "jpata/so100_pick_place_tangerine": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2,
    },
    "pranavsaroha/so100_onelego3": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE,
    },
    "pranavsaroha/so100_carrot_2": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE,
    },
    "pranavsaroha/so100_carrot_5": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE,
    },
    "pandaRQ/pick_med_1": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.laptop1": OBS_IMAGE_2,
        "observation.images.laptop2": OBS_IMAGE_3,
    },
    "HITHY/so100_strawberry": {
        "observation.images.laptop": OBS_IMAGE
    },
    "vladfatu/so100_above": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE,
    },
    "koenvanwijk/orange50-1": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "koenvanwijk/orange50-variation-2": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "FeiYjf/new_GtoR": {
        "observation.images.left": OBS_IMAGE,
        "observation.images.right": OBS_IMAGE_2
    },
    "CSCSXX/pick_place_cube_1.18": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.side": OBS_IMAGE_2
    },
    "vladfatu/so100_office": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "dragon-95/so100_sorting": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "dragon-95/so100_sorting_1": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "nbaron99/so100_pick_and_place4": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "Beegbrain/pick_place_green_block": {
        "observation.images.realsense_side": OBS_IMAGE_2,
        "observation.images.realsense_top": OBS_IMAGE
    },
    "Ityl/so100_recording2": {
        "observation.images.realsense_side": OBS_IMAGE_2,
        "observation.images.realsense_top": OBS_IMAGE
    },
    "dragon-95/so100_sorting_2": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "dragon-95/so100_sorting_3": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "aractingi/push_cube_offline_data": {
        "observation.images.front": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2
    },
    "HITHY/so100_peach3": {
        "observation.images.laptop": OBS_IMAGE
    },
    "HITHY/so100_peach4": {
        "observation.images.laptop": OBS_IMAGE
    },
    "shreyasgite/so100_legocube_50": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "shreyasgite/so100_base_env": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "triton7777/so100_dataset_mix": {
        "observation.images.s_left": None,
        "observation.images.s_right": OBS_IMAGE_3,
        "observation.images.gripper": OBS_IMAGE_2,
        "observation.images.top": OBS_IMAGE
    },
    "Deason11/Open_the_drawer_to_place_items": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "Deason11/PLACE_TAPE_PUSH_DRAWER": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "NONHUMAN-RESEARCH/SOARM100_TASK_VENDA": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "mikechambers/block_cup_14": {
        "observation.images.main_cam": OBS_IMAGE,
        "observation.images.secondary_cam": OBS_IMAGE_2
    },
    "samsam0510/tooth_extraction_3": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "samsam0510/tooth_extraction_4": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "samsam0510/cube_reorientation_2": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "samsam0510/cube_reorientation_4": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "samsam0510/glove_reorientation_1": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "DorayakiLin/so100_pick_charger_on_tissue": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "zijian2022/noticehuman3": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "liuhuanjim013/so100_th": {
        "observation.images.front": OBS_IMAGE,
        "observation.images.top": OBS_IMAGE_2
    },

    # Community datasets V2
    "00ri/so100_battery": {"observation.images.laptop": OBS_IMAGE_2,
                            "observation.images.phone": OBS_IMAGE},
    "00ri/so100_battery_bin_center": {"observation.images.laptop": OBS_IMAGE_2,
                                    "observation.images.phone": OBS_IMAGE},
    "1g0rrr/sam_openpi03": {"observation.images.laptop": OBS_IMAGE,
                            "observation.images.phone": OBS_IMAGE_2},
    "1g0rrr/sam_openpi_solder1": {"observation.images.laptop": OBS_IMAGE,
                                "observation.images.phone": OBS_IMAGE_2},
    "1g0rrr/sam_openpi_solder2": {"observation.images.laptop": OBS_IMAGE,
                                "observation.images.phone": OBS_IMAGE_2},
    "356c/so100_duck_reposition_1": {"observation.images.body": OBS_IMAGE_2,
                                    "observation.images.overhead": OBS_IMAGE,
                                    "observation.images.side": OBS_IMAGE_3},
    "356c/so100_nut_sort_1": {"observation.images.body": OBS_IMAGE_2,
                            "observation.images.overhead": OBS_IMAGE,
                            "observation.images.side": OBS_IMAGE_3},
    "AndrejOrsula/lerobot_double_ball_stacking_random": {"observation.images.realsense": OBS_IMAGE,
                                                        "observation.images.side": OBS_IMAGE_2},
    "Bartm3/dice2": {"observation.images.laptop": OBS_IMAGE,
                    "observation.images.phone": OBS_IMAGE_2},
    "Beegbrain/pick_lemon_and_drop_in_bowl": {"observation.images.realsense_side": OBS_IMAGE,
                                            "observation.images.realsense_top": OBS_IMAGE_2},
    "Beegbrain/sweep_tissue_cube": {"observation.images.realsense_side": OBS_IMAGE,
                                    "observation.images.realsense_top": OBS_IMAGE_2},
    "Chojins/chess_game_000_white_red": {"observation.images.laptop": OBS_IMAGE_2,
                                        "observation.images.phone": OBS_IMAGE},
    "Chojins/chess_game_001_blue_stereo": {"observation.images.laptop": OBS_IMAGE_2,
                                            "observation.images.phone": OBS_IMAGE},
    "Chojins/chess_game_001_red_stereo": {"observation.images.laptop": OBS_IMAGE_2,
                                        "observation.images.phone": OBS_IMAGE},
    "Chojins/chess_game_009_white": {"observation.images.laptop": OBS_IMAGE_2,
                                    "observation.images.phone": OBS_IMAGE},
    "CrazyYhang/A1234-B-C_mvA2B": {"observation.images.laptop": OBS_IMAGE,
                                    "observation.images.phone": OBS_IMAGE_2},
    "FsqZ/so100_1": {"observation.images.side": OBS_IMAGE},
    "Gano007/so100_medic": {"observation.images.laptop": OBS_IMAGE,
                            "observation.images.phone": OBS_IMAGE_2},
    "Jiafei1224/so100_pa222per": {"observation.images.laptop": OBS_IMAGE_2,
                                "observation.images.phone": OBS_IMAGE},
    "Jiangeng/so100_413": {"observation.images.top": OBS_IMAGE_2,
                            "observation.images.wrist_left": OBS_IMAGE},
    "LemonadeDai/so100_coca": {"observation.images.top": OBS_IMAGE,
                                "observation.images.wrist": OBS_IMAGE_2},
    "Loki0929/so100_duck": {"observation.images.third": OBS_IMAGE_3,
                            "observation.images.top": OBS_IMAGE,
                            "observation.images.wrist": OBS_IMAGE_2},
    "Mohamedal/put_banana": {"observation.images.realsense_side": OBS_IMAGE,
                            "observation.images.realsense_top": OBS_IMAGE_2},
    "Mwuqiu/so100_0408_muti": {"observation.images.laptop": OBS_IMAGE,
                                "observation.images.phone": OBS_IMAGE_2},
    "Odog16/so100_cube_drop_pick_v1": {"observation.images.workspace": OBS_IMAGE,
                                        "observation.images.wrist": OBS_IMAGE_2},
    "Odog16/so100_cube_stacking_v1": {"observation.images.Workspace": OBS_IMAGE,
                                    "observation.images.Wrist": OBS_IMAGE_2},
    "RasmusP/so100_Orange2Green": {"observation.images.phone": OBS_IMAGE,
                                    "observation.images.webcam": OBS_IMAGE_2},
    "VoicAndrei/so100_banana_to_plate_only": {"observation.images.right": OBS_IMAGE_2,
                                            "observation.images.top": OBS_IMAGE},
    "ZGGZZG/so100_drop0": {"observation.images.left": OBS_IMAGE_2,
                            "observation.images.up": OBS_IMAGE},
    "ad330/cubePlace": {"observation.images.phone": OBS_IMAGE,
                        "observation.images.wristCam": OBS_IMAGE_2},
    "aimihat/so100_tape": {"observation.images.laptop": OBS_IMAGE_2,
                            "observation.images.phone": OBS_IMAGE},
    "andlyu/so100_indoor_1": {"observation.images.arm_left": OBS_IMAGE_3,
                            "observation.images.arm_right": OBS_IMAGE_2,
                            "observation.images.base_left": None,
                            "observation.images.base_right": OBS_IMAGE},
    "andlyu/so100_indoor_3": {"observation.images.arm_left": OBS_IMAGE_3,
                            "observation.images.arm_right": OBS_IMAGE_2,
                            "observation.images.base_left": None,
                            "observation.images.base_right": OBS_IMAGE},
    "bensprenger/chess_game_001_blue_stereo": {"observation.images.laptop": OBS_IMAGE,
                                                "observation.images.phone": OBS_IMAGE_2},
    "bensprenger/left_arm_yellow_brick_in_box_v0": {"observation.images.laptop": OBS_IMAGE,
                                                    "observation.images.phone": OBS_IMAGE_2},
    "bensprenger/left_arm_yellow_brick_in_box_with_purple_noise_v0": {"observation.images.laptop": OBS_IMAGE,
                                                                    "observation.images.phone": OBS_IMAGE_2},
    "bensprenger/right_arm_p_brick_in_box_with_y_noise_v0": {"observation.images.laptop": OBS_IMAGE,
                                                            "observation.images.phone": OBS_IMAGE_2},
    "frk2/so100large": {"observation.images.top": OBS_IMAGE,
                        "observation.images.wrist": OBS_IMAGE_2},
    "frk2/so100largediffcam": {"observation.images.top": OBS_IMAGE,
                                "observation.images.wrist": OBS_IMAGE_2},
    "ganker5/so100_color_0328": {"observation.images.laptop": OBS_IMAGE,
                                "observation.images.phone": OBS_IMAGE_2},
    "ganker5/so100_dataline_0328": {"observation.images.laptop": OBS_IMAGE,
                                    "observation.images.phone": OBS_IMAGE_2},
    "ganker5/so100_dataline_20250331": {"observation.images.laptop": OBS_IMAGE,
                                        "observation.images.phone": OBS_IMAGE_2},
    "ganker5/so100_push_20250328": {"observation.images.laptop": OBS_IMAGE,
                                    "observation.images.phone": OBS_IMAGE_2},
    "ganker5/so100_push_20250331": {"observation.images.laptop": OBS_IMAGE,
                                    "observation.images.phone": OBS_IMAGE_2},
    "ganker5/so100_toy_20250402": {"observation.images.laptop": OBS_IMAGE,
                                    "observation.images.phone": OBS_IMAGE_2},
    "gxy1111/so100_pick_place": {"observation.images.eye": OBS_IMAGE,
                                "observation.images.wrist": OBS_IMAGE_2},
    "isadev/bougies3": {"observation.images.laptop": OBS_IMAGE,
                        "observation.images.phone": OBS_IMAGE_2},
    "jlesein/TestBoulon7": {"observation.images.robor": OBS_IMAGE_2,
                            "observation.images.top": OBS_IMAGE},
    "lirislab/close_top_drawer_teabox": {"observation.images.realsense_side": OBS_IMAGE_2,
                                        "observation.images.realsense_top": OBS_IMAGE},
    "lirislab/fold_bottom_right": {"observation.images.realsense_side": OBS_IMAGE_2,
                                    "observation.images.realsense_top": OBS_IMAGE},
    "lirislab/guess_who_so100": {"observation.images.mounted": OBS_IMAGE},
    "lirislab/lemon_into_bowl": {"observation.images.realsense_side": OBS_IMAGE_2,
                                "observation.images.realsense_top": OBS_IMAGE},
    "lirislab/open_top_drawer_teabox": {"observation.images.realsense_side": OBS_IMAGE_2,
                                        "observation.images.realsense_top": OBS_IMAGE},
    "lirislab/push_cup_target": {"observation.images.realsense_side": OBS_IMAGE_2,
                                "observation.images.realsense_top": OBS_IMAGE},
    "lirislab/put_banana_bowl": {"observation.images.realsense_side": OBS_IMAGE_2,
                                "observation.images.realsense_top": OBS_IMAGE},
    "lirislab/put_caps_into_teabox": {"observation.images.realsense_side": OBS_IMAGE_2,
                                    "observation.images.realsense_top": OBS_IMAGE},
    "lirislab/put_coffee_cap_teabox": {"observation.images.realsense_side": OBS_IMAGE_2,
                                        "observation.images.realsense_top": OBS_IMAGE},
    "lirislab/red_cube_into_blue_cube": {"observation.images.realsense_side": OBS_IMAGE_2,
                                        "observation.images.realsense_top": OBS_IMAGE},
    "lirislab/red_cube_into_green_lego_block": {"observation.images.realsense_side": OBS_IMAGE_2,
                                                "observation.images.realsense_top": OBS_IMAGE},
    "lirislab/so100_demo": {"observation.images.front": OBS_IMAGE},
    "lirislab/sweep_tissue_cube": {"observation.images.realsense_side": OBS_IMAGE_2,
                                    "observation.images.realsense_top": OBS_IMAGE},
    "lirislab/unfold_bottom_right": {"observation.images.realsense_side": OBS_IMAGE_2,
                                    "observation.images.realsense_top": OBS_IMAGE},
    "paszea/so100_lego": {"observation.images.laptop": OBS_IMAGE,
                        "observation.images.phone": OBS_IMAGE_2},
    "paszea/so100_lego_2cam": {"observation.images.front": OBS_IMAGE,
                                "observation.images.top": OBS_IMAGE_2},
    "paszea/so100_whale_2": {"observation.images.laptop": OBS_IMAGE,
                            "observation.images.phone": OBS_IMAGE_2},
    "paszea/so100_whale_3": {"observation.images.laptop": OBS_IMAGE,
                            "observation.images.phone": OBS_IMAGE_2},
    "paszea/so100_whale_4": {"observation.images.laptop": OBS_IMAGE,
                            "observation.images.phone": OBS_IMAGE_2},
    "pierfabre/chicken": {"observation.images.robot": OBS_IMAGE_2,
                        "observation.images.webcam": OBS_IMAGE},
    "pierfabre/cow2": {"observation.images.robot": OBS_IMAGE_2,
                        "observation.images.webcam": OBS_IMAGE},
    "pierfabre/horse": {"observation.images.robot": OBS_IMAGE_2,
                        "observation.images.webcam": OBS_IMAGE},
    "pierfabre/pig2": {"observation.images.robot": OBS_IMAGE_2,
                        "observation.images.webcam": OBS_IMAGE},
    "pierfabre/pig3": {"observation.images.robot": OBS_IMAGE_2,
                        "observation.images.webcam": OBS_IMAGE},
    "pierfabre/rabbit": {"observation.images.robot": OBS_IMAGE_2,
                        "observation.images.webcam": OBS_IMAGE},
    "pierfabre/sheep": {"observation.images.robot": OBS_IMAGE_2,
                        "observation.images.webcam": OBS_IMAGE},
    "pietroom/actualeasytask": {"observation.images.laptop": OBS_IMAGE_2,
                                "observation.images.phone": OBS_IMAGE},
    "pietroom/holdthis": {"observation.images.laptop": OBS_IMAGE_2,
                        "observation.images.phone": OBS_IMAGE},
    "roboticshack/left-arm-grasp-lego-brick": {"observation.images.laptop": OBS_IMAGE,
                                                "observation.images.phone": OBS_IMAGE_2},
    "roboticshack/team-7-left-arm-grasp-motor": {"observation.images.laptop": OBS_IMAGE,
                                                "observation.images.phone": OBS_IMAGE_2},
    "roboticshack/team-7-right-arm-grasp-tape": {"observation.images.laptop": OBS_IMAGE,
                                                "observation.images.phone": OBS_IMAGE_2},
    "roboticshack/team13-three-balls-stacking": {"observation.images.realsense": OBS_IMAGE,
                                                "observation.images.side": OBS_IMAGE_2},
    "roboticshack/team13-two-balls-stacking": {"observation.images.realsense": OBS_IMAGE,
                                                "observation.images.side": OBS_IMAGE_2},
    "roboticshack/team16-can-stacking": {"observation.images.head": OBS_IMAGE,
                                        "observation.images.wrist": OBS_IMAGE_2},
    "roboticshack/team16-water-pouring": {"observation.images.head": OBS_IMAGE,
                                        "observation.images.wrist": OBS_IMAGE_2},
    "roboticshack/team9-pick_chicken_place_plate": {"observation.images.static_left": OBS_IMAGE,
                                                    "observation.images.static_right": OBS_IMAGE_2},
    "roboticshack/team9-pick_cube_place_static_plate": {"observation.images.static_left": OBS_IMAGE,
                                                        "observation.images.static_right": OBS_IMAGE_2},
    "samanthalhy/so100_herding_1": {"observation.images.laptop": OBS_IMAGE,
                                    "observation.images.phone": OBS_IMAGE_2},
    "sihyun77/mond_1": {"observation.images.laptop": OBS_IMAGE,
                        "observation.images.phone": OBS_IMAGE_2},
    "sihyun77/sihyun_3_17_2": {"observation.images.laptop": OBS_IMAGE,
                                "observation.images.phone": OBS_IMAGE_2},
    "sihyun77/sihyun_3_17_5": {"observation.images.laptop": OBS_IMAGE,
                                "observation.images.phone": OBS_IMAGE_2},
    "sihyun77/sihyun_main_2": {"observation.images.laptop": OBS_IMAGE,
                                "observation.images.phone": OBS_IMAGE_2},
    "sihyun77/sihyun_main_3": {"observation.images.laptop": OBS_IMAGE,
                                "observation.images.phone": OBS_IMAGE_2},
    "sihyun77/suho_3_17_1": {"observation.images.laptop": OBS_IMAGE,
                            "observation.images.phone": OBS_IMAGE_2},
    "sihyun77/suho_3_17_3": {"observation.images.laptop": OBS_IMAGE,
                            "observation.images.phone": OBS_IMAGE_2},
    "sihyun77/suho_main_2": {"observation.images.laptop": OBS_IMAGE,
                            "observation.images.phone": OBS_IMAGE_2},
    "sixpigs1/so100_pick_cube_in_box": {"observation.images.above": OBS_IMAGE_2,
                                        "observation.images.rightfront": OBS_IMAGE},
    "sixpigs1/so100_stack_cube_error": {"observation.images.above": OBS_IMAGE_2,
                                        "observation.images.rightfront": OBS_IMAGE},
    "smanni/train_so100_fluffy_box": {"observation.images.intel_realsense": OBS_IMAGE},
    "therarelab/so100_pick_place_2": {"observation.images.laptop": OBS_IMAGE_2,
                                    "observation.images.phone": OBS_IMAGE},
    "tkc79/so100_lego_box_1": {"observation.images.arm": OBS_IMAGE_2,
                                "observation.images.laptop": OBS_IMAGE},
    "tkc79/so100_lego_box_2": {"observation.images.arm": OBS_IMAGE_2,
                                "observation.images.laptop": OBS_IMAGE},
    "weiye11/so100_410_zwy": {"observation.images.top": OBS_IMAGE,
                            "observation.images.wrist_left": OBS_IMAGE_2},
    "zijian2022/321": {"observation.images.laptop": OBS_IMAGE,
                        "observation.images.phone": OBS_IMAGE_2},
    "zijian2022/backgrounda": {"observation.images.laptop": OBS_IMAGE,
                                "observation.images.phone": OBS_IMAGE_2},
    "zijian2022/backgroundb": {"observation.images.laptop": OBS_IMAGE,
                                "observation.images.phone": OBS_IMAGE_2},
    "zijian2022/close3": {"observation.images.laptop": OBS_IMAGE,
                        "observation.images.phone": OBS_IMAGE_2},
    "zijian2022/insert2": {"observation.images.laptop": OBS_IMAGE,
                            "observation.images.phone": OBS_IMAGE_2},
    "zijian2022/sort1": {"observation.images.laptop": OBS_IMAGE,
                        "observation.images.phone": OBS_IMAGE_2},
    # Community datasets V3
    "satvikahuja/mixer_on_off_new_1": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": None,
        "observation.images.Lwebcam": OBS_IMAGE,
        "observation.images.macwebcam": OBS_IMAGE_3,
    },
    "aergogo/so100_pick_place": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2,
    },
    "andy309/so100_0314_fold_cloths": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.back": OBS_IMAGE_2
    },
    "jchun/so100_pickplace_small_20250323_120056": {
        "observation.images.main": OBS_IMAGE_2,
        "observation.images.cv": OBS_IMAGE_3,
        "observation.images.webcam": OBS_IMAGE
    },
    "astroyat/cube": {
        "observation.images.laptop": OBS_IMAGE
    },
    "Ofiroz91/so_100_cube2bowl": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "HappyPablo/dec3_data2": {
        "observation.images.laptop": None,
        "observation.images.phone": OBS_IMAGE_2
    },
    "ZCM5115/so100_1210": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.hikvision": OBS_IMAGE_2
    },
    "francescocrivelli/orange_feeding": {
        "observation.images.endeffector": OBS_IMAGE_2,
        "observation.images.workspace": OBS_IMAGE
    },
    "francescocrivelli/carrot_eating": {
        "observation.images.endeffector": OBS_IMAGE_2,
        "observation.images.workspace": OBS_IMAGE
    },
    "0x00raghu/toffee_red": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "0x00raghu/toffee_red_2": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "0x00raghu/toffee_red_3__": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "0x00raghu/toffee_blue": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "0x00raghu/toffee_blue_2": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "0x00raghu/toffee_to_hand_1": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "0x00raghu/toffee_to_hand_2": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "liyitenga/so100_bi_hello": {
        "observation.images.center": OBS_IMAGE,
        "observation.images.left_follower": OBS_IMAGE_2,
        "observation.images.right_follower": OBS_IMAGE_3
    },
    "liyitenga/so100_bi_giveme5": {
        "observation.images.center": OBS_IMAGE,
        "observation.images.left_follower": OBS_IMAGE_2,
        "observation.images.right_follower": OBS_IMAGE_3
    },
    "ZCM5115/so100_2Arm3cameras_movebox": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.left": OBS_IMAGE_2,
        "observation.images.right": OBS_IMAGE_3
    },
    "pranavsaroha/so100_carrot_1": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "pranavsaroha/so100_carrot_3": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "pranavsaroha/so100_carrot_4": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "maximilienroberti/so100_lego_red_box": {
        "observation.images.cam_left": OBS_IMAGE,
        "observation.images.cam_middle": OBS_IMAGE_3,
        "observation.images.cam_right": OBS_IMAGE_2
    },
    "pranavsaroha/so100_squishy": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "rabhishek100/so100_train_dataset": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "pranavsaroha/so100_squishy100": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "swarajgosavi/kikobot_pusht_real_v2": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "pandaRQ/pickmed": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.laptop1": OBS_IMAGE_2,
        "observation.images.laptop2": OBS_IMAGE_3
    },
    "swarajgosavi/act_kikobot_pusht_real": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "pranavsaroha/so100_squishy2colors": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "pranavsaroha/so100_squishy2colors_1": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "Chojins/chess_game_001_white": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "jmrog/so100_sweet_pick": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "Chojins/chess_game_002_white": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "pranavsaroha/so100_squishy2colors_2_new": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "Chojins/chess_game_003_white": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "aractingi/pick_place_lego_cube": {
        "observation.images.wrist": OBS_IMAGE,
        "observation.images.top": OBS_IMAGE_2
    },
    "Chojins/chess_game_004_white": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "Chojins/chess_game_005_white": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "Chojins/chess_game_006_white": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "Chojins/chess_game_007_white": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "koenvanwijk/blue2": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": None
    },
    "jlitch/so100multicam3": {
        "observation.images.wrist": OBS_IMAGE_2,
        "observation.images.overview2": OBS_IMAGE
    },
    "koenvanwijk/blue52": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": None
    },
    "jlitch/so100multicam6": {
        "observation.images.wrist": OBS_IMAGE_2,
        "observation.images.overview2": OBS_IMAGE
    },
    "aractingi/pick_place_lego_cube_1": {
        "observation.images.wrist": OBS_IMAGE,
        "observation.images.top": OBS_IMAGE_2
    },
    "jlitch/so100multicam7": {
        "observation.images.wrist": OBS_IMAGE_2,
        "observation.images.overview2": OBS_IMAGE
    },
    "vladfatu/so100_ds": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "Chojins/chess_game_000_white": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "HITHY/so100-kiwi": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "HITHY/so100_peach1": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "HITHY/so100_redstrawberry": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "satvikahuja/orange_mixer_1": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": None,
        "observation.images.Lwebcam": OBS_IMAGE,
        "observation.images.macwebcam": None
    },
    "satvikahuja/mixer_on_off": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": None,
        "observation.images.Lwebcam": OBS_IMAGE,
        "observation.images.macwebcam": None
    },
    "satvikahuja/orange_pick_place_new1": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": None,
        "observation.images.Lwebcam": OBS_IMAGE,
        "observation.images.macwebcam": None
    },
    "satvikahuja/mixer_on_off_new": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": None,
        "observation.images.Lwebcam": OBS_IMAGE,
        "observation.images.macwebcam": None
    },
    "danmac1/real_real332": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "FeiYjf/Makalu_push": {
        "observation.images.left": OBS_IMAGE_2,
        "observation.images.right": OBS_IMAGE
    },
    "liyitenga/so100_pick_taffy1": {
        "observation.images.left": OBS_IMAGE
    },
    "chmadran/so100_dataset04": {
        "observation.images.laptop": OBS_IMAGE
    },
    "FeiYjf/Maklu_dataset": {
        "observation.images.left": OBS_IMAGE,
        "observation.images.right": OBS_IMAGE_2
    },
    "FeiYjf/new_Dataset": {
        "observation.images.left": OBS_IMAGE,
        "observation.images.right": OBS_IMAGE_2
    },
    "liyitenga/so100_pick_taffy2": {
        "observation.images.gripper": OBS_IMAGE_2,
        "observation.images.top": OBS_IMAGE
    },
    "satvikahuja/mixer_on_off_new_4": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": None,
        "observation.images.Lwebcam": OBS_IMAGE,
        "observation.images.macwebcam": None
    },
    "CSCSXX/pick_place_cube_1.17": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.side": OBS_IMAGE_2
    },
    "liyitenga/so100_pick_taffy3": {
        "observation.images.gripper": OBS_IMAGE_2,
        "observation.images.top": OBS_IMAGE
    },
    "liyitenga/so100_pick_taffy4": {
        "observation.images.gripper": OBS_IMAGE_2,
        "observation.images.top": OBS_IMAGE
    },
    "yuz1wan/so100_pick_pink": {
        "observation.images.wrist": OBS_IMAGE_2,
        "observation.images.side": OBS_IMAGE
    },
    "yuz1wan/so100_pick_wahaha": {
        "observation.images.wrist": OBS_IMAGE_2,
        "observation.images.side": OBS_IMAGE
    },
    "yuz1wan/so100_pp_pink": {
        "observation.images.wrist": OBS_IMAGE_2,
        "observation.images.side": OBS_IMAGE
    },
    "yuz1wan/so100_pour_cup": {
        "observation.images.wrist": OBS_IMAGE_2,
        "observation.images.side": OBS_IMAGE
    },
    "liyitenga/so100_pick_taffy5": {
        "observation.images.gripper": OBS_IMAGE_2,
        "observation.images.top": OBS_IMAGE
    },
    "liyitenga/so100_pick_taffy6": {
        "observation.images.gripper": OBS_IMAGE_2,
        "observation.images.top": OBS_IMAGE
    },
    "yuz1wan/so100_button": {
        "observation.images.wrist": OBS_IMAGE_2,
        "observation.images.side": OBS_IMAGE
    },
    "yuz1wan/so100_pickplace": {
        "observation.images.wrist": OBS_IMAGE_2,
        "observation.images.side": OBS_IMAGE
    },
    "liyitenga/so100_pick_taffy7": {
        "observation.images.gripper": OBS_IMAGE_2,
        "observation.images.top": OBS_IMAGE,
        "observation.images.left_top": OBS_IMAGE_3
    },
    "FeiYjf/push_gg": {
        "observation.images.left": OBS_IMAGE_2,
        "observation.images.right": OBS_IMAGE
    },
    "FeiYjf/push_0094": {
        "observation.images.left": OBS_IMAGE,
        "observation.images.right": OBS_IMAGE_2
    },
    "swarajgosavi/act_kikobot_block_real": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "liyitenga/so100_pick_taffy8": {
        "observation.images.gripper": OBS_IMAGE_2,
        "observation.images.top": OBS_IMAGE
    },
    "phospho-ai/OrangeBrick3Cameras": {
        "observation.images.main.left": OBS_IMAGE_3,
        "observation.images.main.right": OBS_IMAGE_2,
        "observation.images.secondary_0": OBS_IMAGE
    },
    "vaishanthr/toy_pick_place": {
        "observation.images.webcam": OBS_IMAGE,
        "observation.images.gipper_cam": OBS_IMAGE_2
    },
    "SeanLMH/so100_picknplace_v2": {
        "observation.images.overhead": OBS_IMAGE,
        "observation.images.front": OBS_IMAGE_2
    },
    "pepijn223/yellow_lego_in_box1": {
        "observation.images.phone": OBS_IMAGE
    },
    "DimiSch/so100_50ep_2": {
        "observation.images.realsense": OBS_IMAGE
    },
    "DimiSch/so100_50ep_3": {
        "observation.images.realsense": OBS_IMAGE
    },
    "SeanLMH/so100_picknplace": {
        "observation.images.overhead": OBS_IMAGE,
        "observation.images.front": OBS_IMAGE_2
    },
    "nbaron99/so100_pick_and_place2": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "chmadran/so100_dataset08": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "vaishanthr/toy_pickplace_50ep": {
        "observation.images.webcam": OBS_IMAGE,
        "observation.images.gipper_cam": OBS_IMAGE_2
    },
    "Beegbrain/pick_place_green_block_lr": {
        "observation.images.realsense_side": OBS_IMAGE_2,
        "observation.images.realsense_top": OBS_IMAGE
    },
    "Ityl/so100_recording1": {
        "observation.images.realsense_side": OBS_IMAGE_2,
        "observation.images.realsense_top": OBS_IMAGE
    },
    "vaishanthr/toy_pickplace": {
        "observation.images.webcam": OBS_IMAGE,
        "observation.images.gipper_cam": OBS_IMAGE_2
    },
    "ad330/so100_box_pickPlace": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "Beegbrain/so100_put_cube_cup": {
        "observation.images.realsense_side": OBS_IMAGE_2,
        "observation.images.realsense_top": OBS_IMAGE
    },
    "aractingi/push_green_cube_hf": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "aractingi/push_green_cube_hf_cropped_resized": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "carpit680/giraffe_task": {
        "observation.images.webcam": OBS_IMAGE
    },
    "carpit680/giraffe_sock_demo_1": {
        "observation.images.webcam": OBS_IMAGE
    },
    "DimiSch/so100_terra_50_2": {
        "observation.images.realsense": OBS_IMAGE
    },
    "carpit680/giraffe_sock_demo_2": {
        "observation.images.webcam": OBS_IMAGE
    },
    "aractingi/push_cube_to_face_reward": {
        "observation.images.front": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2
    },
    "aractingi/push_cube_to_face_reward_cropped_resized": {
        "observation.images.front": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2
    },
    "aractingi/push_cube_reward_data": {
        "observation.images.front": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2
    },
    "aractingi/push_cube_reward_data_cropped_resized": {
        "observation.images.front": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2
    },
    "aractingi/push_cube_offline_data_cropped_resized": {
        "observation.images.front": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2
    },
    "aractingi/push_cube_front_side_reward": {
        "observation.images.front": OBS_IMAGE,
        "observation.images.side": OBS_IMAGE_2
    },
    "aractingi/push_cube_front_side_reward_cropped_resized": {
        "observation.images.front": OBS_IMAGE,
        "observation.images.side": OBS_IMAGE_2
    },
    "aractingi/push_cube_front_side_reward_long": {
        "observation.images.front": OBS_IMAGE,
        "observation.images.side": OBS_IMAGE_2
    },
    "aractingi/push_cube_front_side_reward_long_cropped_resized": {
        "observation.images.front": OBS_IMAGE,
        "observation.images.side": OBS_IMAGE_2
    },
    "aractingi/push_cube_reward": {
        "observation.images.front": OBS_IMAGE,
        "observation.images.side": OBS_IMAGE_2
    },
    "aractingi/push_cube_reward_cropped_resized": {
        "observation.images.front": OBS_IMAGE,
        "observation.images.side": OBS_IMAGE_2
    },
    "aractingi/push_cube_square_reward_cropped_resized": {
        "observation.images.front": OBS_IMAGE,
        "observation.images.side": OBS_IMAGE_2
    },
    "aractingi/push_cube_square_reward_1": {
        "observation.images.front": OBS_IMAGE,
        "observation.images.side": OBS_IMAGE_2
    },
    "aractingi/push_cube_square_reward_1_cropped_resized": {
        "observation.images.front": OBS_IMAGE,
        "observation.images.side": OBS_IMAGE_2
    },
    "aractingi/push_cube_square_light_reward": {
        "observation.images.front": OBS_IMAGE,
        "observation.images.side": OBS_IMAGE_2
    },
    "aractingi/push_cube_square_light_offline_demo": {
        "observation.images.front": OBS_IMAGE,
        "observation.images.side": OBS_IMAGE_2
    },
    "aractingi/push_cube_square_light_offline_demo_cropped_resized": {
        "observation.images.front": OBS_IMAGE,
        "observation.images.side": OBS_IMAGE_2
    },
    "denghj/dataset_red_tape01": {
        "observation.images.laptop": OBS_IMAGE
    },
    "aractingi/push_cube_square_offline_demo": {
        "observation.images.front": OBS_IMAGE,
        "observation.images.side": OBS_IMAGE_2
    },
    "aractingi/push_cube_square_offline_demo_cropped_resized": {
        "observation.images.front": OBS_IMAGE,
        "observation.images.side": OBS_IMAGE_2
    },
    "Beegbrain/stack_two_cubes": {
        "observation.images.realsense_side": OBS_IMAGE_2,
        "observation.images.realsense_top": OBS_IMAGE
    },
    "FeiYjf/Test_NNNN": {
        "observation.images.left": OBS_IMAGE,
        "observation.images.right": OBS_IMAGE_2
    },
    "LegrandFrederic/Orange-brick-lower-resolution": {
        "observation.images.main.left": OBS_IMAGE,
        "observation.images.main.right": OBS_IMAGE_2,
        "observation.images.secondary_0": OBS_IMAGE_3
    },
    "aractingi/pick_place_lego_cube_cropped_resized": {
        "observation.images.wrist": OBS_IMAGE,
        "observation.images.top": OBS_IMAGE_2
    },
    "aractingi/push_cube_overfit": {
        "observation.images.front": OBS_IMAGE,
        "observation.images.side": OBS_IMAGE_2
    },
    "aractingi/push_cube_overfit_cropped_resized": {
        "observation.images.front": OBS_IMAGE,
        "observation.images.side": OBS_IMAGE_2
    },
    "HITHY/so100_peach": {
        "observation.images.laptop": OBS_IMAGE
    },
    "zaringleb/so100_cube_2": {
        "observation.images.cam_high": OBS_IMAGE
    },
    "andreasBihlmaier/dual_arm_transfer_2025_02_16": {
        "observation.images.webcam_1": OBS_IMAGE_3,
        "observation.images.webcam_2": OBS_IMAGE_2,
        "observation.images.webcam_3": OBS_IMAGE
    },
    "zaringleb/so100_cube_4_binary": {
        "observation.images.cam_high": OBS_IMAGE
    },
    "1g0rrr/reward_pickplace1": {
        "observation.images.laptop": OBS_IMAGE
    },
    "1g0rrr/reward_pickplace1_cropped_resized": {
        "observation.images.laptop": OBS_IMAGE
    },
    "FeiYjf/Hold_Pieces": {
        "observation.images.left": OBS_IMAGE,
        "observation.images.right": OBS_IMAGE_2
    },
    "FeiYjf/Grab_Pieces": {
        "observation.images.left": OBS_IMAGE,
        "observation.images.right": OBS_IMAGE_2
    },
    "hegdearyandev/so100_eraser_cup_v1": {
        "observation.images.webcam-0": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "jbraumann/so100_1902": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "liyitenga/so100_pick_taffy10": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.left": OBS_IMAGE_3,
        "observation.images.gripper": OBS_IMAGE_2
    },
    "mikechambers/block_cup_5": {
        "observation.images.main_cam": OBS_IMAGE,
        "observation.images.secondary_cam": OBS_IMAGE_2
    },
    "zaringleb/so100_cube_5_linear": {
        "observation.images.cam_high": OBS_IMAGE
    },
    "yuz1wan/so100_pickplace_0223_2": {
        "observation.images.side": OBS_IMAGE
    },
    "yuz1wan/so100_pickplace_0223_3": {
        "observation.images.side": OBS_IMAGE
    },
    "samsam0510/mj_data_temp": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "samsam0510/tape_insert_1": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "samsam0510/tape_insert_2": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "pengjunkun/so100_push_to_hole": {
        "observation.images.laptop": OBS_IMAGE
    },
    "Deason11/Random_Kitchen": {
        "observation.images.L_OverheadCamera": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2,
        "observation.images.R_OverheadCamera": OBS_IMAGE_3
    },
    "1g0rrr/reward_dataset_name2": {
        "observation.images.side": OBS_IMAGE_2,
        "observation.images.front": OBS_IMAGE
    },
    "1g0rrr/reward_dataset_name2_cropped_resized": {
        "observation.images.side": OBS_IMAGE_2,
        "observation.images.front": OBS_IMAGE
    },
    "1g0rrr/offline_dataset_name2": {
        "observation.images.side": OBS_IMAGE_2,
        "observation.images.front": OBS_IMAGE
    },
    "1g0rrr/offline_dataset_name2_cropped_resized": {
        "observation.images.side": OBS_IMAGE_2,
        "observation.images.front": OBS_IMAGE
    },
    "aractingi/push_cube_simp_cropped_resized": {
        "observation.images.front": OBS_IMAGE,
        "observation.images.side": OBS_IMAGE_2
    },
    "danielkr452/so100_work6": {
        "observation.images.laptop": None,
        "observation.images.phone": OBS_IMAGE_2
    },
    "Loki0929/so100_100": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "yuz1wan/so100_fold_0227_1": {
        "observation.images.side": OBS_IMAGE
    },
    "yuz1wan/so100_fold_0227_2": {
        "observation.images.side": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2
    },
    "speedyyoshi/so100_grasp_pink_block": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "lirislab/stack_two_red_cubes": {
        "observation.images.realsense_side": OBS_IMAGE_2,
        "observation.images.realsense_top": OBS_IMAGE
    },
    "lirislab/red_cube_into_mug": {
        "observation.images.realsense_side": OBS_IMAGE_2,
        "observation.images.realsense_top": OBS_IMAGE
    },
    "lirislab/green_lego_block_into_mug": {
        "observation.images.realsense_side": OBS_IMAGE_2,
        "observation.images.realsense_top": OBS_IMAGE
    },
    "lirislab/green_lego_block_into_mug_easy": {
        "observation.images.realsense_side": OBS_IMAGE_2,
        "observation.images.realsense_top": OBS_IMAGE
    },
    "kevin510/lerobot-cat-toy-placement": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "NONHUMAN-RESEARCH/SOARM100_TASK_VENDA_BOX": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },

    "wangjl1512/pour_water": {
        "observation.images.laptop": OBS_IMAGE
    },
    "airthebear/so100_GL": {
        "observation.images.laptop": OBS_IMAGE
    },
    "zijian2022/noticehuman1": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "zijian2022/noticehuman2": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "kantine/so100_kapla_tower6": {
        "observation.images.logitech_1": OBS_IMAGE,
        "observation.images.logitech_2": OBS_IMAGE_2
    },
    "zijian2022/noticehuman5": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "zijian2022/llm40": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "Ashton3/lerobot-aloha": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "zijian2022/noticehuman50": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "AaronNewman/screwdriver_task_batch1": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "AaronNewman/screwdriver_task_batch2": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "AaronNewman/screwdriver_task_batch3": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "zijian2022/noticehuman60": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "zijian2022/noticehuman70": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "Bartm3/tape_to_bin": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "liuhuanjim013/so100_th_1": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.hand": OBS_IMAGE_2
    },
    "Pi-robot/barbecue_flip": {
        "observation.images.top": OBS_IMAGE_2,
        "observation.images.wrist": OBS_IMAGE
    },
    "Pi-robot/barbecue_put": {
        "observation.images.top": OBS_IMAGE_2,
        "observation.images.wrist": OBS_IMAGE
    },
    "wangjl1512/doll": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2
    },
    "sshh11/so100_orange_50ep_1": {
        "observation.images.laptop": OBS_IMAGE
    },
    "sshh11/so100_orange_50ep_2": {
        "observation.images.laptop": OBS_IMAGE
    },
    "DorayakiLin/so100_pick_cube_in_box": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "Bartm3/tape_to_bin2": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "luke250305/play_dice_250311.1": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "andy309/so100_0311_1152": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.back": OBS_IMAGE_2,
        "observation.images.wrist_right": OBS_IMAGE_3
    },
    "sihyun77/suho_so100": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "sihyun77/si_so100": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "shreyasgite/so100_base_left": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "sihyun77/suho_red": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "liuhuanjim013/so100_block": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.hand": OBS_IMAGE_2
    },
    "andy309/so100_0313_no_wrist_camera": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.back": OBS_IMAGE_2
    },
    "zijian2022/l9": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    }, 
    ## Annotated by Mustafa
    "zijian2022/n1_2": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "DorayakiLin/so100_stack_cube": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "andy309/so100_0313_no_wrist_camera_with_two_arms_cloths": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.back": OBS_IMAGE_2
    },
    "joaoocruz00/so100_makeitD1": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.realsensergb": OBS_IMAGE
    },
    "zijian2022/l10_1": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "zijian2022/l10_5": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "sihyun77/suho_red2": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "sihyun77/suho_angel": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "sihyun77/sihyun_king": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "acrampette/third_arm_01": {
        "observation.images.wrist": OBS_IMAGE_2
    },
    "Winster/so100_cube": {
        "observation.images.laptop": OBS_IMAGE
    },
    "1g0rrr/sam_openpi03": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "thedevansh/mar16_1336": {
        "observation.images.laptop": OBS_IMAGE
    },
    "hkphoooey/throw_stuffie": {
        "observation.images.phone": OBS_IMAGE
    },
    "doujiangwang/task1_10epi_100000step": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "sihyun77/sihyun_3_17_1": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "acrampette/third_arm_02": {
        "observation.images.wrist": OBS_IMAGE_2
    },
    "imsyed00/so100_yellowbowl_pickplace_1": {
        "observation.images.laptop": OBS_IMAGE
    },
    "kumarhans/so100_tape_task": {
        "observation.images.laptop1": OBS_IMAGE,
        "observation.images.laptop2": OBS_IMAGE_2
    },
    "sihyun77/sihyun_main": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "doujiangwang/task2_10epi_100000step": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "kantine/industrial_robothon_buttons_expert": {
        "observation.images.logitech_1": OBS_IMAGE_2,
        "observation.images.logitech_2": OBS_IMAGE
    },
    "kantine/industrial_robothon_buttons_anomaly": {
        "observation.images.logitech_1": OBS_IMAGE_2,
        "observation.images.logitech_2": OBS_IMAGE
    },
    "kantine/industrial_robothon_hatchAndProbe_expert": {
        "observation.images.logitech_1": OBS_IMAGE_2,
        "observation.images.logitech_2": OBS_IMAGE
    },
    "kantine/industrial_robothon_hatchAndProbe_anomaly": {
        "observation.images.logitech_1": OBS_IMAGE_2,
        "observation.images.logitech_2": OBS_IMAGE
    },
    "Odog16/so100_tea_towel_folding_v1": {
        "observation.images.workspace": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2
    },
    "zijian2022/so100_318": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "zijian2022/so100_318_1": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "Congying1112/so100_place_blue_bottle_with_two_cameras": {
        "observation.images.onhand": OBS_IMAGE_2,
        "observation.images.top": OBS_IMAGE
    },
    "Congying1112/so100_place_blue_bottle_with_two_cameras2": {
        "observation.images.onhand": OBS_IMAGE_2,
        "observation.images.top": OBS_IMAGE
    },
    "Congying1112/so100_place_blue_bottle_with_single_camera": {
        "observation.images.onhand": OBS_IMAGE_2
    },
    "pietroom/first_task_short": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "kantine/industrial_screws_sorting_expert": {
        "observation.images.logitech_1": OBS_IMAGE,
        "observation.images.logitech_2": OBS_IMAGE_2
    },
    "kantine/industrial_screws_sorting_anomaly": {
        "observation.images.logitech_1": OBS_IMAGE_2,
        "observation.images.logitech_2": OBS_IMAGE
    },
    "pietroom/second_task": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "zijian2022/c0": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "doujiangwang/task4_10epi_100000step": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "Congying1112/so100_switch_with_onhand_camera": {
        "observation.images.onhand": OBS_IMAGE_2
    },
    "HYAIYN/so100_get_orange_10epi": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "doujiangwang/task5_10epi_100000step": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "1g0rrr/sam_openpi_cube_low10": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "1g0rrr/sam_openpi_cube_top10": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "1g0rrr/sam_openpi_wire10": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "1g0rrr/sam_openpi_solder1": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "1g0rrr/sam_openpi_solder2": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    # "cranberrysoft/so100": {
    #     "observation.images.laptop": OBS_IMAGE,
    #     "observation.images.phone": OBS_IMAGE_2
    # },
    "wcode/so100_put_pen_50": {
        "observation.images.hand": OBS_IMAGE_2,
        "observation.images.top": OBS_IMAGE
    },
    "jchun/so100_pickplace_small_20250322_193929": {
        "observation.images.main": OBS_IMAGE_3,
        "observation.images.cv": OBS_IMAGE_2,
        "observation.images.webcam": OBS_IMAGE
    },
    "bnarin/so100_tic_tac_toe_we_do_it_live": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "dc2ac/so100-t5": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "chmadran/so100_home_dataset": {
        "observation.images.logitech": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2,
        "observation.images.laptop": OBS_IMAGE_3
    },
    "baladhurgesh97/so100_final_picking_3": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "bnarin/so100_tic_tac_toe_move_0_0": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "bnarin/so100_tic_tac_toe_move_1_0": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "bnarin/so100_tic_tac_toe_move_2_1": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "bnarin/so100_tic_tac_toe_move_4_0": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "zaringleb/so100_cube_6_2d": {
        "observation.images.cam_high": OBS_IMAGE
    },
    "andlyu/so100_indoor_0": {
        "observation.images.arm_left": OBS_IMAGE_2,
        "observation.images.arm_right": OBS_IMAGE_3,
        "observation.images.base_left": None,
        "observation.images.base_right": OBS_IMAGE
    },
    "andlyu/so100_indoor_2": {
        "observation.images.arm_left": OBS_IMAGE_2,
        "observation.images.arm_right": OBS_IMAGE_3,
        "observation.images.base_left": None,
        "observation.images.base_right": OBS_IMAGE
    },
    "Winster/so100_sim": {
        "observation.images.laptop": OBS_IMAGE
    },
    "badwolf256/so100_twin_cam_duck": {
        "observation.images.realsense": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2
    },
    "Congying1112/so100_simplepick_with_2_cameras_from_top": {
        "observation.images.onhand": OBS_IMAGE_2,
        "observation.images.top": OBS_IMAGE
    },
    "andlyu/so100_indoor_4": {
        "observation.images.arm_left": OBS_IMAGE_2,
        "observation.images.arm_right": OBS_IMAGE_3,
        "observation.images.base_left": None,
        "observation.images.base_right": OBS_IMAGE
    },
    "Zak-Y/so100_grap_dataset": {
        "observation.images.Logic_camera": OBS_IMAGE,
        "observation.images.Left_follower": OBS_IMAGE_2,
        "observation.images.Right_follower": OBS_IMAGE_3
    },
    "kantine/domotic_pouringCoffee_expert": {
        "observation.images.logitech_1": OBS_IMAGE,
        "observation.images.logitech_2": OBS_IMAGE_2
    },
    "kantine/domotic_pouringCoffee_anomaly": {
        "observation.images.logitech_1": OBS_IMAGE,
        "observation.images.logitech_2": OBS_IMAGE_2
    },
    "lucasngoo/so100_strawberry_grape": {
        "observation.images.webcam": OBS_IMAGE
    },
    "kantine/domotic_makingCoffee_expert": {
        "observation.images.logitech_1": OBS_IMAGE,
        "observation.images.logitech_2": OBS_IMAGE_2
    },
    "kantine/domotic_makingCoffee_anomaly": {
        "observation.images.logitech_1": OBS_IMAGE,
        "observation.images.logitech_2": OBS_IMAGE_2
    },
    "ZGGZZG/so100_drop1": {
        "observation.images.left": OBS_IMAGE_2,
        "observation.images.up": OBS_IMAGE
    },
    "kantine/industrial_soldering_expert": {
        "observation.images.logitech_1": OBS_IMAGE,
        "observation.images.logitech_2": OBS_IMAGE_2
    },
    "kantine/industrial_soldering_anomaly": {
        "observation.images.logitech_1": OBS_IMAGE,
        "observation.images.logitech_2": OBS_IMAGE_2
    },
    "kantine/domotic_dishTidyUp_expert": {
        "observation.images.logitech_1": OBS_IMAGE,
        "observation.images.logitech_2": OBS_IMAGE_2
    },
    "kantine/domotic_dishTidyUp_anomaly": {
        "observation.images.logitech_1": OBS_IMAGE,
        "observation.images.logitech_2": OBS_IMAGE_2
    },
    "kantine/domotic_groceriesSorting_expert": {
        "observation.images.logitech_1": OBS_IMAGE,
        "observation.images.logitech_2": OBS_IMAGE_2
    },
    "kantine/domotic_groceriesSorting_anomaly": {
        "observation.images.logitech_1": OBS_IMAGE,
        "observation.images.logitech_2": OBS_IMAGE_2
    },
    "badwolf256/so100_twin_cam_duck_v2": {
        "observation.images.realsense": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2
    },
    # "andlyu/so100_indoor_5": {
    #     "observation.images.arm_left": OBS_IMAGE,
    #     "observation.images.arm_right": OBS_IMAGE_2,
    #     "observation.images.base_left": OBS_IMAGE_3,
    #     "observation.images.base_right": None
    # },
    "kantine/domotic_vegetagblesAndFruitsSorting_expert": {
        "observation.images.logitech_1": OBS_IMAGE,
        "observation.images.logitech_2": OBS_IMAGE_2
    },
    "kantine/domotic_vegetagblesAndFruitsSorting_anomaly": {
        "observation.images.logitech_1": OBS_IMAGE,
        "observation.images.logitech_2": OBS_IMAGE_2
    },
    "kantine/domotic_setTheTable_expert": {
        "observation.images.logitech_1": OBS_IMAGE,
        "observation.images.logitech_2": OBS_IMAGE_2
    },
    "kantine/domotic_setTheTable_anomaly": {
        "observation.images.logitech_1": OBS_IMAGE,
        "observation.images.logitech_2": OBS_IMAGE_2
    },
    "therarelab/so100_pick_place": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "abhisb/so100_51_ep": {
        "observation.images.front": OBS_IMAGE_3,
        "observation.images.overhead": OBS_IMAGE,
        "observation.images.mobile": OBS_IMAGE_2
    },
    "andlyu/so100_indoor_val_0": {
        "observation.images.arm": OBS_IMAGE,
        "observation.images.gripper": OBS_IMAGE_2,
        "observation.images.base": OBS_IMAGE_3
    },
    "allenchienxxx/so100Test": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "lizi178119985/so100_jia": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "badwolf256/so100_twin_cam_duck_v3": {
        "observation.images.realsense": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2
    },
    "andrewcole712/so100_tape_bin_place": {
        "observation.images.phone": OBS_IMAGE
    },
    "Gano007/so100_lolo": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "Zak-Y/so100_three_cameras_dataset": {
        "observation.images.Left_follower": OBS_IMAGE_2,
        "observation.images.Right_follower": OBS_IMAGE_2,
        "observation.images.Logic_camera": OBS_IMAGE
    },
    "Gano007/so100_doliprane": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "XXRRSSRR/so100_v3_num_episodes_50": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "zijian2022/assemblyarm2": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "ganker5/so100_action_20250403": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "andlyu/so100_indoor_val2": {
        "observation.images.base_right": None,
        "observation.images.arm": OBS_IMAGE,
        "observation.images.gripper": OBS_IMAGE_2,
        "observation.images.base": OBS_IMAGE_3
    },
    "Gano007/so100_gano": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "paszea/so100_whale_grab": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "paszea/so100_whale": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "Clementppr/lerobot_pick_and_place_dataset_world_model": {
        "observation.images.laptop": OBS_IMAGE
    },
    "andlyu/so100_indoor_10": {
        "observation.images.base_right": None,
        "observation.images.arm": OBS_IMAGE,
        "observation.images.gripper": OBS_IMAGE_2,
        "observation.images.base": OBS_IMAGE_3
    },
    "RasmusP/so100_dataset50ep_a": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "RasmusP/so100_dataset50ep": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "Gano007/so100_second": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "zaringleb/so100_cude_linear_and_2d_comb": {
        "observation.images.cam_high": OBS_IMAGE
    },
    "dsfsg/grasp_pens": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2,
        "observation.images.phone2": OBS_IMAGE_3
    },
    "zijian2022/digitalfix": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "zijian2022/digitalfix2": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "zijian2022/digitalfix3": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "T1g3rGE/so100_pickplace_small_20250407_171912": {
        "observation.images.webcam": OBS_IMAGE_2,
    },
    "sihyun77/mond_13": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "abokinala/sputnik_100_11_pick_place_container": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "dsfsg/bring_bottle": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2,
        "observation.images.phone2": OBS_IMAGE_3
    },
    # "duthvik/sputnik_100_13_pick_place_container": {
    #     "observation.images.laptop": OBS_IMAGE,
    #     "observation.images.phone": OBS_IMAGE_2
    # },
    "abokinala/sputnik_100_12_pick_place_container": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "Mwuqiu/so100_0408": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "AK51/4090_01": {
        "observation.images.left_arm_cam": OBS_IMAGE_2,
        "observation.images.base_cam": OBS_IMAGE_3,
        "observation.images.top_cam": OBS_IMAGE,
        "observation.images.right_arm_cam": None
    },
    "356c/so100_rope_reposition_1": {
        "observation.images.side": OBS_IMAGE_3,
        "observation.images.overhead": OBS_IMAGE,
        "observation.images.body": OBS_IMAGE_2
    },
    "paszea/so100_lego_mix": {
        "observation.images.phone": OBS_IMAGE
    },
    "abokinala/sputnik_100_14_pick_place_container": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "abokinala/sputnik_100_23_pick_place_surface": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "jiajun001/eraser00_2": {
        "observation.images.side": OBS_IMAGE,
        "observation.images.hand": OBS_IMAGE_2
    },
    "jlesein/TestBoulon2": {
        "observation.images.robor": OBS_IMAGE_2,
        "observation.images.top": OBS_IMAGE
    },
    "yskim2025/unitylerobot": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "duthvik/sputnik_100_31_pour_liquid": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "duthvik/sputnik_100_24_pick_place_surface": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "duthvik/sputnik_100_25_pick_place_surface": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "duthvik/sputnik_100_17_pick_place_container": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "duthvik/sputnik_100_26_pick_place_surface": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "isadev/bougies1": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "danaaubakirova/svla_so100_task4_v3_clean": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2
    },
    "danaaubakirova/svla_so100_task5_v3_clean": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2
    },
    "danaaubakirova/svla_so100_task4_v3": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2
    },
    "danaaubakirova/svla_so100_task5_v3": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2
    },
    "danaaubakirova/so100_task_1": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2
    },
    "danaaubakirova/so100_task_2": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2
    },
    "danaaubakirova/so100_task_3": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2
    },
    "danaaubakirova/so100_task_4": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2
    },
    "sixpigs1/so100_pick_cube_in_box_error": {
        "observation.images.above": OBS_IMAGE,
        "observation.images.rightfront": OBS_IMAGE_2
    },
    "sixpigs1/so100_push_cube_error": {
        "observation.images.above": OBS_IMAGE,
        "observation.images.rightfront": OBS_IMAGE_2
    },
    "sixpigs1/so100_pull_cube_error": {
        "observation.images.above": OBS_IMAGE,
        "observation.images.rightfront": OBS_IMAGE_2
    },
    "isadev/bougies2": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "therarelab/med_dis_rare_6": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "duthvik/sputnik_100_27_pick_place_surface": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "zijian2022/closer3": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "duthvik/sputnik_100_41_custom_tasks": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.usb_front": OBS_IMAGE_2,
        "observation.images.side_view": OBS_IMAGE_3
    },
    "duthvik/sputnik_100_42_custom_tasks": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.usb_front": OBS_IMAGE_2,
        "observation.images.side_view": OBS_IMAGE_3
    },
    "duthvik/sputnik_100_43_custom_tasks": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.usb_front": OBS_IMAGE_2,
        "observation.images.side_view": OBS_IMAGE_3
    },
    "duthvik/sputnik_100_44_custom_tasks": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.usb_front": OBS_IMAGE_2,
        "observation.images.side_view": OBS_IMAGE_3
    },
    "duthvik/sputnik_100_51_kitchen_tasks": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.usb_front": OBS_IMAGE_2,
        "observation.images.side_view": OBS_IMAGE_3
    },
    "duthvik/sputnik_100_52_kitchen_tasks": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.usb_front": OBS_IMAGE_2,
        "observation.images.side_view": OBS_IMAGE_3
    },
    "duthvik/sputnik_100_53_kitchen_tasks": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.usb_front": OBS_IMAGE_2,
        "observation.images.side_view": OBS_IMAGE_3
    },
    "duthvik/sputnik_100_45_custom_tasks": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.usb_front": OBS_IMAGE_2,
        "observation.images.side_view": OBS_IMAGE_3
    },
    "duthvik/sputnik_100_32_pour_liquid": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.usb_front": OBS_IMAGE_2,
        "observation.images.side_view": OBS_IMAGE_3
    },
    "duthvik/sputnik_100_29_pick_place_surface": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.usb_front": OBS_IMAGE_2,
        "observation.images.side_view": OBS_IMAGE_3
    },
    "duthvik/sputnik_100_18_pick_place_container": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.usb_front": OBS_IMAGE_2,
        "observation.images.side_view": OBS_IMAGE_3
    },
    "sixpigs1/so100_pull_cube_by_tool_error": {
        "observation.images.above": OBS_IMAGE_2,
        "observation.images.rightfront": OBS_IMAGE
    },
    "sixpigs1/so100_insert_cylinder_error": {
        "observation.images.above": OBS_IMAGE_2,
        "observation.images.rightfront": OBS_IMAGE
    },
    "abokinala/sputnik_100_54_kitchen_tasks": {
        "observation.images.laptop": None,
        "observation.images.phone": OBS_IMAGE
    },
    "abokinala/sputnik_100_55_kitchen_tasks": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "m1b/so100_bluelego": {
        "observation.images.side": OBS_IMAGE,
        "observation.images.phone": None
    },
    "abokinala/sputnik_100_46_custom_tasks": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "m1b/so100_bluelego_updt": {
        "observation.images.side": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "kantine/flip_A0": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "kantine/flip_A1": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "kantine/flip_A2": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "kantine/flip_A3": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "lirislab/guess_who_no_cond": {
        "observation.images.mounted": OBS_IMAGE_2
    },
    "kantine/flip_A4": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "kantine/flip_A5": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "lirislab/guess_who_lighting": {
        "observation.images.mounted": OBS_IMAGE_2
    },
    "nguyen-v/so100_press_red_button": {
        "observation.images.back": OBS_IMAGE_3,
        "observation.images.top": OBS_IMAGE,
        "observation.images.left": OBS_IMAGE_2,
        "observation.images.right": None
    },
    "nguyen-v/so100_bimanual_grab_lemon_put_in_box2": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.left": OBS_IMAGE_2,
        "observation.images.right": OBS_IMAGE_3
    },
    "pierfabre/cow": {
        "observation.images.robot": OBS_IMAGE_2,
        "observation.images.webcam": OBS_IMAGE
    },
    "nguyen-v/press_red_button_new": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.left": OBS_IMAGE_2,
        "observation.images.right": OBS_IMAGE_3
    },
    "nguyen-v/so100_rotate_red_button": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.left": OBS_IMAGE_2,
        "observation.images.right": OBS_IMAGE_3
    },
    # "raghav-katta-1/lerobot2": {
    #     "observation.images.cam": OBS_IMAGE
    # },

    "Cidoyi/so100_all_notes": {
        "observation.images.keyboard_camera": OBS_IMAGE
    },
    "roboticshack/team10-red-block": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "Cidoyi/so100_all_notes_1": {
        "observation.images.keyboard_camera": OBS_IMAGE
    },
    "roboticshack/team_5-QuiEstCe_everyBox": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "roboticshack/team11_pianobot": {
        "observation.images.keyboard_camera": OBS_IMAGE
    },
    "roboticshack/team2-guess_who_so100": {
        "observation.images.mounted": OBS_IMAGE
    },
    "roboticshack/team2-guess_who_so100_light": {
        "observation.images.mounted": OBS_IMAGE_2
    },
    "roboticshack/team2-guess_who_so100_edge_case": {
        "observation.images.mounted": OBS_IMAGE_2
    },
    "roboticshack/team2-guess_who_less_ligth": {
        "observation.images.mounted": OBS_IMAGE_2
    },
    "Cidoyi/so100_all_notes_3": {
        "observation.images.keyboard_camera": OBS_IMAGE
    },
    "dsfsg/grasp_pen_and_bottle": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_3,
        "observation.images.phone2": OBS_IMAGE_2
    },
    "abokinala/sputnik_100_60_kitchen_tasks": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "abokinala/sputnik_100_58_kitchen_tasks": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "danaaubakirova/so100_v2_task_1": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2
    },
    "danaaubakirova/so100_v2_task_2": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2
    },
    "danaaubakirova/so100_v2_task_3": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2
    },
    "danaaubakirova/so100_v2_task_4": {
        "observation.images.top": OBS_IMAGE,
        "observation.images.wrist": OBS_IMAGE_2
    },
    "zijian2022/force1": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "zijian2022/force2": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "zijian2022/force3": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "jiajun001/eraser00_3": {
        "observation.images.side": OBS_IMAGE_3,
        "observation.images.hand": OBS_IMAGE_2,
        "observation.images.front": OBS_IMAGE
    },
    "zijian2022/bi2": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "zijian2022/bi1": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "zijian2022/hand1": {
        "observation.images.laptop": OBS_IMAGE,
        "observation.images.phone": OBS_IMAGE_2
    },
    "Setchii/so100_grab_ball": {
        "observation.images.laptop": OBS_IMAGE_2,
        "observation.images.phone": OBS_IMAGE
    },
    "MossProphet/so100_square-1-2-3.2": {
        "observation.images.External": OBS_IMAGE_3,
        "observation.images.Arm_left": OBS_IMAGE_2,
        "observation.images.Arm_right": None
    },
    "Yotofu/so100_sweeper_shoes": {
        "observation.images.front_rgb": OBS_IMAGE,
        "observation.images.front_depth": OBS_IMAGE_2,
        "observation.images.end_rgb": OBS_IMAGE_3,
        "observation.images.end_depth": None
    },
    "VoicAndrei/so100_banana_to_plate_rebel_full": {
        "observation.images.right": OBS_IMAGE_2,
        "observation.images.top": OBS_IMAGE
    },
    # V4 datasets
    '1g0rrr/screw1': {'observation.images.laptop': OBS_IMAGE,
                    'observation.images.phone': OBS_IMAGE_2},
    'Allen-488/koch_dataset_50': {'observation.images.laptop': OBS_IMAGE,
                                'observation.images.phone': OBS_IMAGE_3},
    'Anas0711/Gr00t_lerobot_state_action': {'observation.images.image': OBS_IMAGE,
                                            'observation.images.wrist_image': OBS_IMAGE_2,
                                            'absolute_action': None,
                                    'annotation.human.action.task_description': None,
                                    'annotation.human.validity': None,'next.reward': None},
    'Anas0711/Gr00t_lerobot_state_action_180': {'observation.images.image': OBS_IMAGE,
                                                'observation.images.wrist_image': OBS_IMAGE_2,
                                                'absolute_action': None,
                                    'annotation.human.action.task_description': None,
                                    'annotation.human.validity': None,'next.reward': None},
    'Anas0711/Gr00t_lerobot_state_action_600': {'observation.images.image': OBS_IMAGE,
                                                'observation.images.wrist_image': OBS_IMAGE_2,
                                                'absolute_action': None,
                                    'annotation.human.action.task_description': None,
                                    'annotation.human.validity': None,'next.reward': None},
    'Anas0711/Gr00t_lerobot_state_action_60_continuous': {'observation.images.image': OBS_IMAGE,
                                                        'observation.images.wrist_image': OBS_IMAGE_2,
                                                        'absolute_action': None,
                                    'annotation.human.action.task_description': None,
                                    'annotation.human.validity': None,'next.reward': None},
    'Anas0711/Gr00t_lerobot_state_action_box_1': {'observation.images.image': OBS_IMAGE,
                                                'observation.images.wrist_image': OBS_IMAGE_2,
                                                'absolute_action': None,
                                    'annotation.human.action.task_description': None,
                                    'annotation.human.validity': None,'next.reward': None},
    'Anas0711/Gr00t_lerobot_state_action_box_2': {'observation.images.image': OBS_IMAGE,
                                                'observation.images.wrist_image': OBS_IMAGE_2,
                                                'absolute_action': None,
                                    'annotation.human.action.task_description': None,
                                    'annotation.human.validity': None,'next.reward': None},
    'Beegbrain/align_cubes_green_blue': {'observation.images.back': OBS_IMAGE_3,
                                        'observation.images.side': OBS_IMAGE},
    'Beegbrain/align_three_pens': {'observation.images.back': OBS_IMAGE_3,
                                    'observation.images.side': OBS_IMAGE},
    'Beegbrain/moss_close_drawer_teabox': {'observation.images.back': OBS_IMAGE_3,
                                            'observation.images.front': OBS_IMAGE},
    'Beegbrain/moss_open_drawer_teabox': {'observation.images.back': OBS_IMAGE_3,
                                        'observation.images.front': OBS_IMAGE},
    'Beegbrain/moss_put_cube_teabox': {'observation.images.back': OBS_IMAGE_3,
                                        'observation.images.front': OBS_IMAGE},
    'Beegbrain/moss_stack_cubes': {'observation.images.back': OBS_IMAGE_3,
                                    'observation.images.front': OBS_IMAGE},
    'Beegbrain/oc_stack_cubes': {'observation.images.realsense': OBS_IMAGE_3,
                                'observation.images.realsense_top': OBS_IMAGE},
    'Beegbrain/put_green_into_blue_bin': {'observation.images.back': OBS_IMAGE_3,
                                        'observation.images.side': OBS_IMAGE},
    'Beegbrain/put_red_triangle_green_rect': {'observation.images.back': OBS_IMAGE_3,
                                            'observation.images.side': OBS_IMAGE},
    'Beegbrain/put_screwdriver_box': {'observation.images.back': OBS_IMAGE_3,
                                    'observation.images.side': OBS_IMAGE},
    'Beegbrain/stack_2_cubes': {'observation.images.laptop': OBS_IMAGE},
    'Beegbrain/stack_green_on_blue_cube': {'observation.images.back': OBS_IMAGE_3,
                                            'observation.images.side': OBS_IMAGE},
    'BlobDieKatze/GrabBlocks': {'observation.images.laptop': OBS_IMAGE,
                                'observation.images.phone': OBS_IMAGE_2},
    'Deason11/mobile_manipulator_0319': {'observation.images.L_OverheadCamera': OBS_IMAGE,
                                        'observation.images.R_OverheadCamera': OBS_IMAGE_3,
                                        'observation.images.wrist': OBS_IMAGE_2},
    'Deason11/mobile_manipulator_0326': {'observation.images.L_OverheadCamera': OBS_IMAGE,
                                        'observation.images.R_OverheadCamera': OBS_IMAGE_3,
                                        'observation.images.wrist': OBS_IMAGE_2},
    'Dongkkka/cable_pick_and_place2': {'observation.images.laptop': OBS_IMAGE_2,
                                        'observation.images.phone': OBS_IMAGE},
    'Eyas/grab_bouillon': {'observation.images.laptop': OBS_IMAGE,
                            'observation.images.phone': OBS_IMAGE_3},
    'Eyas/grab_pink_lighter_10_per_loc': {'observation.images.laptop': OBS_IMAGE},
    'Gongsta/grasp_duck_in_cup': {'observation.images.laptop': OBS_IMAGE},
    'HWJ658970/fat_fish': {'observation.images.front': OBS_IMAGE,
                            'observation.images.phone': OBS_IMAGE_3},
    'HWJ658970/lego_100_class': {'observation.images.front': OBS_IMAGE,
                                'observation.images.phone': OBS_IMAGE_3},
    'HWJ658970/lego_50': {'observation.images.front': OBS_IMAGE,
                        'observation.images.phone': OBS_IMAGE_3},
    'HWJ658970/lego_50_camera_change': {'observation.images.front': OBS_IMAGE,
                                        'observation.images.phone': OBS_IMAGE_3},
    'HuaihaiLyu/groceries': {'observation.images.cam_high': OBS_IMAGE,
                            'observation.images.cam_left_wrist': OBS_IMAGE_2,
                            'observation.images.cam_right_wrist': OBS_IMAGE_3},
    'HuaihaiLyu/stackbasket': {'observation.images.cam_high': OBS_IMAGE,
                                'observation.images.cam_left_wrist': OBS_IMAGE_2,
                                'observation.images.cam_right_wrist': OBS_IMAGE_3},
    'IPEC-COMMUNITY/berkeley_mvp_lerobot': {'observation.images.hand_image': OBS_IMAGE},
    'IPEC-COMMUNITY/ucsd_kitchen_dataset_lerobot': {'observation.images.image': OBS_IMAGE},
    'JJwuj/koch_static_grasp_0402_v5': {'observation.images.E12': OBS_IMAGE_2,
                                        'observation.images.E22S': OBS_IMAGE},
    'KeWangRobotics/piper_push_cube_gamepad_1': {'observation.images.top': OBS_IMAGE,
                                                'observation.images.wrist': OBS_IMAGE_2},
    'KeWangRobotics/piper_push_cube_gamepad_1_cropped_resized': {'observation.images.top': OBS_IMAGE,
                                                                'observation.images.wrist': OBS_IMAGE_2},
    'KeWangRobotics/piper_push_cube_gamepad_2': {'observation.images.top': OBS_IMAGE,
                                                'observation.images.wrist': OBS_IMAGE_2},
    'KeWangRobotics/piper_push_cube_gamepad_2_cropped_resized': {'observation.images.top': OBS_IMAGE,
                                                                'observation.images.wrist': OBS_IMAGE_2},
    'KeWangRobotics/piper_rl_1': {'observation.images.top': OBS_IMAGE,
                                'observation.images.wrist': OBS_IMAGE_2},
    'KeWangRobotics/piper_rl_1_cropped_resized': {'observation.images.top': OBS_IMAGE,
                                                'observation.images.wrist': OBS_IMAGE_2},
    'KeWangRobotics/piper_rl_2': {'observation.images.top': OBS_IMAGE,
                                'observation.images.wrist': OBS_IMAGE_2},
    'KeWangRobotics/piper_rl_2_cropped_resized': {'observation.images.top': OBS_IMAGE,
                                                'observation.images.wrist': OBS_IMAGE_2},
    'KeWangRobotics/piper_rl_3': {'observation.images.top': OBS_IMAGE,
                                'observation.images.wrist': OBS_IMAGE_2},
    'KeWangRobotics/piper_rl_3_cropped_resized': {'observation.images.top': OBS_IMAGE,
                                                'observation.images.wrist': OBS_IMAGE_2},
    'KeWangRobotics/piper_rl_4': {'observation.images.top': OBS_IMAGE,
                                'observation.images.wrist': OBS_IMAGE_2},
    'KeWangRobotics/piper_rl_4_cropped_resized': {'observation.images.top': OBS_IMAGE,
                                                'observation.images.wrist': OBS_IMAGE_2},
    'Lugenbott/koch_1225': {'observation.images.Camera0': OBS_IMAGE,
                            'observation.images.Camera2': OBS_IMAGE_2},
    'RLWRLD/put_cube_sync': {'observation.images.cam_high': OBS_IMAGE,
                            'observation.images.cam_right_wrist': OBS_IMAGE_2},
    'T-K-233/koch_k1_pour_shot': {'observation.images.gripper': OBS_IMAGE_2,
                                'observation.images.side': OBS_IMAGE_3,
                                'observation.images.top': OBS_IMAGE},
    'TrossenRoboticsCommunity/aloha_ai_block': {'observation.images.cam_high': OBS_IMAGE,
                                                'observation.images.cam_left_wrist': OBS_IMAGE_2,
                                                'observation.images.cam_low': None,
                                                'observation.images.cam_right_wrist': OBS_IMAGE_3},
    'TrossenRoboticsCommunity/aloha_baseline_dataset': {'observation.images.cam_high': OBS_IMAGE,
                                                        'observation.images.cam_left_wrist': OBS_IMAGE_2,
                                                        'observation.images.cam_low': None,
                                                        'observation.images.cam_right_wrist': OBS_IMAGE_3},
    'TrossenRoboticsCommunity/aloha_fold_tshirt': {'observation.images.cam_left_wrist': OBS_IMAGE_2,
                                                    'observation.images.cam_low': OBS_IMAGE,
                                                    'observation.images.cam_right_wrist': OBS_IMAGE_3},
    'TrossenRoboticsCommunity/aloha_stationary_logo_assembly': {'observation.images.cam_high': OBS_IMAGE,
                                                                'observation.images.cam_left_wrist': OBS_IMAGE_2,
                                                                'observation.images.cam_low': OBS_IMAGE_3,
                                                                'observation.images.cam_right_wrist': None},
    # 'YoelChornton/moss_get_rope_15ep': {'observation.images.laptop': OBS_IMAGE,
    #                                     'observation.images.phone': OBS_IMAGE_2},
    # 'YoelChornton/moss_get_rope_20ep_2': {'observation.images.laptop': OBS_IMAGE,
    #                                     'observation.images.phone': OBS_IMAGE_2},
    # 'YoelChornton/moss_get_rope_20ep_new': {'observation.images.laptop': OBS_IMAGE,
    #                                         'observation.images.phone': OBS_IMAGE_2},
    # 'YoelChornton/moss_get_rope_20ep_old': {'observation.images.laptop': OBS_IMAGE,
    #                                         'observation.images.phone': OBS_IMAGE_2},
    # 'YoelChornton/moss_push_ball': {'observation.images.phone': OBS_IMAGE},
    # 'YoelChornton/moss_push_ball_absolute_joint': {'observation.images.phone': OBS_IMAGE},
    # 'YoelChornton/moss_push_ball_cropped_resized': {'observation.images.phone': OBS_IMAGE},
    # 'YoelChornton/moss_push_ball_relative_joint': {'observation.images.phone': OBS_IMAGE},
    # 'YoelChornton/moss_push_ball_reljoint_cropresized': {'observation.images.phone': OBS_IMAGE},
    # 'YoelChornton/moss_push_ball_reljoint_stop': {'observation.images.phone': OBS_IMAGE},
    # 'YoelChornton/moss_push_ball_reljoint_stop2': {'observation.images.phone': OBS_IMAGE},
    # 'YoelChornton/moss_push_ball_reljoint_stop2_cropped': {'observation.images.phone': OBS_IMAGE},
    # 'YoelChornton/moss_push_ball_reljoint_stop_cropped': {'observation.images.phone': OBS_IMAGE},
    'Yuanzhu/koch_bimanual_grasp_0': {'observation.images.top_camera': OBS_IMAGE},
    'Yuanzhu/koch_bimanual_grasp_3': {'observation.images.hand_camera': OBS_IMAGE_2,
                                    'observation.images.side_camera': OBS_IMAGE_3,
                                    'observation.images.top_camera': OBS_IMAGE},
    'Zhaoting123/koch_cleanDesk_': {'observation.images.phone': OBS_IMAGE},
    'abbyoneill/data_w_mug': {'observation.images.logitech1': OBS_IMAGE,
                            'observation.images.logitech2': OBS_IMAGE_2},
    'abbyoneill/new_dataset_pick_place': {'observation.images.logitech1': OBS_IMAGE,
                                        'observation.images.logitech2': OBS_IMAGE_3},
    'abbyoneill/pusht': {'observation.images.logitech1': OBS_IMAGE,
                        'observation.images.logitech2': OBS_IMAGE_3},
    'abbyoneill/thurs1120pickplace': {'observation.images.logitech1': OBS_IMAGE,
                                    'observation.images.logitech2': OBS_IMAGE_3},
    'abougdira/cube_target': {'observation.images.realsense': OBS_IMAGE_3,
                            'observation.images.realsense_top': OBS_IMAGE},
    'agonyxx/koch-aloha': {'observation.images.laptop': OBS_IMAGE,
                            'observation.images.phone': OBS_IMAGE_2,
                            'observation.images.phone2': OBS_IMAGE_3},
    'aliberts/koch_tutorial': {'observation.images.laptop': OBS_IMAGE,
                                'observation.images.phone': OBS_IMAGE_3},
    'andabi/D10': {'observation.images.bird': OBS_IMAGE,
                    'observation.images.wrist_left': OBS_IMAGE_2,
                    'observation.images.wrist_right': OBS_IMAGE_3},
    'andabi/D11': {'observation.images.bird': OBS_IMAGE,
                    'observation.images.wrist_left': OBS_IMAGE_2,
                    'observation.images.wrist_right': OBS_IMAGE_3},
    'andabi/D12': {'observation.images.bird': OBS_IMAGE,
                    'observation.images.wrist_left': OBS_IMAGE_2,
                    'observation.images.wrist_right': OBS_IMAGE_3},
    'andabi/D13': {'observation.images.bird': OBS_IMAGE,
                    'observation.images.wrist_left': OBS_IMAGE_2,
                    'observation.images.wrist_right': OBS_IMAGE_3},
    'andabi/D14': {'observation.images.bird': OBS_IMAGE,
                    'observation.images.wrist_left': OBS_IMAGE_2,
                    'observation.images.wrist_right': OBS_IMAGE_3},
    'andabi/D15': {'observation.images.bird': OBS_IMAGE,
                    'observation.images.wrist_left': OBS_IMAGE_2,
                    'observation.images.wrist_right': OBS_IMAGE_3},
    'andabi/D16': {'observation.images.bird': OBS_IMAGE,
                    'observation.images.wrist_left': OBS_IMAGE_2,
                    'observation.images.wrist_right': OBS_IMAGE_3},
    'andabi/D17': {'observation.images.bird': OBS_IMAGE,
                    'observation.images.wrist_left': OBS_IMAGE_2,
                    'observation.images.wrist_right': OBS_IMAGE_3},
    'andabi/D2': {'observation.images.bird': OBS_IMAGE,
                'observation.images.wrist_left': OBS_IMAGE_2,
                'observation.images.wrist_right': OBS_IMAGE_3},
    'andabi/D3': {'observation.images.bird': OBS_IMAGE,
                'observation.images.wrist_left': OBS_IMAGE_2,
                'observation.images.wrist_right': OBS_IMAGE_3},
    'andabi/D4': {'observation.images.bird': OBS_IMAGE,
                'observation.images.wrist_left': OBS_IMAGE_2,
                'observation.images.wrist_right': OBS_IMAGE_3},
    'andabi/D5': {'observation.images.bird': OBS_IMAGE,
                'observation.images.wrist_left': OBS_IMAGE_2,
                'observation.images.wrist_right': OBS_IMAGE_3},
    'andabi/D6': {'observation.images.bird': OBS_IMAGE,
                'observation.images.wrist_left': OBS_IMAGE_2,
                'observation.images.wrist_right': OBS_IMAGE_3},
    'andabi/D7': {'observation.images.bird': OBS_IMAGE,
                'observation.images.wrist_left': OBS_IMAGE_2,
                'observation.images.wrist_right': OBS_IMAGE_3},
    'andabi/D8': {'observation.images.bird': OBS_IMAGE,
                'observation.images.wrist_left': OBS_IMAGE_2,
                'observation.images.wrist_right': OBS_IMAGE_3},
    'andabi/D9': {'observation.images.bird': OBS_IMAGE,
                'observation.images.wrist_left': OBS_IMAGE_2,
                'observation.images.wrist_right': OBS_IMAGE_3},
    'andabi/a_shoe_easy_10': {'observation.images.bird': OBS_IMAGE,
                            'observation.images.wrist_left': OBS_IMAGE_2,
                            'observation.images.wrist_right': OBS_IMAGE_3},
    'andabi/shoes_easy': {'observation.images.bird': OBS_IMAGE,
                        'observation.images.wrist_left': OBS_IMAGE_2,
                        'observation.images.wrist_right': OBS_IMAGE_3},
    'arclabmit/Koch_twoarms': {'observation.images.laptop': OBS_IMAGE,
                                'observation.images.phone': OBS_IMAGE_3},
    'arclabmit/koch_gear_and_bin': {'observation.images.nexigo_webcam': OBS_IMAGE,
                                    'observation.images.realsense': OBS_IMAGE_3},
    'cmcgartoll/cube_color_organizer': {'observation.images.claw': OBS_IMAGE_2,
                                        'observation.images.front': OBS_IMAGE,
                                        'observation.images.phone': OBS_IMAGE_3},
    'ctbfl/sort_battery': {'observation.images.laptop': OBS_IMAGE_3,
                            'observation.images.phone': OBS_IMAGE,
                            'observation.images.wrist': OBS_IMAGE_2},
    'dboemer/koch_50-samples': {'observation.images.side': OBS_IMAGE_3,
                                'observation.images.top': OBS_IMAGE},
    'dkdltu1111/omx-bottle1': {'observation.images.laptop': OBS_IMAGE,
                                'observation.images.phone': OBS_IMAGE_2},
    'dop0/koch_pick_terminal': {'observation.images.side': OBS_IMAGE_3,
                                'observation.images.top': OBS_IMAGE,
                                'observation.images.wrist': OBS_IMAGE_2},
    'ellen2imagine/pusht_green1': {'observation.images.phone': OBS_IMAGE},
    'ellen2imagine/pusht_green_same_init2': {'observation.images.phone': OBS_IMAGE},
    'engineer0002/pepper': {'observation.images.left_wrist': OBS_IMAGE_2,
                            'observation.images.right_wrist': OBS_IMAGE_3,
                            'observation.images.top': OBS_IMAGE},
    'ethanCSL/your_task_name': {'observation.images.front': OBS_IMAGE,
                                'observation.images.phone': OBS_IMAGE_2},
    'hangwu/koch_pick_terminal': {'observation.images.side': OBS_IMAGE_3,
                                'observation.images.top': OBS_IMAGE,
                                'observation.images.wrist': OBS_IMAGE_2},
    'hangwu/piper_pick_terminal_2': {'observation.images.depth': None,
                                    'observation.images.realsense_color': OBS_IMAGE_3,
                                    'observation.images.side': OBS_IMAGE_2,
                                    'observation.images.top': OBS_IMAGE},
    'hangwu/piper_pick_terminal_and_place': {'observation.images.depth': None,
                                            'observation.images.realsense_color': OBS_IMAGE_3,
                                            'observation.images.side': OBS_IMAGE_2,
                                            'observation.images.top': OBS_IMAGE},
    'hannesill/koch_pnp_2_blocks_2_bins_200': {'observation.images.laptop': OBS_IMAGE_3,
                                                'observation.images.phone': OBS_IMAGE},
    'hannesill/koch_pnp_simple_50': {'observation.images.laptop': OBS_IMAGE_3,
                                    'observation.images.phone': OBS_IMAGE},
    'helper2424/hil-serl-push-circle-classifier': {'observation.images.web0': OBS_IMAGE,
                                                    'observation.images.web1': OBS_IMAGE_3},
    'ibru/bob_jetson': {'observation.images.front': OBS_IMAGE,
                        'observation.images.wrist': OBS_IMAGE_2},
    'ibru/bobo_groot_n1_trash_picker': {'observation.images.front': OBS_IMAGE,
                                        'observation.images.wrist': OBS_IMAGE_2},
    'ibru/bobo_jetson': {'observation.images.front': OBS_IMAGE,
                        'observation.images.wrist': OBS_IMAGE_2},
    'ibru/bobo_trash_collector': {'observation.images.front': OBS_IMAGE,
                                'observation.images.wrist': OBS_IMAGE_2},
    'imatrixlee/koch_place': {'observation.images.eye': OBS_IMAGE_2,
                            'observation.images.laptop': OBS_IMAGE_3,
                            'observation.images.phone': OBS_IMAGE},
    'jackvial/koch_with_rewards_4': {'observation.images.main': OBS_IMAGE},
    'jainamit/koch': {'observation.images.nexigo': OBS_IMAGE},
    'jainamit/koch_pickcube': {'observation.images.front_camera': OBS_IMAGE},
    'jainamit/koch_realcube3': {'observation.images.front_camera': OBS_IMAGE},
    'jannick-st/classifier': {'observation.images.cam_high': OBS_IMAGE,
                            'observation.images.cam_left_wrist': OBS_IMAGE_2,
                            'observation.images.cam_low': OBS_IMAGE_3,
                            'observation.images.cam_right_wrist': None},
    'jannick-st/push-cube-classifier_cropped_resized': {'observation.images.cam_low': OBS_IMAGE,
                                                        'observation.images.cam_right_wrist': OBS_IMAGE_2,
                                                        'observation.images.cam_top': OBS_IMAGE},
    'lalalala0620/koch_blue_paper_tape': {'observation.images.front': OBS_IMAGE,
                                        'observation.images.phone': OBS_IMAGE_3},
    'ma3oun/rpi_squares_1': {'observation.images.laptop': OBS_IMAGE},
    'meteorinc/koch_tea': {'observation.images.laptop': OBS_IMAGE,
                            'observation.images.phone': OBS_IMAGE_2},
    'mlfu7/pi0_conversion_no_pad_video': {'exterior_image_1_left': OBS_IMAGE,
                                        'wrist_image_left': OBS_IMAGE_2},
    'ncavallo/moss_train_gc_block': {'observation.images.front': OBS_IMAGE,
                                    'observation.images.top': OBS_IMAGE_2},
    'ncavallo/moss_train_grasp': {'observation.images.front': OBS_IMAGE,
                                'observation.images.top': OBS_IMAGE_2},
    'ncavallo/moss_train_grasp_new': {'observation.images.front': OBS_IMAGE,
                                    'observation.images.top': OBS_IMAGE_2},
    'nduque/act_50_ep': {'observation.images.above': OBS_IMAGE_3,
                        'observation.images.front': OBS_IMAGE},
    'nduque/act_50_ep2': {'observation.images.above': OBS_IMAGE_3,
                        'observation.images.front': OBS_IMAGE},
    'nduque/cam_setup2': {'observation.images.above': OBS_IMAGE_3,
                        'observation.images.front': OBS_IMAGE},
    'nduque/robustness_e1': {'observation.images.above': OBS_IMAGE_3,
                            'observation.images.front': OBS_IMAGE},
    'nduque/robustness_e2': {'observation.images.above': OBS_IMAGE_3,
                            'observation.images.front': OBS_IMAGE},
    'nduque/robustness_e3': {'observation.images.above': OBS_IMAGE_3,
                            'observation.images.front': OBS_IMAGE},
    'nduque/robustness_e4': {'observation.images.above': OBS_IMAGE_3,
                            'observation.images.front': OBS_IMAGE},
    'nduque/robustness_e5': {'observation.images.above': OBS_IMAGE_3,
                            'observation.images.front': OBS_IMAGE},
    # 'near0248/kinose_single_put_gum_in_bottle': {'observation.images.csi_camera': OBS_IMAGE_2,
    #                                             'observation.images.realsense': OBS_IMAGE},
    'nimitvasavat/Gr00t_lerobot': {'observation.images.back_image': OBS_IMAGE_3,
                                    'observation.images.image': OBS_IMAGE,
                                    'observation.images.wrist_image': OBS_IMAGE_2,
                                    'absolute_action': None,
                                    'annotation.human.action.task_description': None,
                                    'annotation.human.validity': None,'next.reward': None},
    'nimitvasavat/Gr00t_lerobotV2': {'observation.images.image': OBS_IMAGE,
                                    'observation.images.wrist_image': OBS_IMAGE_2,
                                    'absolute_action': None,
                                    'annotation.human.action.task_description': None,
                                    'annotation.human.validity': None,'next.reward': None},
    'nimitvasavat/Gr00t_lerobot_state_action': {'observation.images.image': OBS_IMAGE,
                                                'observation.images.wrist_image': OBS_IMAGE_2,
                                                'absolute_action': None,
                                                'annotation.human.action.task_description': None,
                                                'annotation.human.validity': None,'next.reward': None},
    'pepijn223/lekiwi_block_cleanup2': {'observation.images.front': OBS_IMAGE,
                                        'observation.images.wrist': OBS_IMAGE_2},
    'pepijn223/lekiwi_drive_in_circle': {'observation.images.front': OBS_IMAGE,
                                        'observation.images.wrist': OBS_IMAGE_2},
    'pepijn223/lekiwi_drive_in_circle_recover': {'observation.images.front': OBS_IMAGE,
                                                'observation.images.wrist': OBS_IMAGE_2},
    'pepijn223/lekiwi_pen': {'observation.images.front': OBS_IMAGE,
                            'observation.images.wrist': OBS_IMAGE_2},
    'rgarreta/koch_pick_place_lego': {'observation.images.laptop': OBS_IMAGE_2,
                                    'observation.images.phone': OBS_IMAGE},
    'rgarreta/koch_pick_place_lego_v2': {'observation.images.laptop': OBS_IMAGE_2,
                                        'observation.images.phone': OBS_IMAGE},
    'rgarreta/koch_pick_place_lego_v3': {'observation.images.laptop': OBS_IMAGE_2,
                                        'observation.images.phone': OBS_IMAGE},
    'rgarreta/koch_pick_place_lego_v6': {'observation.images.laptop': OBS_IMAGE_2,
                                        'observation.images.phone': OBS_IMAGE},
    'rgarreta/koch_pick_place_lego_v7': {'observation.images.laptop': OBS_IMAGE_2,
                                        'observation.images.phone': OBS_IMAGE},
    'rgarreta/koch_pick_place_lego_v8': {'observation.images.lateral': OBS_IMAGE_3,
                                        'observation.images.top': OBS_IMAGE,
                                        'observation.images.wrist': OBS_IMAGE_2},
    'roboticshack/sandee-kiwiv10': {'observation.images.front': OBS_IMAGE_3,
                                    'observation.images.wrist': OBS_IMAGE_2},
    'seeingrain/241228_pick_place_2cams': {'observation.images.side': OBS_IMAGE_3,
                                            'observation.images.top': OBS_IMAGE},
    'seeingrain/lego_3cameras': {'observation.images.side': OBS_IMAGE_3,
                                'observation.images.top': OBS_IMAGE,
                                'observation.images.wrist': OBS_IMAGE_2},
    'seeingrain/one_shot_learning_18episodes': {'observation.images.laptop': OBS_IMAGE_3,
                                                'observation.images.phone': OBS_IMAGE},
    'seeingrain/pick_lego_to_hand': {'observation.images.laptop': OBS_IMAGE_3,
                                    'observation.images.phone': OBS_IMAGE},
    'seeingrain/pick_place_lego': {'observation.images.laptop': OBS_IMAGE_3,
                                    'observation.images.phone': OBS_IMAGE},
    'seeingrain/pick_place_lego_wider_range_dang': {'observation.images.laptop': OBS_IMAGE_3,
                                                    'observation.images.phone': OBS_IMAGE},
    'seeingrain/pick_place_lego_wider_range_dong': {'observation.images.laptop': OBS_IMAGE_3,
                                                    'observation.images.phone': OBS_IMAGE},
    'seeingrain/pick_place_lego_wider_range_richard': {'observation.images.laptop': OBS_IMAGE_3,
                                                        'observation.images.phone': OBS_IMAGE},
    'seeingrain/pick_place_pink_lego': {'observation.images.laptop': OBS_IMAGE_3,
                                        'observation.images.phone': OBS_IMAGE},
    'seeingrain/pick_place_pink_lego_few_samples': {'observation.images.laptop': OBS_IMAGE_3,
                                                    'observation.images.phone': OBS_IMAGE},
    'seeingrain/pick_place_red_lego': {'observation.images.laptop': OBS_IMAGE_3,
                                        'observation.images.phone': OBS_IMAGE},
    'shin1107/koch_move_block_with_some_positions': {'observation.images.front': OBS_IMAGE,
                                                    'observation.images.top': OBS_IMAGE_3},
    'shin1107/koch_move_block_with_some_shapes': {'observation.images.front': OBS_IMAGE,
                                                'observation.images.top': OBS_IMAGE_3},
    'shin1107/koch_train_block': {'observation.images.front': OBS_IMAGE,
                                'observation.images.top': OBS_IMAGE_3},
    'takuzennn/aloha-pick100': {'observation.image.camera1': OBS_IMAGE,
                                'observation.image.camera2': OBS_IMAGE_2,
                                'observation.image.camera3': OBS_IMAGE_3},
    'takuzennn/square3': {'observations.images.agentview': OBS_IMAGE,
                        'observations.images.robot0_eye_in_hand': OBS_IMAGE_2},
    'theo-michel/lekiwi_v2': {'observation.images.front': OBS_IMAGE,
                            'observation.images.wrist': OBS_IMAGE_2},
    'theo-michel/lekiwi_v5': {'observation.images.front': OBS_IMAGE,
                            'observation.images.wrist': OBS_IMAGE_2},
    'twerdster/koch_new_training_red': {'observation.images.iphone': OBS_IMAGE_2,
                                        'observation.images.laptop': OBS_IMAGE_3},
    'twerdster/koch_training_red': {'observation.images.rightegocam': OBS_IMAGE_2,
                                    'observation.images.rightstereocam': OBS_IMAGE_3},
    'underctrl/handcamera_single_blue': {'observation.images.android': OBS_IMAGE,
                                        'observation.images.handcam': OBS_IMAGE_2,
                                        'observation.images.webcam': OBS_IMAGE_3},
    'underctrl/mutli-stacked-block_mutli-color_pick-up_80': {'observation.images.phone': OBS_IMAGE_3,
                                                            'observation.images.webcam': OBS_IMAGE},
    'underctrl/single-block_blue-color_pick-up_80': {'observation.images.phone': OBS_IMAGE_3,
                                                    'observation.images.webcam': OBS_IMAGE},
    'underctrl/single-block_multi-color_pick-up_50': {'observation.images.phone': OBS_IMAGE_3,
                                                    'observation.images.webcam': OBS_IMAGE},
    'underctrl/single-stacked-block_mutli-color_pick-up_80': {'observation.images.phone': OBS_IMAGE_3,
                                                            'observation.images.webcam': OBS_IMAGE},
    'underctrl/single-stacked-block_two-color_pick-up_80': {'observation.images.phone': OBS_IMAGE_3,
                                                            'observation.images.webcam': OBS_IMAGE},
    'yg-dev/koch_red_pen2': {'observation.images.laptop': OBS_IMAGE,
                            'observation.images.phone': OBS_IMAGE_2},
    'zliu157/i3r': {'observation.images.laptop': OBS_IMAGE,
                    'observation.images.phone': OBS_IMAGE_3},
    'zliu157/i3r2': {'observation.images.laptop': OBS_IMAGE,
                    'observation.images.phone': OBS_IMAGE_3},
    'zliu157/i3r3': {'observation.images.laptop': OBS_IMAGE,
                    'observation.images.phone': OBS_IMAGE_3},
    'zliu157/i3r5': {'observation.images.laptop': OBS_IMAGE,
                    'observation.images.phone': OBS_IMAGE_3},
    "jadechoghari/genesis-1k": {'observation.image.top': OBS_IMAGE,
                    'observation.image.wrist': OBS_IMAGE_2,
                    'observation.image.side': OBS_IMAGE_3},
    "jadechoghari/svla_so101-sim_task4_v3_multiple_1": {'observation.image.top': OBS_IMAGE,
                    'observation.image.wrist': OBS_IMAGE_2,
                    'observation.image.side': OBS_IMAGE_3},
    "jadechoghari/svla_so101-sim_task4_v3_multiple_2": {'observation.image.top': OBS_IMAGE,
                    'observation.image.wrist': OBS_IMAGE_2,
                    'observation.image.side': OBS_IMAGE_3},
}

EPISODES_DATASET_MAPPING = {
    "cadene/droid_1.0.1": list(range(50)),
    "danaaubakirova/svla_so100_task5_v3": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
    "danaaubakirova/svla_so100_task4_v3": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
}
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
