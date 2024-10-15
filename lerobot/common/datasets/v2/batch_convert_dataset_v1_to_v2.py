from pprint import pprint

from lerobot import available_datasets

# from lerobot.common.datasets.v2.convert_dataset_v1_to_v2 import convert_dataset

pprint(available_datasets)

for repo_id in available_datasets:
    name = repo_id.split("/")[1]
    if "aloha" in name:
        if "insertion" in name:
            single_task = "Insert the peg into the socket."
        elif "transfer" in name:
            single_task = "Pick up the cube with the right arm and transfer it to the left arm."
        elif "battery" in name:
            single_task = "Place the battery into the slot of the remote controller."
        elif "candy" in name:
            single_task = "Pick up the candy and unwrap it."
        elif "coffee_new" in name:
            single_task = "Place the coffee capsule inside the capsule container, then place the cup onto the center of the cup tray, then push the 'Hot Water' and 'Travel Mug' buttons."
        elif "coffee" in name:
            single_task = "Place the coffee capsule inside the capsule container, then place the cup onto the center of the cup tray."
        elif "cups_open" in name:
            single_task = "Pick up the plastic cup and open its lid."
        elif "fork_pick_up" in name:
            single_task = "Pick up the fork and place it on the plate."
        elif "pingpong_test" in name:
            single_task = "Transfer one of the two balls in the right glass into the left glass, then transfer it back to the right glass."
        elif "pro_pencil" in name:
            single_task = "Pick up the pencil with the right arm, hand it over to the left arm then place it back onto the table."
        elif "screw_driver" in name:
            single_task = "Pick up the screwdriver with the right arm, hand it over to the left arm then place it into the cup."
        elif "tape" in name:
            single_task = (
                "Cut a small piece of tape from the tape dispenser then place it on the cardboard box's edge."
            )
        elif "towel" in name:
            single_task = "Pick up a piece of paper towel and place it on the spilled liquid."
        elif "vinh_cup_left" in name:
            single_task = "Pick up the platic cup with the right arm, then pop its lid open with the left arm"
        elif "thread_velcro" in name:
            single_task = "Pick up the velcro cable tie with the left arm, then insert the end of the velcro tie into the other end's loop with the right arm."
        elif "shrimp" in name:
            single_task = "Sauté the raw shrimp on both sides, then serve it in the bowl."
        elif "wash_pan" in name:
            single_task = ""


# datasets = [
#     'lerobot/aloha_mobile_cabinet',
#     'lerobot/aloha_mobile_chair',
#     'lerobot/aloha_mobile_elevator',
#     'lerobot/aloha_mobile_shrimp',
#     'lerobot/aloha_mobile_wash_pan',
#     'lerobot/aloha_mobile_wipe_wine',
#     'lerobot/aloha_sim_insertion_human',
#     'lerobot/aloha_sim_insertion_human_image',
#     'lerobot/aloha_sim_insertion_scripted',
#     'lerobot/aloha_sim_insertion_scripted_image',
#     'lerobot/aloha_sim_transfer_cube_human',
#     'lerobot/aloha_sim_transfer_cube_human_image',
#     'lerobot/aloha_sim_transfer_cube_scripted',
#     'lerobot/aloha_sim_transfer_cube_scripted_image',
#     'lerobot/aloha_static_battery',
#     'lerobot/aloha_static_candy',
#     'lerobot/aloha_static_coffee',
#     'lerobot/aloha_static_coffee_new',
#     'lerobot/aloha_static_cups_open',
#     'lerobot/aloha_static_fork_pick_up',
#     'lerobot/aloha_static_pingpong_test',
#     'lerobot/aloha_static_pro_pencil',
#     'lerobot/aloha_static_screw_driver',
#     'lerobot/aloha_static_tape',
#     'lerobot/aloha_static_thread_velcro',
#     'lerobot/aloha_static_towel',
#     'lerobot/aloha_static_vinh_cup',
#     'lerobot/aloha_static_vinh_cup_left',
#     'lerobot/aloha_static_ziploc_slide',
#     'lerobot/asu_table_top',
#     'lerobot/austin_buds_dataset',
#     'lerobot/austin_sailor_dataset',
#     'lerobot/austin_sirius_dataset',
#     'lerobot/berkeley_autolab_ur5',
#     'lerobot/berkeley_cable_routing',
#     'lerobot/berkeley_fanuc_manipulation',
#     'lerobot/berkeley_gnm_cory_hall',
#     'lerobot/berkeley_gnm_recon',
#     'lerobot/berkeley_gnm_sac_son',
#     'lerobot/berkeley_mvp',
#     'lerobot/berkeley_rpt',
#     'lerobot/cmu_franka_exploration_dataset',
#     'lerobot/cmu_play_fusion',
#     'lerobot/cmu_stretch',
#     'lerobot/columbia_cairlab_pusht_real',
#     'lerobot/conq_hose_manipulation',
#     'lerobot/dlr_edan_shared_control',
#     'lerobot/dlr_sara_grid_clamp',
#     'lerobot/dlr_sara_pour',
#     'lerobot/droid_100',
#     'lerobot/fmb',
#     'lerobot/iamlab_cmu_pickup_insert',
#     'lerobot/imperialcollege_sawyer_wrist_cam',
#     'lerobot/jaco_play',
#     'lerobot/kaist_nonprehensile',
#     'lerobot/nyu_door_opening_surprising_effectiveness',
#     'lerobot/nyu_franka_play_dataset',
#     'lerobot/nyu_rot_dataset',
#     'lerobot/pusht',
#     'lerobot/pusht_image',
#     'lerobot/roboturk',
#     'lerobot/stanford_hydra_dataset',
#     'lerobot/stanford_kuka_multimodal_dataset',
#     'lerobot/stanford_robocook',
#     'lerobot/taco_play',
#     'lerobot/tokyo_u_lsmo',
#     'lerobot/toto',
#     'lerobot/ucsd_kitchen_dataset',
#     'lerobot/ucsd_pick_and_place_dataset',
#     'lerobot/uiuc_d3field',
#     'lerobot/umi_cup_in_the_wild',
#     'lerobot/unitreeh1_fold_clothes',
#     'lerobot/unitreeh1_rearrange_objects',
#     'lerobot/unitreeh1_two_robot_greeting',
#     'lerobot/unitreeh1_warehouse',
#     'lerobot/usc_cloth_sim',
#     'lerobot/utaustin_mutex',
#     'lerobot/utokyo_pr2_opening_fridge',
#     'lerobot/utokyo_pr2_tabletop_manipulation',
#     'lerobot/utokyo_saytap',
#     'lerobot/utokyo_xarm_bimanual',
#     'lerobot/utokyo_xarm_pick_and_place',
#     'lerobot/viola',
#     'lerobot/xarm_lift_medium',
#     'lerobot/xarm_lift_medium_image',
#     'lerobot/xarm_lift_medium_replay',
#     'lerobot/xarm_lift_medium_replay_image',
#     'lerobot/xarm_push_medium',
#     'lerobot/xarm_push_medium_image',
#     'lerobot/xarm_push_medium_replay',
#     'lerobot/xarm_push_medium_replay_image',
# ]

# convert_dataset(repo_id=repo_id)
