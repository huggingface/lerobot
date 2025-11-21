from multiprocessing import shared_memory, Value, Array, Lock
from typing import Any
import numpy as np
import argparse
import threading
import torch
from unitree_lerobot.eval_robot.image_server.image_client import ImageClient
from unitree_lerobot.eval_robot.robot_control.robot_arm import (
    G1_29_ArmController,
    G1_23_ArmController,
)
from unitree_lerobot.eval_robot.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK
from unitree_lerobot.eval_robot.robot_control.robot_hand_unitree import (
    Dex3_1_Controller,
    Dex1_1_Gripper_Controller,
)

from unitree_lerobot.eval_robot.utils.episode_writer import EpisodeWriter

from unitree_lerobot.eval_robot.robot_control.robot_hand_inspire import Inspire_Controller
from unitree_lerobot.eval_robot.robot_control.robot_hand_brainco import Brainco_Controller


from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

# Configuration for robot arms
ARM_CONFIG = {
    "G1_29": {"controller": G1_29_ArmController, "ik_solver": G1_29_ArmIK, "dof": 14},
    "G1_23": {"controller": G1_23_ArmController, "ik_solver": G1_23_ArmIK, "dof": 14},
    # Add other arms here
}

# Configuration for end-effectors
EE_CONFIG: dict[str, dict[str, Any]] = {
    "dex3": {
        "controller": Dex3_1_Controller,
        "dof": 7,
        "shared_mem_type": "Array",
        "shared_mem_size": 7,
        # "out_len": 14,
    },
    "dex1": {
        "controller": Dex1_1_Gripper_Controller,
        "dof": 1,
        "shared_mem_type": "Value",
        # "out_len": 2,
    },
    "inspire1": {
        "controller": Inspire_Controller,
        "dof": 6,
        "shared_mem_type": "Array",
        "shared_mem_size": 6,
        # "out_len": 12,
    },
    "brainco": {
        "controller": Brainco_Controller,
        "dof": 6,
        "shared_mem_type": "Array",
        "shared_mem_size": 6,
        # "out_len": 12,
    },
}


def setup_image_client(args: argparse.Namespace) -> dict[str, Any]:
    """Initializes and starts the image client and shared memory."""
    # image client: img_config should be the same as the configuration in image_server.py (of Robot's development computing unit)
    if getattr(args, "sim", False):
        img_config = {
            "fps": 30,
            "head_camera_type": "opencv",
            "head_camera_image_shape": [480, 640],  # Head camera resolution
            "head_camera_id_numbers": [0],
            "wrist_camera_type": "opencv",
            "wrist_camera_image_shape": [480, 640],  # Wrist camera resolution
            "wrist_camera_id_numbers": [2, 4],
        }
    else:
        img_config = {#we dont have wrist for now 
            "fps": 30,
            "head_camera_type": "opencv",
            "head_camera_image_shape": [480, 640],  # Head camera resolution
            "head_camera_id_numbers": [0],
            #"wrist_camera_type": "opencv",
            #"wrist_camera_image_shape": [480, 640],  # Wrist camera resolution
            #"wrist_camera_id_numbers": [2, 4],
        }

    ASPECT_RATIO_THRESHOLD = 2.0  # If the aspect ratio exceeds this value, it is considered binocular
    if len(img_config["head_camera_id_numbers"]) > 1 or (
        img_config["head_camera_image_shape"][1] / img_config["head_camera_image_shape"][0] > ASPECT_RATIO_THRESHOLD
    ):
        BINOCULAR = True
    else:
        BINOCULAR = False
    if "wrist_camera_type" in img_config:
        WRIST = True
    else:
        WRIST = False
   # BINOCULAR = True#add flag properly
    if BINOCULAR and not (
        img_config["head_camera_image_shape"][1] / img_config["head_camera_image_shape"][0] > ASPECT_RATIO_THRESHOLD
    ):
        tv_img_shape = (img_config["head_camera_image_shape"][0], img_config["head_camera_image_shape"][1] * 2, 3)
    else:
        tv_img_shape = (img_config["head_camera_image_shape"][0], img_config["head_camera_image_shape"][1], 3)

    tv_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=tv_img_shm.buf)
    
    # Initialize wrist camera variables
    wrist_img_shm = None
    wrist_img_array = None
    wrist_img_shape = None

    if WRIST and getattr(args, "sim", False):
        wrist_img_shape = (img_config["wrist_camera_image_shape"][0], img_config["wrist_camera_image_shape"][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype=np.uint8, buffer=wrist_img_shm.buf)
        img_client = ImageClient(
            tv_img_shape=tv_img_shape,
            tv_img_shm_name=tv_img_shm.name,
            wrist_img_shape=wrist_img_shape,
            wrist_img_shm_name=wrist_img_shm.name,
            server_address="127.0.0.1",
        )
    elif WRIST and not getattr(args, "sim", False):
        wrist_img_shape = (img_config["wrist_camera_image_shape"][0], img_config["wrist_camera_image_shape"][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype=np.uint8, buffer=wrist_img_shm.buf)
        img_client = ImageClient(
            tv_img_shape=tv_img_shape,
            tv_img_shm_name=tv_img_shm.name,
            wrist_img_shape=wrist_img_shape,
            wrist_img_shm_name=wrist_img_shm.name,
        )
    else:
        img_client = ImageClient(tv_img_shape=tv_img_shape, tv_img_shm_name=tv_img_shm.name)

    has_wrist_cam = "wrist_camera_type" in img_config

    image_receive_thread = threading.Thread(target=img_client.receive_process, daemon=True)
    image_receive_thread.daemon = True
    image_receive_thread.start()

    # Only include shared memory objects that exist
    shm_resources = [tv_img_shm]
    if wrist_img_shm is not None:
        shm_resources.append(wrist_img_shm)

    return {
        "tv_img_array": tv_img_array,
        "wrist_img_array": wrist_img_array,
        "tv_img_shape": tv_img_shape,
        "wrist_img_shape": wrist_img_shape,
        "is_binocular": BINOCULAR,
        "has_wrist_cam": has_wrist_cam,
        "shm_resources": shm_resources,
    }


def _resolve_out_len(spec: dict[str, Any]) -> int:
    return int(spec.get("out_len", 2 * int(spec["dof"])))


def setup_robot_interface(args: argparse.Namespace) -> dict[str, Any]:
    """
    Initializes robot controllers and IK solvers based on configuration.
    """
    # ---------- Arm ----------
    arm_spec = ARM_CONFIG[args.arm]
    arm_ik = arm_spec["ik_solver"]()
    is_sim = getattr(args, "sim", False)
    motion_mode = getattr(args, "motion", False)
    arm_ctrl = arm_spec["controller"](motion_mode=motion_mode, simulation_mode=is_sim)

    result = {
        "arm_ctrl": arm_ctrl,
        "arm_ik": arm_ik,
        "arm_dof": int(arm_spec["dof"]),
    }
    
    # ---------- Simulation Components ----------
    if is_sim:

        from unitree_lerobot.eval_robot.utils.sim_state_topic import start_sim_state_subscribe, start_sim_reward_subscribe
        from unitree_lerobot.eval_robot.utils.episode_writer import EpisodeWriter
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
        
        # Create simulation DDS subscribers
        sim_state_subscriber = start_sim_state_subscribe()
        sim_reward_subscriber = start_sim_reward_subscribe()
        
        # Initialize DDS channel 1 for simulation mode (if not already initialized)
        try:
            ChannelFactoryInitialize(1)
        except:
            pass  # May already be initialized by arm controller
        
        # Create reset pose publisher (publishes to rt/reset_pose/cmd to match simulation)
        reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
        reset_pose_publisher.Init()
        
        # Create episode writer for data collection
        episode_writer = EpisodeWriter(
            task_dir=args.task_dir if hasattr(args, 'task_dir') else "./data",
            frequency=args.frequency if hasattr(args, 'frequency') else 30
        )
        
        result.update({
            "sim_state_subscriber": sim_state_subscriber,
            "sim_reward_subscriber": sim_reward_subscriber,
            "reset_pose_publisher": reset_pose_publisher,
            "episode_writer": episode_writer,
        })
    
    # ---------- End Effector ----------
    if hasattr(args, 'ee') and args.ee:
        ee_spec = EE_CONFIG.get(args.ee)
        if ee_spec:
            result["ee_dof"] = int(ee_spec["dof"])
            # Add end effector shared memory setup here if needed
            result["ee_shared_mem"] = {"lock": None, "state": [], "left": [], "right": []}
        else:
            result["ee_dof"] = 0
            result["ee_shared_mem"] = {"lock": None, "state": [], "left": [], "right": []}
    else:
        result["ee_dof"] = 0
        result["ee_shared_mem"] = {"lock": None, "state": [], "left": [], "right": []}

        EE_DOF = 7
        left_in  = Array("d", EE_DOF, lock=True)   # desired left finger q targets
        right_in = Array("d", EE_DOF, lock=True)   # desired right finger q targets

        # outputs (states + echoed actions), both length 14 (left 7 + right 7)
        dual_state_out  = Array("d", 2*EE_DOF, lock=True)
        dual_action_out = Array("d", 2*EE_DOF, lock=True)
        dual_lock = Lock()

        if getattr(args, 'ee', None) == 'dex3':
            from robot_control.robot_hand_unitree import Dex3_1_Controller
            ctrl = Dex3_1_Controller(
                left_hand_array_in=left_in,
                right_hand_array_in=right_in,
                dual_hand_data_lock=dual_lock,
                dual_hand_state_array_out=dual_state_out,
                dual_hand_action_array_out=dual_action_out,
                fps=100.0,
                Unit_Test=False,
                simulation_mode=getattr(args, "sim", False),  # True in sim
            )
            result["ee_dof"] = EE_DOF
            result["ee_shared_mem"] = {
                "lock": dual_lock,
                "state": dual_state_out,   # 14-length: [L7, R7]
                "left":  left_in,          # input commands for left hand
                "right": right_in,         # input commands for right hand
            }

    return result


def process_images_and_observations(
    tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam, arm_ctrl
):
    """Processes images and generates observations."""
    current_tv_image = tv_img_array.copy()
    current_wrist_image = wrist_img_array.copy() if has_wrist_cam else None

    left_top_cam = current_tv_image[:, : tv_img_shape[1] // 2] if is_binocular else current_tv_image
    right_top_cam = current_tv_image[:, tv_img_shape[1] // 2 :] if is_binocular else left_top_cam.copy()

    left_wrist_cam = right_wrist_cam = None
    if has_wrist_cam and current_wrist_image is not None:
        left_wrist_cam = current_wrist_image[:, : wrist_img_shape[1] // 2]
        right_wrist_cam = current_wrist_image[:, wrist_img_shape[1] // 2 :]
    observation = {
        "observation.image": torch.from_numpy(left_top_cam),
        "observation.images.cam_right_high": torch.from_numpy(right_top_cam),
        "observation.images.cam_left_wrist": torch.from_numpy(left_wrist_cam) if has_wrist_cam else np.zeros((480, 640, 3)),
        "observation.images.cam_right_wrist": torch.from_numpy(right_wrist_cam) if has_wrist_cam else np.zeros((480, 640, 3)),
    }
    current_arm_q = arm_ctrl.get_current_dual_arm_q()
    
    return observation, current_arm_q


def publish_reset_category(category: int, publisher):  # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")
