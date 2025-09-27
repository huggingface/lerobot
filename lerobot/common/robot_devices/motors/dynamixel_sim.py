
import time
import numpy as np
from tqdm import tqdm

import mujoco
import mujoco.viewer

import argparse
import copy
import os
import pathlib
import time

import numpy as np
import torch
from datasets import Dataset, Features, Sequence, Value

from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, DATA_DIR, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, save_images_concurrently
from lerobot.common.datasets.utils import (
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames
from lerobot.scripts.push_dataset_to_hub import push_meta_data_to_hub, push_videos_to_hub, save_meta_data
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.robot_devices.robots.koch import KochRobot

from lerobot.common.robot_devices.cameras.sim import SimCamera


### SimCamera classes
class SimRobotDeviceNotConnectedError(Exception):
    """Exception raised when the robot device is not connected."""

    def __init__(
        self, message="This robot device is not connected. Try calling `robot_device.connect()` first."
    ):
        self.message = message
        super().__init__(self.message)

class SimRobotDeviceAlreadyConnectedError(Exception):
    """Exception raised when the robot device is already connected."""

    def __init__(
        self,
        message="This robot device is already connected. Try not calling `robot_device.connect()` twice.",
    ):
        self.message = message
        super().__init__(self.message)



### SimDynamixelMotorsBus class
class SimDynamixelMotorsBus:
    
    # TODO(rcadene): Add a script to find the motor indices without DynamixelWizzard2
    """
    The DynamixelMotorsBus class allows to efficiently read and write to the simulated Bus managing mujoco environment. 
    The class is designed to be used with a mujoco environment with the low cost robot 6DoF Robot.
    """

    def __init__(
        self,
        motors,
        path_scene="lerobot/assets/low_cost_robot_6dof/pick_place_cube.xml"
    ):
        
        self.path_scene = path_scene
        self.model = mujoco.MjModel.from_xml_path(path_scene)
        self.data  = mujoco.MjData(self.model)
        self.is_connected = False
        self.motors = motors
        self.logs = {}

    def connect(self):
        self.is_connected = True

    def reconnect(self):
        self.is_connected = True

    def are_motors_configured(self):
        return True

    def configure_motors(self):
        print("Configuration is done!")

    def find_motor_indices(self, possible_ids=None):
        return [1, 2, 3, 4, 5, 6]

    def set_bus_baudrate(self, baudrate):
        return

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    def set_calibration(self, calibration: dict[str, tuple[int, bool]]):
        self.calibration = calibration

    def apply_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        # Convert from unsigned int32 original range [0, 2**32[ to centered signed int32 range [-2**31, 2**31[
        values = values.astype(np.int32)
        return values

    def revert_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        """Inverse of `apply_calibration`."""
        return values


    ## read the motor values from the mujoco environment
    # motor_models: The motor models to read
    # motor_ids: The motor ids to read
    # data_name: The data name to read
    
    def _read_with_motor_ids(self, motor_models, motor_ids, data_name):
        return_list = True
        if not isinstance(motor_ids, list):
            return_list = False
            motor_ids = [motor_ids]

        values = []
        for idx in motor_ids:
            values.append(self.data.qpos[-6+idx-1])

        if return_list:
            return values
        else:
            return values[0]

    ## read the motor values from the mujoco environment
    # data_name: The data name to read
    # motor_names: The motor names to read

    def read(self, data_name, motor_names: str | list[str] | None = None):

        if not self.is_connected:
            raise SimRobotDeviceNotConnectedError(f"SimDynamixelMotorsBus({self.path_scene}) is not connected. You need to run `motors_bus.connect()`.")

        values = []

        if motor_names is None:
            for idx in range(1, 7):
                values.append(self.data.qpos[idx-6-1])
        else:
            for name in motor_names:
                idx_motor = self.motors[name][0]-6-1
                values.append(self.data.qpos[idx_motor])

        return np.asarray(values)


    ## write the motor values to the mujoco environment
    # data_name: The data name to write
    # values: The values to write
    # motor_names: The motor names to write

    def _write_with_motor_ids(self, motor_models, motor_ids, data_name, values):
        if not self.is_connected:
            raise SimRobotDeviceNotConnectedError(
                f"SimDynamixelMotorsBus({self.path_scene}) is not connected. You need to run `motors_bus.connect()`."
            )        
        for idx, value in zip(motor_ids, values):
            self.data.qpos[idx-6-1] = value


    ## convert the real robot joint positions to mujoco joint positions
    # with support for inverted joints
    # real_positions: Joint positions in degrees
    # transforms: List of transforms to apply to each joint
    # oppose: List of oppositions to apply to each joint

    @staticmethod
    def real_to_mujoco(real_positions, transforms, oppose):
        """
        Convert joint positions from real robot (in degrees) to Mujoco (in radians),
        with support for inverted joints.
        
        Parameters:
        real_positions (np.array): Joint positions in degrees.
        transforms (list): List of transforms to apply to each joint.
        oppose (list): List of oppositions to apply to each joint.
        
        Returns:
        np.array: Joint positions in radians.
        """
        real_positions = np.array(real_positions)
        mujoco_positions = real_positions * (np.pi / 180.0)

        for id in range(6):
            mujoco_positions[id] = transforms[id] + mujoco_positions[id]
            mujoco_positions[id] *= oppose[id]

        return mujoco_positions


    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):

        if not self.is_connected:
            raise SimRobotDeviceNotConnectedError(
                f"SimDynamixelMotorsBus({self.path_scene}) is not connected. You need to run `motors_bus.connect()`."
            )
        
        ## do not write the following data for simulation motors so far
        if data_name in ["Torque_Enable", "Operating_Mode", "Homing_Offset", "Drive_Mode", "Position_P_Gain", "Position_I_Gain", "Position_D_Gain"]:
            return

        if motor_names is None or len(motor_names) == 6:
            self.data.qpos[-6:] = self.real_to_mujoco(values, transforms=[0, 
                                                                          -np.pi / 2.0,
                                                                          np.pi + np.pi / 2.0,
                                                                          0,
                                                                          np.pi -np.pi / 2.0,
                                                                          0], 
                                                                          oppose=[-1,1,-1,1,-1,-1])

            ## update the mujoco environment
            mujoco.mj_step(follower.model, follower.data)
            viewer.sync()


    def disconnect(self):
        if not self.is_connected:
            raise SimRobotDeviceNotConnectedError(
                f"SimDynamixelMotorsBus({self.path_scene}) is not connected. Try running `motors_bus.connect()` first."
            )

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()


## test the leader motors reading
def test_read_leader_position():

    leader = DynamixelMotorsBus(
                port="/dev/ttyACM0",
                motors={
                    # name: (index, model)
                    "shoulder_pan": (1, "xl330-m077"),
                    "shoulder_lift": (2, "xl330-m077"),
                    "elbow_flex": (3, "xl330-m077"),
                    "wrist_flex": (4, "xl330-m077"),
                    "wrist_roll": (5, "xl330-m077"),
                    "gripper": (6, "xl330-m077"),
                },
            )

    leader.connect()
    while True:
        print(leader.read("Present_Position", 
                          ["shoulder_pan", "shoulder_lift", 
                           "elbow_flex", "wrist_flex", 
                           "wrist_roll", "gripper"]))

    leader.disconnect()

## global variables
current_motor_ids=1
stop_episode = False
stop_record  = False

## callback function for the keyboard control over mujoco environment
# [1-6] to select the current controlled motor using the keyboard
# [8] to increase the current motor position
# [9] to decrease the current motor position
# [7] to stop the episode
# [space] to stop the recording

def key_callback(keycode):

    global current_motor_ids
    global stop_episode
    global stop_record

    print(f"Key pressed: [{chr(keycode)}]")

    ## stop the episode
    if chr(keycode) == "7":
        stop_episode = True

    if chr(keycode) == " ":
        stop_record = True

    ## change the motor id to control
    if chr(keycode) in ["1", "2", "3", "4", "5", "6"]:
        current_motor_ids = int(chr(keycode))
        print(f"Current motor id: {current_motor_ids}")

    ## increase the motor position
    if chr(keycode) == "8":
        idx_motor = current_motor_ids-6-1
        follower.data.qpos[idx_motor] += 0.1
        mujoco.mj_forward(follower.model, 
                follower.data)
        viewer.sync()

    ## decrease the motor position
    if chr(keycode) == "9":
        idx_motor = current_motor_ids-6-1
        follower.data.qpos[idx_motor] -= 0.1
        mujoco.mj_forward(follower.model, 
                follower.data)
        viewer.sync()


## function to replace the cube in the mujoco environment
def mujoco_replace_cube(model, data):
    cube_low = np.array([-0.15, 0.10, 0.015])
    cube_high = np.array([0.15, 0.25, 0.015])
    cube_pos = np.random.uniform(cube_low, cube_high)
    cube_rot = np.array([1.0, 0.0, 0.0, 0.0])
    data.qpos[0:7] = np.concatenate([cube_pos, cube_rot])
    mujoco.mj_forward(model, data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--leader-port", type=str, default="/dev/ttyACM0", help="Port for the leader motors")
    parser.add_argument("--calibration-path", type=str, default=".cache/calibration/koch.pkl", help="Path to the robots calibration file")  
    parser.add_argument("--test-leader", action="store_true", help="Test the leader motors")
    parser.add_argument("--mujoco-replace-cube", action="store_true", help="Replace the cube in the mujoco environment")


    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--keep-last", action="store_true")
    parser.add_argument("--repo-id", type=str, default="jnm38")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second of the recording.")
    parser.add_argument(
        "--fps_tolerance",
        type=float,
        default=0.1,
        help="Tolerance in fps for the recording before dropping episodes.",
    )
    parser.add_argument(
        "--revision", type=str, default=CODEBASE_VERSION, help="Codebase version used to generate the dataset."
    )

    args = parser.parse_args()

    ## test the leader motors reading 
    if args.test_leader:
        test_read_leader_position()
        exit()


    ## create the leader and follower motors
    leader = DynamixelMotorsBus(
                port=args.leader_port,
                motors={
                    # name: (index, model)
                    "shoulder_pan": (1, "xl330-m077"),
                    "shoulder_lift": (2, "xl330-m077"),
                    "elbow_flex": (3, "xl330-m077"),
                    "wrist_flex": (4, "xl330-m077"),
                    "wrist_roll": (5, "xl330-m077"),
                    "gripper": (6, "xl330-m077"),
                },
            )

    follower = SimDynamixelMotorsBus(
                path_scene="lerobot/assets/low_cost_robot_6dof/pick_place_cube.xml",
                motors={
                    # name: (index, model)
                    "shoulder_pan": (1, "xl430-w250"),
                    "shoulder_lift": (2, "xl430-w250"),
                    "elbow_flex": (3, "xl330-m288"),
                    "wrist_flex": (4, "xl330-m288"),
                    "wrist_roll": (5, "xl330-m288"),
                    "gripper": (6, "xl330-m288"),
                },
            )
    
    ## create cameras which are instantiated to the mujoco environment in the simulated follower robot class
    cameras = {
        "image_top":   SimCamera(id_camera="camera_top",   model=follower.model, data=follower.data, camera_index=0, fps=30, width=640, height=480),
        "image_front": SimCamera(id_camera="camera_front", model=follower.model, data=follower.data, camera_index=1, fps=30, width=640, height=480),
    }

    ## define the path to store the data
    DATA_DIR = pathlib.Path("data_traces")
    out_data = DATA_DIR / args.repo_id

    # During data collection, frames are stored as png images in `images_dir`
    images_dir = out_data / "images"

    # After data collection, png images of each episode are encoded into a mp4 file stored in `videos_dir`
    videos_dir = out_data / "videos"
    meta_data_dir = out_data / "meta_data"

    # Create image and video directories
    if not os.path.exists(images_dir):
        os.makedirs(images_dir, exist_ok=True)
    if not os.path.exists(videos_dir):
        os.makedirs(videos_dir, exist_ok=True)


    ## Define the episode data index and dictionaries to store the data
    ep_dicts = []
    episode_data_index = {"from": [], "to": []}
    ep_fps = []
    id_from = 0
    id_to = 0


    ## start the mujoco environment and start the teleoperation
    ep_idx = 0
    with mujoco.viewer.launch_passive(follower.model, follower.data, key_callback=key_callback) as viewer:    

        robot = KochRobot(leader_arms   = {"main": leader},
                          follower_arms = {"main": follower},
                          cameras       = cameras,
                          calibration_path=args.calibration_path)
        robot.connect()
        
        while stop_record == False:

            # sample the initial position of the cube
            if args.mujoco_replace_cube:
                mujoco_replace_cube(follower.model, follower.data)

            # Instantiate the episode data storage
            obs_replay = {}
            obs_replay["observation"] = []
            obs_replay["action"] = []
            obs_replay["image_top"] = []
            obs_replay["image_front"] = []
            timestamps = []
            start_time = time.time()

            print(f"Start episode {ep_idx} ...")
            while stop_episode == False and stop_record == False:

                obs_dict, action_dict = robot.teleop_step(record_data=True)
                obs_replay["observation"].append(copy.deepcopy(obs_dict["observation.state"]))
                obs_replay["action"].append(copy.deepcopy(action_dict["action"]))
                obs_replay["image_top"].append(copy.deepcopy(obs_dict["observation.images.image_top"].numpy()))
                obs_replay["image_front"].append(copy.deepcopy(obs_dict["observation.images.image_front"].numpy()))
                timestamps.append(time.time() - start_time)

            stop_episode = False

            ## Tolerance workaround ...
            num_frames = len(timestamps)
            timestamps = np.linspace(0, timestamps[-1], num_frames)

            # os.system(f'spd-say "saving episode"')
            ep_dict = {}

            # store the images of the episode in .png and create the video
            for img_key in ["image_top", "image_front"]:
                save_images_concurrently(
                    obs_replay[img_key],
                    images_dir / f"{img_key}_episode_{ep_idx:06d}",
                    args.num_workers,
                )
                fname = f"{img_key}_episode_{ep_idx:06d}.mp4"

                # store the reference to the video frame
                ep_dict[f"observation.{img_key}"] = [{"path": f"videos/{fname}", "timestamp": tstp} for tstp in timestamps]

            # store the state, action, episode index, frame index, timestamp and next done
            state     = torch.tensor(np.array(obs_replay["observation"]))
            action    = torch.tensor(np.array(obs_replay["action"]))
            next_done = torch.zeros(num_frames, dtype=torch.bool)
            next_done[-1] = True

            ep_dict["observation.state"] = state
            ep_dict["action"] = action
            ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames, dtype=torch.int64)
            ep_dict["frame_index"]   = torch.arange(0, num_frames, 1)
            ep_dict["timestamp"]     = torch.tensor(timestamps)
            ep_dict["next.done"]     = next_done
            ep_fps.append(num_frames / timestamps[-1])
            print(f"Episode {ep_idx} done, fps: {ep_fps[-1]:.2f}")

            ## store the episode data index
            episode_data_index["from"].append(id_from)
            episode_data_index["to"].append(id_from + num_frames if args.keep_last else id_from + num_frames - 1)

            ## update the episode data index
            id_to = id_from + num_frames if args.keep_last else id_from + num_frames - 1
            id_from = id_to

            ## store the episode data in the overall data list
            ep_dicts.append(ep_dict)
            ep_idx += 1

    ## end the teleoperation
    robot.disconnect()

    ## encode the images to videos for all the episodes
    for idx in range(ep_idx):
        for img_key in ["image_top", "image_front"]:
            encode_video_frames(
                images_dir / f"{img_key}_episode_{idx:06d}",
                videos_dir / f"{img_key}_episode_{idx:06d}.mp4",
                ep_fps[idx],
            )

    ## concatenate the episodes and store the data
    data_dict = concatenate_episodes(ep_dicts)  # Since our fps varies we are sometimes off tolerance for the last frame

    ## store the data in the dataset format in a features dictionary
    features = {}
    keys = [key for key in data_dict if "observation.image_" in key]
    for key in keys:
        features[key.replace("observation.image_", "observation.images.")] = VideoFrame()
        data_dict[key.replace("observation.image_", "observation.images.")] = data_dict[key]
        del data_dict[key]

    features["observation.state"] = Sequence(length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None))
    features["action"] = Sequence(length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None))
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"]   = Value(dtype="int64", id=None)
    features["timestamp"]     = Value(dtype="float32", id=None)
    features["next.done"]     = Value(dtype="bool", id=None)
    features["index"]         = Value(dtype="int64", id=None)

    ## store the data in the dataset format    
    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)

    ## store the meta data
    info = {
        #"fps": sum(ep_fps) / len(ep_fps),  # to have a good tolerance in data processing for the slowest video
        "fps": 24,  # to have a good tolerance in data processing for the slowest video
        "video": ep_idx,
    }
    
    ## store the data in the LeRobotDataset format
    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id=args.repo_id,
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )

    ## compute the stats and save the meta data
    stats = compute_stats(lerobot_dataset, num_workers=args.num_workers)
    save_meta_data(info, stats, episode_data_index, meta_data_dir)
    

    ## save the data in the dataset format in disk
    hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
    hf_dataset.save_to_disk(str(out_data / "train"))


    ## push the data to the hub
    if args.push_to_hub:

        repo_name = f"{args.repo_id}/lowcostrobot-mujoco-pickplace"

        hf_dataset.push_to_hub(repo_name, token=True, revision="main")
        hf_dataset.push_to_hub(repo_name, token=True, revision=args.revision)

        push_meta_data_to_hub(repo_name, meta_data_dir, revision="main")
        push_meta_data_to_hub(repo_name, meta_data_dir, revision=args.revision)

        push_videos_to_hub(repo_name, videos_dir, revision="main")
        push_videos_to_hub(repo_name, videos_dir, revision=args.revision)
