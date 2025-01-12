import base64
import time
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import zmq

from lerobot.common.robot_devices.control_utils import log_control_info, serialize_log_items
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait


class ControlPhase:
    TELEOPERATE = "Teleoperate"
    WARMUP = "Warmup"
    RECORD = "Record"
    RESET = "Reset"
    SAVING = "Saving"
    PROCESSING_DATASET = "Processing Dataset"
    UPLOADING_DATASET_TO_HUB = "Uploading Dataset to Hub"
    RECORDING_COMPLETE = "Recording Complete"


@dataclass
class ControlContextConfig:
    assign_rewards: bool = False
    control_phase: str = ControlPhase.TELEOPERATE
    num_episodes: int = 0
    robot: Robot = None
    fps: Optional[int] = None


class ControlContext:
    def __init__(self, config: ControlContextConfig):
        self.config = config
        self.modes_with_no_observation = [
            ControlPhase.RESET,
            ControlPhase.SAVING,
            ControlPhase.PROCESSING_DATASET,
            ControlPhase.UPLOADING_DATASET_TO_HUB,
            ControlPhase.RECORDING_COMPLETE,
        ]
        self.last_observation = None
        self._initialize_communication()
        self._initialize_state()

    def _initialize_state(self):
        self.events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "next_reward": 0,
        }

        if self.config.assign_rewards:
            self.events["next_reward"] = 0

        self.current_episode_index = 0

        # Define the control instructions
        self.controls = [
            ("Right Arrow", "Exit Early"),
            ("Left Arrow", "Rerecord"),
            ("Escape", "Stop"),
            ("Space", "Toggle Reward"),
        ]

    def _initialize_communication(self):
        self.zmq_context = zmq.Context()
        self.publisher_socket = self.zmq_context.socket(zmq.PUB)
        self.publisher_socket.bind("tcp://127.0.0.1:5555")

        self.command_sub_socket = self.zmq_context.socket(zmq.SUB)
        self.command_sub_socket.connect("tcp://127.0.0.1:5556")
        self.command_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    def _handle_browser_events(self):
        try:
            # Set a non-blocking polls
            if self.command_sub_socket.poll(timeout=0):  # Check if there's a message
                msg = self.command_sub_socket.recv_json()

                if msg.get("type") == "command" and msg.get("command") == "keydown":
                    key_pressed = msg.get("key_pressed")

                    if key_pressed == "ArrowRight":
                        print("Received 'ArrowRight' from browser -> Exit Early")
                        self.events["exit_early"] = True
                    elif key_pressed == "ArrowLeft":
                        print("Received 'ArrowLeft' from browser -> Rerecord Episode")
                        self.events["rerecord_episode"] = True
                        self.events["exit_early"] = True
                    elif key_pressed == "Escape":
                        print("Received 'Escape' from browser -> Stop")
                        self.events["stop_recording"] = True
                        self.events["exit_early"] = True
                    elif key_pressed == "Space":
                        # Toggle "next_reward"
                        self.events["next_reward"] = 1 if self.events["next_reward"] == 0 else 0
                        print(f"Space toggled reward to {self.events['next_reward']}")
            else:
                # No message available, continue
                pass

        except zmq.Again:
            # No message received within timeout
            pass
        except Exception as e:
            print(f"Error while polling for commands: {e}")

    def update_config(self, config: ControlContextConfig):
        """Update configuration and reinitialize UI components as needed"""
        self.config = config

        # Update ZMQ message with new config
        self._publish_config_update()

        return self

    def _publish_config_update(self):
        """Publish configuration update to ZMQ subscribers"""
        config_data = {
            "assign_rewards": self.config.assign_rewards,
            "control_phase": self.config.control_phase,
            "num_episodes": self.config.num_episodes,
            "current_episode": self.current_episode_index,
        }

        message = {
            "type": "config_update",
            "timestamp": time.time(),
            "config": config_data,
        }

        self.publisher_socket.send_json(message)

    def update_with_observations(
        self, observation: Dict[str, np.ndarray], start_loop_t: int, countdown_time: int
    ):
        if observation is not None:
            self.last_observation = observation

        if self.config.control_phase in self.modes_with_no_observation:
            observation = self.last_observation

        log_items = self.log_control_info(start_loop_t)
        self._publish_observations(observation, log_items, countdown_time)
        self._handle_browser_events()
        return self

    def _publish_observations(self, observation: Dict[str, np.ndarray], log_items: list, countdown_time: int):
        """Encode and publish observation data with current configuration"""
        processed_data = {}
        for key, value in observation.items():
            if "image" in key:
                image = value.numpy() if torch.is_tensor(value) else value
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                success, buffer = cv2.imencode(".jpg", bgr_image)
                if success:
                    b64_jpeg = base64.b64encode(buffer).decode("utf-8")
                    processed_data[key] = {
                        "type": "image",
                        "encoding": "jpeg_base64",
                        "data": b64_jpeg,
                        "shape": image.shape,
                    }
            else:
                tensor_data = value.detach().cpu().numpy() if torch.is_tensor(value) else value
                processed_data[key] = {
                    "type": "tensor",
                    "data": tensor_data.tolist(),
                    "shape": tensor_data.shape,
                }

        # Include current configuration in observation update
        config_data = {
            "assign_rewards": self.config.assign_rewards,
            "control_phase": self.config.control_phase,
            "num_episodes": self.config.num_episodes,
            "current_episode": self.current_episode_index,
        }

        # Sanitize countdown time. if inf set to max 32-bit int
        countdown_time = int(countdown_time) if countdown_time != float("inf") else 2 ** 31 - 1
        if self.config.control_phase == ControlPhase.TELEOPERATE:
            countdown_time = 0

        message = {
            "type": "observation_update",
            "timestamp": time.time(),
            "data": processed_data,
            "events": self.get_events(),
            "config": config_data,
            "log_items": serialize_log_items(log_items),
            "countdown_time": countdown_time,
        }

        self.publisher_socket.send_json(message)

    def update_current_episode(self, episode_index):
        self.current_episode_index = episode_index
        return self

    def get_events(self):
        return self.events.copy()

    def log_control_info(self, start_loop_t):
        log_items = []
        fps = self.config.fps
        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

            dt_s = time.perf_counter() - start_loop_t
            log_items = log_control_info(self.config.robot, dt_s, fps=fps)

        return log_items
    
    def log_say(self, message):
        self._publish_log_say(message)

    def _publish_log_say(self, message):
        message = {
            "type": "log_say",
            "timestamp": time.time(),
            "message": message,
        }

        self.publisher_socket.send_json(message)

    def cleanup(self, robot=None):
        """Clean up resources and connections"""
        if robot:
            robot.disconnect()

        self.publisher_socket.close()
        self.command_sub_socket.close()
        self.zmq_context.term()


if __name__ == "__main__":
    import time

    import cv2
    import numpy as np
    import torch

    def read_image_from_camera(cap):
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return torch.tensor(frame_rgb).float()

    config = ControlContextConfig(
        assign_rewards=True,
        control_phase=ControlPhase.RECORD,
        num_episodes=200,
        fps=30,
    )
    context = ControlContext(config)
    context.update_current_episode(199)

    cameras = {"main": cv2.VideoCapture(0), "top": cv2.VideoCapture(4)}

    for name, cap in cameras.items():
        if not cap.isOpened():
            raise Exception(f"Error: Could not open {name} camera")

    while True:
        images = {}
        camera_logs = {}
        for name, cap in cameras.items():
            before_camread_t = time.perf_counter()
            images[name] = read_image_from_camera(cap)
            camera_logs[f"read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Create state tensor (simulating follower positions)
        state = torch.tensor([10.0195, 128.9355, 173.0566, -13.2715, -7.2070, 34.4531])

        obs_dict = {"observation.state": state}

        for name in cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        # Update context with observations
        context.update_with_observations(obs_dict, time.perf_counter(), countdown_time=10)
        events = context.get_events()

        if events["exit_early"]:
            break

    for cap in cameras.values():
        cap.release()
    cv2.destroyAllWindows()
