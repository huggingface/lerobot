import pygame
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import cv2
import base64
import zmq
import torch
import time


class ControlPhase:
    TELEOPERATE = "Teleoperate"
    WARMUP = "Warmup"
    RECORD = "Record"
    RESET = "Reset"


@dataclass
class ControlContextConfig:
    display_cameras: bool = False
    play_sounds: bool = False
    assign_rewards: bool = False
    debug_mode: bool = False
    control_phase: str = ControlPhase.TELEOPERATE
    num_episodes: int = 0


class ControlContext:
    def __init__(self, config: Optional[ControlContextConfig] = None):
        self.config = config or ControlContextConfig()
        pygame.init()
        if not self.config.display_cameras:
            pygame.display.set_mode((1, 1), pygame.HIDDEN)

        self.screen = None
        self.image_positions = {}
        self.padding = 20
        self.title_height = 30
        self.events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "next_reward": 0,
        }

        if self.config.assign_rewards:
            self.events["next_reward"] = 0

        self.pressed_keys = []
        self.font = pygame.font.SysFont("courier", 24)
        self.small_font = pygame.font.SysFont("courier", 18)
        self.current_episode_index = 0

        # Color theme
        self.text_bg_color = (0, 0, 0)
        self.text_color = (0, 255, 0)

        # Define the control instructions
        self.controls = [
            ("Right Arrow", "Exit Early"),
            ("Left Arrow", "Rerecord"),
            ("Escape", "Stop"),
            ("Space", "Toggle Reward"),
        ]

        self.zmq_context = zmq.Context()

        self.publisher_socket = self.zmq_context.socket(zmq.PUB)
        self.publisher_socket.bind("tcp://127.0.0.1:5555")

        self.command_sub_socket = self.zmq_context.socket(zmq.SUB)
        self.command_sub_socket.connect("tcp://127.0.0.1:5556")

        # Subscribe to all messages
        self.command_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    def calculate_window_size(self, images: Dict[str, np.ndarray]):
        """Calculate required window size based on images"""
        max_width = 0
        max_height = 0
        n_images = len(images)

        # Calculate grid dimensions
        grid_cols = min(2, n_images)
        grid_rows = (n_images + 1) // 2

        for image in images.values():
            max_width = max(max_width, image.shape[1])
            max_height = max(max_height, image.shape[0])

        total_width = max_width * grid_cols
        total_height = max_height * grid_rows + self.title_height

        return total_width, total_height, grid_cols

    def draw_top_bar(self, window_width: int):
        top_text_str = f"Mode: {self.config.control_phase}"
        if self.config.control_phase == ControlPhase.RECORD:
            top_text_str += f" | Episode: {self.current_episode_index}/{self.config.num_episodes}"
            if self.config.assign_rewards:
                next_reward = self.events["next_reward"]
                top_text_str += f" | Reward: {next_reward}"

        top_text = self.font.render(top_text_str, True, self.text_color)
        text_rect = top_text.get_rect(center=(window_width // 2, self.title_height // 2))
        self.screen.blit(top_text, text_rect)

    def handle_events(self):
        """Handle pygame events and update internal state"""
        for event in pygame.event.get():
            if self.config.debug_mode:
                print(event)
            if event.type == pygame.QUIT:
                self.events["stop_recording"] = True
                self.events["exit_early"] = True
            elif event.type == pygame.KEYDOWN:
                key_name = pygame.key.name(event.key)
                self.pressed_keys.append(key_name)

                if event.key == pygame.K_RIGHT:
                    print("Right arrow key pressed. Exiting loop...")
                    self.events["exit_early"] = True
                elif event.key == pygame.K_LEFT:
                    print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                    self.events["rerecord_episode"] = True
                    self.events["exit_early"] = True
                elif event.key == pygame.K_ESCAPE:
                    print("Escape key pressed. Stopping data recording...")
                    self.events["stop_recording"] = True
                    self.events["exit_early"] = True
                elif self.config.assign_rewards and event.key == pygame.K_SPACE:
                    self.events["next_reward"] = 1 if self.events["next_reward"] == 0 else 0
                    print(f"Space key pressed. New reward: {self.events['next_reward']}")

            elif event.type == pygame.KEYUP:
                key_name = pygame.key.name(event.key)
                if key_name in self.pressed_keys:
                    self.pressed_keys.remove(key_name)

        return self.events
    
    def handle_browser_events(self, key_pressed: str):
        """
        Translate a key pressed in the web UI to the same event logic used in Pygame.
        """

        try:
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

        except zmq.Again:
            pass
        except Exception as e:
            print(f"Error while polling for commands: {e}")

    def publish_observations(self, observation: Dict[str, np.ndarray]):
        """
        Encode and publish the full observation object via ZeroMQ PUB socket.
        Includes observation data, events, and config information.
        
        Args:
            observation (Dict[str, np.ndarray]): Dictionary containing observation data,
                including images and state information
        """
        processed_data = {}
        
        # Process observation data
        for key, value in observation.items():
            if "image" in key:
                # Handle image data
                image = value.numpy() if torch.is_tensor(value) else value
                # Convert from RGB to BGR for JPEG encoding
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                success, buffer = cv2.imencode(".jpg", bgr_image)
                if success:
                    # Convert to base64
                    b64_jpeg = base64.b64encode(buffer).decode("utf-8")
                    processed_data[key] = {
                        "type": "image",
                        "encoding": "jpeg_base64",
                        "data": b64_jpeg,
                        "shape": image.shape
                    }
            else:
                tensor_data = value.detach().cpu().numpy() if torch.is_tensor(value) else value
                    
                processed_data[key] = {
                    "type": "tensor",
                    "data": tensor_data.tolist(),
                    "shape": tensor_data.shape
                }
        
        # Add events and config information
        events_data = self.get_events()
        config_data = {
            "display_cameras": self.config.display_cameras,
            "play_sounds": self.config.play_sounds,
            "assign_rewards": self.config.assign_rewards,
            "debug_mode": self.config.debug_mode,
            "control_phase": self.config.control_phase,
            "num_episodes": self.config.num_episodes,
            "current_episode": self.current_episode_index
        }
        
        message = {
            "type": "observation_update",
            "timestamp": time.time(),
            "data": processed_data,
            "events": events_data,
            "config": config_data
        }
        
        # Send JSON over ZeroMQ
        self.publisher_socket.send_json(message)

    def render_scene_from_observations(self, observation: Dict[str, np.ndarray]):
        """Render in a Pygame window AND publish frames via ZeroMQ."""
        image_keys = [key for key in observation if "image" in key]
        images = {k: observation[k].numpy() for k in image_keys}
        if not images:
            return

        if self.config.display_cameras:
            window_width, window_height, grid_cols = self.calculate_window_size(images)
            if self.screen is None or self.screen.get_size() != (window_width, window_height):
                self.screen = pygame.display.set_mode((window_width, window_height))
                pygame.display.set_caption("LeRobot")

            self.screen.fill(self.text_bg_color)

            # Update image positions and draw images
            for idx, (key, image) in enumerate(images.items()):
                col = idx % grid_cols
                row = idx // grid_cols
                x = col * (image.shape[1] + self.padding)
                y = row * (image.shape[0] + self.title_height + self.padding)

                image_surface = pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
                self.screen.blit(image_surface, (x, y + self.title_height))

                camera_label_text = key.split(".")[-1]
                camera_label = self.font.render(camera_label_text, True, self.text_color)
                self.screen.blit(camera_label, (x + 5, y + self.title_height + 5))

            pygame.draw.rect(self.screen, self.text_bg_color, (0, 0, window_width, self.title_height))
            self.draw_top_bar(window_width)
            pygame.display.flip()

    def update_with_observations(self, observation: Dict[str, np.ndarray]):
        self.render_scene_from_observations(observation)
        self.publish_observations(observation)

        self.handle_events()
        self.handle_browser_events()
        return self

    def update_current_episode(self, episode_index):
        self.current_episode_index = episode_index
        return self

    def get_events(self):
        return self.events.copy()

    def cleanup(self, robot):
        robot.disconnect()
        pygame.quit()
        # Clean up ZMQ socket
        self.publisher_socket.close()
        self.zmq_context.term()


if __name__ == "__main__":
    import torch
    import cv2
    import time
    import numpy as np

    def read_image_from_camera(cap):
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return torch.tensor(frame_rgb).float()

    config = ControlContextConfig(
        display_cameras=False,
        assign_rewards=True,
        debug_mode=True,
        control_phase=ControlPhase.RECORD,
        num_episodes=200,
    )
    context = ControlContext(config)
    context.update_current_episode(199)

    cameras = {"main": cv2.VideoCapture(0), "top": cv2.VideoCapture(4), "web": cv2.VideoCapture(6)}

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
        context.update_with_observations(obs_dict)
        events = context.get_events()

        if events["exit_early"]:
            break

    for cap in cameras.values():
        cap.release()
    cv2.destroyAllWindows()
