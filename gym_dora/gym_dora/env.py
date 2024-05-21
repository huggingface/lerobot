import gymnasium as gym
import numpy as np
from dora import Node
import os
import pyarrow as pa

IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "640"))
IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "480"))
FPS = int(os.getenv("FPS", "30"))

class DoraEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": FPS}

    def __init__(self, model="aloha"):
        # Initialize a new node
        self.node = Node()
        self.observation = {"pixels": {}, "terminated": False}

    def _update(self) -> dict: 
        while True:
            event = self.node.next(timeout=0.001)

            ## If event is None, the node event stream is closed and we should terminate the env
            if event is None:
                self.observation["terminated"] = True
                break

            if event["type"] == "INPUT":
                # Map Image input into pixels key within Aloha environment
                if "cam" in event["id"]:
                    self.observation["pixels"][event["id"]] = event["value"].to_numpy().reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
                else:
                    # Map other inputs into the observation dictionary using the event id as key
                    self.observation[event["id"]] = event["value"].to_numpy()
            
            # If the event is a timeout error break the update loop.
            elif event["type"] == "ERROR":
                break
        

    def reset(self, seed: int | None = None):

        self._update()
        
        reward = 0
        terminated = truncated = self.observation["terminated"]
        info = {}
        return self.observation, reward, terminated, truncated, info

    def render(self): ...

    def step(self, action: np.ndarray):
        self._update()

        # Send the action to the dataflow as action key.
        self.node.send_output("action", pa.array(action))
        reward = 0
        terminated = truncated = self.observation["terminated"]
        info = {}
        return self.observation, reward, terminated, truncated, info

    def close(self):

        # Drop the node
        del self.node

