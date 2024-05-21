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
        self.node = Node()
        self.observation = {"pixels": {}, "terminated": False}

    def _update(self) -> dict: 
        while True:
            event = self.node.next(timeout=0.001)

            if event is None:
                self.observation["terminated"] = True
                break
            if event["type"] == "INPUT":
                if "cam" in event["id"]:
                    self.observation["pixels"][event["id"]] = event["value"].to_numpy().reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
                else:
                    self.observation[event["id"]] = event["value"].to_numpy()
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
        self.node.send_output("action", pa.array(action))
        reward = 0
        terminated = truncated = self.observation["terminated"]
        info = {}
        return self.observation, reward, terminated, truncated, info

    def close(self):
        del self.node

