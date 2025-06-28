"""Server/Client side: Sometimes you just want the environment to wait a tiny bit"""

DEFAULT_IDLE_WAIT = 0.01

"""Client side: The environment evolves with a time resolution equal to 1/fps"""
DEFAULT_FPS = 30

"""Server side: Running inference on (at most) 1/fps"""
DEFAULT_INFERENCE_LATENCY = 1 / DEFAULT_FPS

"""Server side: Timeout for observation queue in seconds"""
DEFAULT_OBS_QUEUE_TIMEOUT = 2

# All action chunking policies
supported_policies = ["act", "smolvla", "diffusion", "pi0", "tdmpc", "vqbet"]

# TODO: Add all other robots
supported_robots = ["so100"]
