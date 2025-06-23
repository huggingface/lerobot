"""Server/Client side: Sometimes you just want the environment to wait a tiny bit"""

DEFAULT_IDLE_WAIT = 0.01

"""Client side: The environment evolves with a time resolution equal to environment_dt"""
DEFAULT_ENVIRONMENT_DT = 1 / 30

"""Server side: Running inference on (at most) environment_dt"""
DEFAULT_INFERENCE_LATENCY = DEFAULT_ENVIRONMENT_DT

"""Server side: Timeout for observation queue in seconds"""
DEFAULT_OBS_QUEUE_TIMEOUT = 2

# TODO:  Support DiffusionPolicy, VQBet and Pi0
supported_policies = ["act", "smolvla"]

# TODO: Add all other robots
supported_robots = ["so100"]
