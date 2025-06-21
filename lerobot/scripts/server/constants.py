"""Server/Client side: Sometimes you just want the environment to wait a tiny bit"""

IDLE_WAIT = 0.01

"""Client side: The environment evolves with a time resolution equal to environment_dt"""
ENVIRONMENT_DT = 1 / 30

"""Server side: Running inference on (at most) environment_dt"""
INFERENCE_LATENCY = ENVIRONMENT_DT

"""Supported policies"""
SUPPORTED_POLICIES = ["act", "smolvla"]
