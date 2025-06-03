"""Server/Client side: Sometimes you just want the environment to wait a tiny bit"""

idle_wait = 0.01

"""Client side: The environment evolves with a time resolution equal to environment_dt"""
environment_dt = 1 / 30

"""Server side: Running inference on (at most) environment_dt"""
inference_latency = environment_dt

"""Supported policies"""
supported_policies = ["act", "smolvla"]
