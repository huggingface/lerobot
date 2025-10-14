from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.policy_server import serve

host = ...  # something like "127.0.0.1" if you're exposing to localhost
port = ...  # something like 8080

config = PolicyServerConfig(
    host=host,
    port=port,
)
serve(config)
