"""
Run a Policy Server.

To expose it to your local network, bind to 0.0.0.0 and connect from other machines
using this machine's LAN IP (e.g. 192.168.x.y), not 0.0.0.0.
"""

import argparse

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.policy_server import serve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help='Host/interface to bind to. Use "127.0.0.1" for local-only.',
    )
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to.")
    args = parser.parse_args()

    config = PolicyServerConfig(host=args.host, port=args.port)
    serve(config)


if __name__ == "__main__":
    main()
