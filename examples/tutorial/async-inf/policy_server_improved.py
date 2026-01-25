"""
Latency-Adaptive Async Inference Policy Server Example

This example demonstrates how to run the improved policy server with:
- 2-thread architecture (observation receiver + main inference loop)
- SPSC one-slot mailbox for observation queue

Usage:
    python examples/tutorial/async-inf/policy_server_improved.py
    python examples/tutorial/async-inf/policy_server_improved.py --host 0.0.0.0 --port 8080

To expose it to your local network, bind to 0.0.0.0 and connect from other machines
using this machine's LAN IP (e.g. 192.168.x.y), not 0.0.0.0.
"""

import argparse
import faulthandler
import logging
import os
import sys

from lerobot.async_inference.policy_server_improved import (
    PolicyServerImprovedConfig,
    serve_improved,
)

faulthandler.enable(file=sys.stderr, all_threads=True)


def main() -> None:
    debug = os.environ.get("LEROBOT_DEBUG", "0") == "1"
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run the Latency-Adaptive Async Inference Policy Server"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help='Host/interface to bind to. Use "127.0.0.1" for local-only.',
    )
    parser.add_argument(
        "--port",
        type=int,
        default=18080,
        help="Port to bind to.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Control frequency in Hz (should match client).",
    )
    parser.add_argument(
        "--obs-queue-timeout",
        type=float,
        default=2.0,
        help="Timeout for observation queue in seconds.",
    )
    args = parser.parse_args()

    config = PolicyServerImprovedConfig(
        host=args.host,
        port=args.port,
        fps=args.fps,
        obs_queue_timeout=args.obs_queue_timeout,
    )

    try:
        serve_improved(config)
    except Exception:
        logging.exception("Policy server example runner crashed")
        raise


if __name__ == "__main__":
    main()

