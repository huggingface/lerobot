# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Delay one live action call for the H2 starvation check, preserving IO and deadlines.

Replace only the server command; keep the known-good client/configuration:
    uv run --no-sync python tests/manual/delayed_policy_server.py \
        --delay-s=0.35 --on-action=3 -- --config_path=server.yaml

Use a delay above remaining playback but below the request deadline. Recovery also
needs old-work cleanup and fresh inference to fit the client's starvation grace.
Restart this launcher to rearm; warmup, control and language calls are unaffected.
"""

import argparse
import logging
import math
import sys
import time
from typing import Any

from lerobot.inference import ActionChunk, ObservationSnapshot
from lerobot.remote_inference import PolicyServer, SessionWorker

logger = logging.getLogger(__name__)


def delay_action_once(worker: SessionWorker, *, delay_s: float, on_action: int) -> None:
    """Instrument this test server's serialized worker, leaving its IO thread running."""
    if not math.isfinite(delay_s) or delay_s <= 0 or on_action < 2:
        raise ValueError("delay_s must be positive and finite; on_action must be >= 2")
    predict = worker.runner.predict
    calls = 0

    def delayed(observation: ObservationSnapshot, **kwargs: Any) -> ActionChunk:
        nonlocal calls
        calls += 1
        if calls == on_action:
            logger.warning(
                "MANUAL TEST: delaying action %d by %.3f s; observation=%s, deadlines unchanged",
                calls,
                delay_s,
                observation.observation_id,
            )
            time.sleep(delay_s)
            logger.warning("MANUAL TEST: one-shot action delay completed")
        return predict(observation, **kwargs)

    worker.runner.predict = delayed


def main() -> None:
    from lerobot.scripts import lerobot_policy_server

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay-s", type=float, required=True)
    parser.add_argument("--on-action", type=int, default=3)
    args, server_args = parser.parse_known_args()
    if not math.isfinite(args.delay_s) or args.delay_s <= 0 or args.on_action < 2:
        parser.error("--delay-s must be positive and finite; --on-action must be >= 2")
    if server_args[:1] == ["--"]:
        server_args = server_args[1:]

    class DelayedPolicyServer(PolicyServer):
        def serve(self) -> None:
            delay_action_once(self.worker, delay_s=args.delay_s, on_action=args.on_action)
            super().serve()

    lerobot_policy_server.PolicyServer = DelayedPolicyServer
    sys.argv = [sys.argv[0], *server_args]
    lerobot_policy_server.main()


if __name__ == "__main__":
    main()
