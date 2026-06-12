#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serve a pretrained policy to remote ``lerobot-rollout`` clients over Zenoh.

One process = one pre-warmed (model, revision, dtype, device) on one GPU.
Robots connect with ``lerobot-rollout --inference.type=remote``.

Usage examples
--------------

Serve a model from a YAML manifest::

    lerobot-policy-server --manifest server.yaml

Minimal manifest::

    model:
      repo_or_path: lerobot/pi0_towels
      device: cuda
    default_task: "fold the towel"
    max_sessions: 5
    zenoh:
      mode: client
      connect_endpoints: ["tcp/router.gpu-cluster.internal:7447"]

Everything in the manifest can also be set directly on the CLI::

    lerobot-policy-server \\
        --model.repo_or_path=lerobot/pi0_towels \\
        --default_task="fold the towel" \\
        --zenoh.mode=peer --zenoh.listen_endpoints='["tcp/0.0.0.0:7447"]'

SIGTERM/SIGINT drains gracefully: the server drops its liveliness token
(clients ride their action buffers through the drain), finishes the
in-flight inference, and exits.
"""

import logging
import signal
import sys

from lerobot.configs import parser
from lerobot.policy_server.manifest import PolicyServerManifest
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


@parser.wrap()
def policy_server(manifest: PolicyServerManifest):
    init_logging()
    from lerobot.policy_server.server import PolicyServer

    server = PolicyServer(manifest)
    server.load_policy()

    def _drain(signum, frame):  # noqa: ARG001
        logger.info("Signal %s received — draining", signum)
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _drain)
    server.start()
    server.serve_forever()


def main():
    # `--manifest path.yaml` is sugar for draccus's `--config_path`.
    sys.argv = [
        arg.replace("--manifest=", "--config_path=") if arg.startswith("--manifest=") else arg
        for arg in sys.argv
    ]
    if "--manifest" in sys.argv:
        sys.argv[sys.argv.index("--manifest")] = "--config_path"
    policy_server()


if __name__ == "__main__":
    main()
