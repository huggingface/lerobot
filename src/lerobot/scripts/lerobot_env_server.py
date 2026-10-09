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

"""Run an environment server from an explicit YAML configuration."""

import argparse
import logging
import signal

from lerobot.env_server.configuration import load_config
from lerobot.env_server.server import EnvServer, ServerConfig
from lerobot.sims.backend import BackendConfig
from lerobot.transport.zenoh import ZenohConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    data = load_config(args.config, args.set)
    data["sim"] = BackendConfig(**data["sim"])
    if "zenoh" in data:
        data["zenoh"] = ZenohConfig(**data["zenoh"])
    logging.basicConfig(level=logging.INFO)
    server = EnvServer(ServerConfig(**data))
    signal.signal(signal.SIGTERM, lambda *_: server.stop.set())
    try:
        server.serve()
    except KeyboardInterrupt:
        server.stop.set()


if __name__ == "__main__":
    main()
