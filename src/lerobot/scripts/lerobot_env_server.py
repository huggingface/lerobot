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

"""Run a simulator server from a YAML configuration or an EnvHub repository.

lerobot-env-server --docker --config=docker/sims/configs/libero.yaml   # in its Docker image, on :7448
lerobot-env-server --docker --envhub=<user/repo[@revision]>            # serve an EnvHub repository
lerobot-env-server --config=docker/sims/configs/metaworld.yaml         # in this environment
"""

import argparse
import logging
import signal

from lerobot.env_server.configuration import envhub_config, load_config
from lerobot.env_server.launcher import run_in_docker
from lerobot.env_server.server import EnvServer, ServerConfig
from lerobot.sims.backend import BackendConfig
from lerobot.sims.conventions import resolve
from lerobot.transport.zenoh import ZenohConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", help="Simulator YAML; sim.type selects the backend and its Docker image")
    parser.add_argument(
        "--envhub",
        metavar="REPO",
        help="Serve an EnvHub repository (user/repo[@revision], or a local directory); trusts and runs its env.py",
    )
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="Override an existing dotted YAML key"
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Run the server in its Docker image (built from this LeRobot checkout), on localhost:7448",
    )
    parser.add_argument(
        "--build", action="store_true", help="With --docker: rebuild the image (it is built when missing)"
    )
    parser.add_argument("--stop", action="store_true", help="With --docker: stop the server")
    args = parser.parse_args()
    if (args.build or args.stop) and not args.docker:
        parser.error("--build and --stop apply to --docker")
    if args.envhub:
        data = envhub_config(args.envhub, args.set, args.config)
    elif args.config:
        data = load_config(args.config, args.set)
    else:
        parser.error("Pass --config or --envhub")
    if args.docker:
        try:
            run_in_docker(data, build=args.build, stop=args.stop)
        except (RuntimeError, ValueError) as error:
            parser.error(str(error))
        return
    # force: an imported module may already have installed a WARNING-level root handler.
    logging.basicConfig(level=logging.INFO, force=True)
    # Settings left null come from the simulator's conventions (an EnvHub repository's LEROBOT_SIM).
    data["sim"] = BackendConfig(**resolve(data["sim"]))
    if "zenoh" in data:
        data["zenoh"] = ZenohConfig(**data["zenoh"])
    server = EnvServer(ServerConfig(**data))
    shutdown_requested = False

    def request_shutdown(_signum: int, _frame: object) -> None:
        # A signal can interrupt Event.wait while its non-reentrant lock is held.
        nonlocal shutdown_requested
        shutdown_requested = True

    previous_handler = signal.signal(signal.SIGTERM, request_shutdown)
    try:
        server.serve(should_stop=lambda: shutdown_requested)
    except KeyboardInterrupt:
        server.stop.set()
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
