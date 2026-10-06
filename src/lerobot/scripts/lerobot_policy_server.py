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

"""Load, validate and warm one pinned deployment, then serve one robot session."""

import logging
import signal
from collections.abc import Callable
from dataclasses import asdict
from typing import cast

from lerobot.configs import parser
from lerobot.remote_inference import (
    PROTOCOL_VERSION,
    SOFTWARE_BUILD,
    PolicyServer,
    ServerConfig,
    SessionWorker,
    load_deployment,
)
from lerobot.transport.zenoh import ZenohTransport
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


@parser.wrap()
def serve(cfg: ServerConfig) -> None:
    """Serve the configured deployment until an operator terminates the process."""
    init_logging(console_level=cfg.log_level)
    logger.info("Policy server software=%s protocol=%s", asdict(SOFTWARE_BUILD), PROTOCOL_VERSION)
    logger.info(
        "Loading deployment=%s model=%s revision=%s device=%s; readiness follows model warmup",
        cfg.deployment,
        cfg.model.repo_or_path,
        cfg.model.revision or "default",
        cfg.model.device,
    )
    runner, identity = load_deployment(cfg)
    worker = SessionWorker(
        runner,
        deployment=cfg.deployment,
        artifact_identity=identity,
        semantics=cfg.semantics,
        action_deadline_s=cfg.execution.action_deadline_s,
        language_deadline_s=cfg.language.deadline_s,
        idle_timeout_s=cfg.execution.idle_timeout_s,
        max_input_chars=cfg.language.max_input_chars,
        max_output_chars=cfg.language.max_output_chars,
        blendable_components=tuple(cfg.execution.blendable_components),
    )
    server = PolicyServer(worker, ZenohTransport(cfg.zenoh))
    signal.signal(signal.SIGTERM, lambda signum, _: server.stop(reason=signal.Signals(signum).name))
    signal.signal(signal.SIGINT, lambda signum, _: server.stop(reason=signal.Signals(signum).name))
    logger.info(
        "Deployment warmed: name=%s instance=%s modes=%s action_rate=%.1f Hz horizon=%.3fs "
        "language=%s; waiting for transport readiness",
        cfg.deployment,
        worker.instance_id,
        ",".join(mode.value for mode in runner.capabilities.modes),
        1 / runner.capabilities.action_interval,
        runner.capabilities.execution_steps * runner.capabilities.action_interval,
        runner.capabilities.language,
    )
    logger.debug("Deployment artifact=%s capabilities=%s", identity, runner.capabilities)
    server.serve()


def main() -> None:
    """Register optional policy plugins and parse the server CLI."""
    register_third_party_plugins()
    cast(Callable[[], None], serve)()


if __name__ == "__main__":
    main()
