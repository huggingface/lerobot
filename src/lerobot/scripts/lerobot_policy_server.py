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

import hashlib
import json
import logging
import signal
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from huggingface_hub import snapshot_download

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.inference import ExecutionMode, ObservationSnapshot, PolicyRunner
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import RenderRuntimeMessagesStep
from lerobot.remote_inference import (
    PROTOCOL_VERSION,
    SOFTWARE_BUILD,
    PolicyServer,
    ServerConfig,
    SessionWorker,
)
from lerobot.transport.zenoh import ZenohTransport
from lerobot.utils.import_utils import _peft_available, register_third_party_plugins, require_package
from lerobot.utils.utils import init_logging

if TYPE_CHECKING or _peft_available:
    from peft import PeftConfig, PeftModel

logger = logging.getLogger(__name__)


def resolve_artifact(repo_or_path: str, revision: str | None) -> Path:
    """Hub downloads are operator-configured, resolved once before readiness."""
    path = Path(repo_or_path)
    if path.is_dir():
        return path.resolve()
    return Path(snapshot_download(repo_or_path, revision=revision))


def artifact_identity(paths: dict[str, Path], effective: dict) -> str:
    """Hash model, adapter and processor contents plus effective serving settings."""
    digest = hashlib.sha256(json.dumps(effective, sort_keys=True, default=str).encode())
    for role, root in sorted(paths.items()):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or any(part.startswith(".") for part in path.relative_to(root).parts):
                continue
            # Include named file boundaries and lengths to avoid concatenation ambiguity.
            name = f"{role}/{path.relative_to(root)}".encode()
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            digest.update(path.stat().st_size.to_bytes(8, "big"))
            with path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
    return "sha256:" + digest.hexdigest()


def _inference_identity(cfg: ServerConfig, policy_cfg: PreTrainedConfig) -> dict:
    """Pin checkpoint contents and inference semantics independently of deployment location."""
    policy = asdict(policy_cfg)
    policy.pop("pretrained_path", None)
    policy.pop("device", None)
    uses_rtc = any(mode != ExecutionMode.CHUNK for mode in cfg.execution.supported_modes)
    return {
        "policy": policy,
        "semantics": cfg.semantics,
        "robot_type": cfg.robot_type,
        "features": [asdict(feature) for feature in cfg.features],
        "action_feature": None if cfg.action_feature is None else asdict(cfg.action_feature),
        "execution": {
            "supported_modes": sorted(cfg.execution.supported_modes),
            "action_fps": cfg.execution.action_fps,
            "rtc": asdict(cfg.execution.rtc) if uses_rtc else None,
            "blendable_components": sorted(cfg.execution.blendable_components),
        },
        "language": {
            "enabled": cfg.language.enabled,
            "max_input_chars": cfg.language.max_input_chars,
            "max_output_chars": cfg.language.max_output_chars,
        },
    }


def _subtask_prompt_unavailable_reason(runner: PolicyRunner) -> str | None:
    """Check known saved renderers without treating model failures as missing support.

    Custom language processors still need to pass normal next-subtask warmup.
    The protocol's language flag describes text generation, not every prompt kind.
    """
    if runner.language_processors is None:
        return None
    for step in runner.language_processors[0].steps:
        if not isinstance(step, RenderRuntimeMessagesStep):
            continue
        if step.recipe is None:
            return "the saved runtime message renderer has no checkpoint recipe"
        try:
            step.recipe.prompt_turns("subtask")
        except ValueError as exc:
            # This recipe inspection raises only when its assistant target is absent.
            # Keep processor execution and model generation outside this handler.
            return str(exc)
    return None


def load_deployment(cfg: ServerConfig) -> tuple[PolicyRunner, str]:
    """Resolve contents, load canonical processors and reset all warmed model paths."""
    cfg.zenoh.validate()
    if cfg.action_feature is None:
        raise ValueError("An explicit action feature is required")
    path = resolve_artifact(cfg.model.repo_or_path, cfg.model.revision)
    artifacts = {"checkpoint": path}
    policy_cfg = PreTrainedConfig.from_pretrained(path)
    execution_steps = cfg.execution.n_action_steps
    if execution_steps is not None:
        if "n_action_steps" not in {field.name for field in fields(policy_cfg)}:
            raise ValueError("This checkpoint config has no configurable n_action_steps execution slice")
        prediction_steps = getattr(policy_cfg, "chunk_size", None)
        if isinstance(prediction_steps, int) and execution_steps > prediction_steps:
            raise ValueError("execution.n_action_steps exceeds the checkpoint prediction horizon")
        # Re-run the policy's own config validation without modifying saved files.
        overrides: dict[str, Any] = {"n_action_steps": execution_steps}
        policy_cfg = replace(policy_cfg, **overrides)
    policy_cfg.pretrained_path = path
    policy_cfg.device = cfg.model.device
    modes = tuple(ExecutionMode(mode) for mode in cfg.execution.supported_modes)
    if any(mode != ExecutionMode.CHUNK for mode in modes):
        # Some RTC-capable policy configs obtain this runtime attribute locally
        # rather than declaring a saved field. The runner still verifies support.
        policy_cfg.rtc_config = deepcopy(cfg.execution.rtc)
    policy_class = get_policy_class(policy_cfg.type)
    if policy_cfg.use_peft:
        require_package("peft", extra="peft")
        adapter = PeftConfig.from_pretrained(str(path))
        base = resolve_artifact(adapter.base_model_name_or_path, adapter.revision)
        artifacts["base"] = base
        policy = policy_class.from_pretrained(base, config=policy_cfg)
        policy = cast(PreTrainedPolicy, PeftModel.from_pretrained(policy, path, config=adapter))
    else:
        policy = policy_class.from_pretrained(path, config=policy_cfg)
    policy.to(cfg.model.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=path,
        preprocessor_overrides={
            "device_processor": {"device": cfg.model.device},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    runner = PolicyRunner(
        policy,
        preprocessor,
        postprocessor,
        action_interval=1 / cfg.execution.action_fps,
        features=tuple(cfg.features),
        action_feature=cfg.action_feature,
        modes=modes,
        robot_type=cfg.robot_type,
        language_enabled=cfg.language.enabled,
        max_text_input=cfg.language.max_input_chars,
        max_text_output=cfg.language.max_output_chars,
    )
    if execution_steps is not None and runner.capabilities.execution_steps != execution_steps:
        raise ValueError("Loaded policy does not honor the requested execution.n_action_steps slice")
    identity = artifact_identity(artifacts, _inference_identity(cfg, policy_cfg))
    warmup = ObservationSnapshot(
        {feature.name: np.zeros(feature.shape, dtype=feature.dtype) for feature in cfg.features},
        0.0,
        "warmup",
        observation_id="warmup",
    )
    try:
        for mode in runner.capabilities.modes:
            previous = None
            for _ in range(cfg.execution.warmup_calls):
                previous = runner.predict(
                    warmup,
                    mode=mode,
                    inference_delay=(
                        min(1, runner.capabilities.training_max_delay)
                        if previous is not None and mode is ExecutionMode.RTC_TRAINED
                        else 0
                    ),
                    model_continuation=None if previous is None else previous.model_actions,
                    canonical_continuation=None if previous is None else previous.canonical_actions,
                )
        if cfg.language.enabled:
            # Reset below erases planner warmup state. Unsupported saved subtask
            # prompts must not prevent action + VQA deployments from starting.
            runner.query(warmup, kind="vqa", text="Describe the scene.")
            reason = _subtask_prompt_unavailable_reason(runner)
            if reason is None:
                runner.query(warmup, kind="next_subtask", text="Describe the next task.")
            else:
                logger.warning(
                    "Skipping next-subtask warmup: %s. Actions and VQA remain available; "
                    "do not enable autosteer for this deployment. Use a checkpoint with a saved "
                    "recipe supervising ${subtask} to enable next-subtask generation.",
                    reason.rstrip("."),
                )
    finally:
        runner.reset(full=True)
    return runner, identity


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
