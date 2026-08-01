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
"""Every `lerobot-wandb` command in the showcase README parses against the real CLI.

Documentation that drifts from its CLI is worse than no documentation: a reader pastes a command
that no longer exists and concludes the tool is broken. This is the cheap mechanical half of
"verified to actually work" — it cannot prove a command does the right thing against live W&B, but
it does prove every flag shown still exists, is still spelled that way, and is still accepted
together with the others. Renaming or removing a flag fails here until the README is updated.
"""

import re
import shlex
from pathlib import Path

import pytest

pytest.importorskip("wandb", reason="wandb is required (install lerobot[training])")

from lerobot.integrations.wandb_artifacts import cli

README = Path(__file__).parents[3] / "examples" / "wandb_showcase" / "README.md"

# A fenced bash block, then every backslash-continued command inside it that starts with the CLI
# under test. Other tools shown in the README (lerobot-record, lerobot-train, lerobot-rollout) are
# parsed by draccus from a much larger config surface and are deliberately out of scope here.
_BASH_BLOCK = re.compile(r"```bash\n(.*?)```", re.S)


def _readme_commands() -> list[str]:
    commands = []
    for block in _BASH_BLOCK.findall(README.read_text()):
        for command in block.replace("\\\n", " ").splitlines():
            command = command.strip()
            if command.startswith("lerobot-wandb "):
                commands.append(" ".join(command.split()))
    return commands


def test_the_readme_actually_contains_commands():
    """Guard the guard: a regex that silently matches nothing would make every test below vacuous."""
    commands = _readme_commands()
    assert len(commands) >= 3
    # The pipeline is only end-to-end if every stage that crosses machines is shown. Promotion is
    # deliberately not here: the CLI can only log a new version, never promote an existing one.
    joined = " ".join(commands)
    for expected in ("dataset upload", "model download", "rollout upload"):
        assert expected in joined


@pytest.mark.parametrize("command", _readme_commands(), ids=lambda c: " ".join(c.split()[:3]))
def test_readme_command_parses_against_the_real_cli(command):
    args = cli.build_parser().parse_args(shlex.split(command)[1:])
    assert callable(args.func)


def _readme_train_command() -> list[str]:
    for block in _BASH_BLOCK.findall(README.read_text()):
        for command in block.replace("\\\n", " ").splitlines():
            if command.strip().startswith("lerobot-train "):
                return shlex.split(" ".join(command.split()))[1:]
    raise AssertionError("the showcase README no longer shows a lerobot-train command")


def test_readme_train_command_parses_and_validates(tmp_path):
    """The training command is the one place the README drives *this effort's* config surface
    (`dataset.artifact_ref`, `wandb.model_artifact_name`, ...) rather than the standalone CLI, so it
    is parsed and validated rather than only eyeballed.
    """
    pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
    import draccus

    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.act.configuration_act import ACTConfig  # noqa: F401  (registers act)

    # Only environment-dependent values may be substituted here. Anything that decides whether the
    # command is *valid* must not be: injecting `--policy.push_to_hub=false` to get a green test is
    # how the first version of this file hid a README command that could not run at all.
    args = [
        arg for arg in _readme_train_command() if not arg.startswith(("--output_dir=", "--policy.device="))
    ]
    args += [f"--output_dir={tmp_path / 'run'}", "--policy.device=cpu"]

    cfg = draccus.parse(TrainPipelineConfig, args=args)
    cfg.validate()

    assert cfg.dataset.artifact_ref == "my-team/so101-pick-cube/pick-cube:raw"
    assert cfg.dataset.repo_id is None
    assert cfg.wandb.model_artifact_name == "pick-cube-policy"
    assert cfg.wandb.registered_model_name == "pick-cube-policy"
    # The showcase promises W&B is the only remote store; `push_to_hub` defaults to True.
    assert cfg.policy.push_to_hub is False


def _readme_command_for(tool: str) -> list[str]:
    for block in _BASH_BLOCK.findall(README.read_text()):
        for command in block.replace("\\\n", " ").splitlines():
            if command.strip().startswith(f"{tool} "):
                return shlex.split(" ".join(command.split()))[1:]
    raise AssertionError(f"the showcase README no longer shows a {tool} command")


def test_readme_record_command_parses():
    """`lerobot-record` is upstream, not built here, but a stale flag in our README misleads the
    reader just as badly. Parsed with plain draccus — it declares no `__get_path_fields__` values
    that the CLI wrapper would have to strip first.
    """
    pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
    import draccus

    from lerobot.scripts.lerobot_record import RecordConfig

    cfg = draccus.parse(RecordConfig, args=_readme_command_for("lerobot-record"))

    assert cfg.dataset.repo_id == "local/pick-cube"
    assert cfg.dataset.push_to_hub is False  # the showcase never touches the Hub


def test_readme_rollout_command_uses_a_rollout_prefixed_dataset_name():
    """`lerobot-rollout` resolves `--policy.path` through `parser.wrap` rather than plain draccus,
    so parsing it here would need a real checkpoint on disk. Check instead the one rule the rollout
    config enforces about the command shown, which a reader would otherwise hit at runtime.
    """
    args = _readme_command_for("lerobot-rollout")
    repo_id = next(a.split("=", 1)[1] for a in args if a.startswith("--dataset.repo_id="))

    assert repo_id.split("/", 1)[-1].startswith("rollout_")
    # `DatasetRecordConfig.push_to_hub` defaults to True and the episodic strategy's teardown acts
    # on it, so without this flag the rollout is published to the Hub behind the reader's back.
    assert "--dataset.push_to_hub=false" in args
    # The rollout upload command must point at the directory this command writes.
    root = next(a.split("=", 1)[1] for a in args if a.startswith("--dataset.root="))
    upload = next(c for c in _readme_commands() if c.startswith("lerobot-wandb rollout upload"))
    assert f"--root {root}" in upload
