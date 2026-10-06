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
"""Policy registration on demand, checked in fresh interpreters.

Test modules register policies as they are collected, so these run each case in a new Python process,
the way a command runs.
"""

import json
import re
import subprocess
import sys

import pytest

from lerobot.utils.import_utils import _datasets_available

pytestmark = pytest.mark.skipif(not _datasets_available, reason="lerobot-train needs the dataset extra")

PARSE = """
import sys
import lerobot.scripts.lerobot_train
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig

sys.argv = ["lerobot-train", *sys.argv[1:]]

@parser.wrap()
def main(cfg: TrainPipelineConfig):
    loaded = sorted({m.split(".")[2] for m in sys.modules if m.startswith("lerobot.policies.")})
    print("POLICY", type(cfg.policy).__name__, cfg.policy.device)
    print("OPTIMIZER", cfg.optimizer)
    print("SCHEDULER", type(cfg.scheduler).__name__)
    print("LOADED", ",".join(loaded))

main()
"""


NESTED_PARSE = """
import sys
from lerobot.configs import parser
from lerobot.rl.train_rl import TrainRLServerPipelineConfig

sys.argv = ["lerobot-train-rl", *sys.argv[1:]]

@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig):
    print("NESTED", type(cfg.algorithm.policy_config).__name__)

main()
"""

PLUGIN = """
from dataclasses import dataclass
from lerobot.configs.policies import PreTrainedConfig

try:
    @PreTrainedConfig.register_subclass("act")
    @dataclass
    class PluginACTConfig(PreTrainedConfig):
        pass
except ValueError as e:
    print("REJECTED", e)
print("ACT", PreTrainedConfig.get_choice_class("act").__name__)
"""

UNKNOWN_NAMES = """
import json, sys, tempfile
sys.modules["transformers"] = None  # pi_gemma, a helper module, fails to import without it
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_policy_config

config_dir = tempfile.mkdtemp()
for name in ["pi_gemma", "nope", 5, None]:
    with open(f"{config_dir}/config.json", "w") as f:
        json.dump({"type": name}, f)
    from_config = (PreTrainedConfig.from_pretrained, config_dir)
    for lookup, arg in [(make_policy_config, name), (get_policy_class, name), from_config]:
        try:
            lookup(arg)
        except ValueError:
            print("VALUEERROR", lookup.__name__, name)
"""


def run_python(code: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, timeout=300)


def run_train_parse(*args: str) -> subprocess.CompletedProcess:
    return run_python(PARSE, *args)


def test_policy_type_loads_only_that_policy():
    result = run_train_parse("--policy.type=act", "--policy.device=cpu", "--dataset.repo_id=u/d")
    assert result.returncode == 0, result.stderr
    assert "POLICY ACTConfig cpu" in result.stdout
    loaded = result.stdout.split("LOADED ")[1].split()[0].split(",")
    assert "act" in loaded
    assert "diffusion" not in loaded


def test_override_with_type_from_a_config_directory(tmp_path):
    (tmp_path / "train_config.json").write_text(
        json.dumps({"dataset": {"repo_id": "u/d"}, "policy": {"type": "act", "push_to_hub": False}})
    )
    result = run_train_parse(f"--config_path={tmp_path}", "--policy.device=cpu")
    assert result.returncode == 0, result.stderr
    assert "POLICY ACTConfig cpu" in result.stdout


def test_optimizer_registered_by_another_policy():
    result = run_train_parse(
        "--policy.type=act",
        "--policy.device=cpu",
        "--dataset.repo_id=u/d",
        "--use_policy_training_preset=false",
        "--optimizer.type=molmoact2_adamw",
    )
    assert result.returncode == 0, result.stderr
    assert "OPTIMIZER MolmoAct2AdamWConfig" in result.stdout


def test_optimizer_registered_by_another_policy_in_a_config_file(tmp_path):
    config = {
        "dataset": {"repo_id": "u/d"},
        "policy": {"type": "act", "device": "cpu", "push_to_hub": False},
        "use_policy_training_preset": False,
        "optimizer": {"type": "molmoact2_adamw", "lr": 1e-5},
    }
    (tmp_path / "train_config.json").write_text(json.dumps(config))
    result = run_train_parse(f"--config_path={tmp_path / 'train_config.json'}")
    assert result.returncode == 0, result.stderr
    assert "OPTIMIZER MolmoAct2AdamWConfig" in result.stdout


@pytest.mark.parametrize("name", ["nope", "foo.bar"])
def test_unknown_policy_type_lists_every_policy(name):
    result = run_train_parse(f"--policy.type={name}", "--dataset.repo_id=u/d")
    error = result.stderr.strip().splitlines()[-1]
    assert result.returncode == 2
    assert "invalid choice" in error
    # Python versions differ on whether argparse quotes the choices.
    assert {"act", "diffusion", "xvla"} <= set(re.findall(r"\w+", error.split("choose from", 1)[1]))


def test_saved_pipeline_with_a_policy_step_loads(tmp_path):
    save = (
        "from lerobot.policies.pi0.processor_pi0 import Pi0NewLineProcessor\n"
        "from lerobot.processor import DataProcessorPipeline\n"
        f"DataProcessorPipeline([Pi0NewLineProcessor()]).save_pretrained({str(tmp_path)!r})\n"
    )
    load = (
        "from lerobot.processor import DataProcessorPipeline\n"
        f"p = DataProcessorPipeline.from_pretrained({str(tmp_path)!r}, config_filename='dataprocessorpipeline.json')\n"
        "print('STEPS', [type(s).__name__ for s in p.steps])\n"
    )
    for code in (save, load):
        result = run_python(code)
        assert result.returncode == 0, result.stderr
    assert "STEPS ['Pi0NewLineProcessor']" in result.stdout


def test_repeated_policy_type_uses_the_last_one():
    result = run_train_parse(
        "--policy.type=act", "--policy.device=cpu", "--dataset.repo_id=u/d", "--policy.type=diffusion"
    )
    assert result.returncode == 0, result.stderr
    assert "POLICY DiffusionConfig cpu" in result.stdout


def test_override_of_a_field_only_a_policy_optimizer_has(tmp_path):
    config = {
        "dataset": {"repo_id": "u/d"},
        "policy": {"type": "act", "device": "cpu", "push_to_hub": False},
        "use_policy_training_preset": False,
        "optimizer": {"type": "molmoact2_adamw"},
    }
    (tmp_path / "train_config.json").write_text(json.dumps(config))
    result = run_train_parse(
        f"--config_path={tmp_path / 'train_config.json'}", "--optimizer.group_grad_clip_norm=2.5"
    )
    assert result.returncode == 0, result.stderr
    assert "OPTIMIZER MolmoAct2AdamWConfig(" in result.stdout
    assert "group_grad_clip_norm=2.5" in result.stdout


def test_scheduler_registered_by_another_policy():
    result = run_train_parse(
        "--policy.type=act",
        "--policy.device=cpu",
        "--dataset.repo_id=u/d",
        "--use_policy_training_preset=false",
        "--optimizer.type=adamw",
        "--scheduler.type=molmoact2_cosine_with_warmup",
        "--scheduler.num_warmup_steps=1",
        "--scheduler.num_decay_steps=10",
        "--scheduler.peak_lr=1e-4",
        "--scheduler.decay_lr=1e-6",
    )
    assert result.returncode == 0, result.stderr
    assert "SCHEDULER MolmoAct2CosineWithWarmupSchedulerConfig" in result.stdout


def test_help_lists_every_policy():
    result = run_train_parse("--help")
    assert result.returncode == 0, result.stderr
    choices = re.search(r"--policy\.type \{([^}]*)\}", result.stdout)
    assert choices, result.stdout
    assert {"act", "diffusion", "xvla"} <= {name.strip() for name in choices.group(1).split(",")}


@pytest.mark.parametrize(
    "args", [["--algorithm.policy_config.type=act"], ["--algorithm.policy_config.type", "act"]]
)
def test_nested_policy_type(args):
    result = run_python(
        NESTED_PARSE, "--algorithm.type=sac", *args, "--policy.type=gaussian_actor", "--policy.device=cpu"
    )
    assert result.returncode == 0, result.stderr
    assert "NESTED ACTConfig" in result.stdout


def test_policy_class_resolves_in_a_fresh_process():
    result = run_python(
        "from lerobot.policies.factory import get_policy_class; print(get_policy_class('act').__name__)"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-1] == "ACTPolicy"


def test_plugin_cannot_take_a_builtin_policy_name():
    result = run_python(PLUGIN)
    assert result.returncode == 0, result.stderr
    assert "REJECTED" in result.stdout
    assert "ACT ACTConfig" in result.stdout


def test_unknown_or_non_string_policy_type_raises_value_error():
    result = run_python(UNKNOWN_NAMES)
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.split("VALUEERROR")) - 1 == 12, result.stdout
