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
from pathlib import Path

import pytest

import lerobot.policies
from lerobot.utils.import_utils import _datasets_available

pytestmark = [
    pytest.mark.dataset,
    pytest.mark.skipif(not _datasets_available, reason="lerobot-train needs the dataset extra"),
]

PARSE = """
import sys
import lerobot.scripts.lerobot_train
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig

sys.argv = ["lerobot-train", *sys.argv[1:]]

@parser.wrap()
def main(cfg: TrainPipelineConfig):
    from lerobot.configs.policies import PreTrainedConfig

    print("POLICY", type(cfg.policy).__name__, cfg.policy.device)
    print("OPTIMIZER", cfg.optimizer)
    print("SCHEDULER", type(cfg.scheduler).__name__)
    print("REGISTERED", ",".join(sorted(PreTrainedConfig.get_known_choices())))

main()
"""


NESTED_PARSE = """
import sys
from lerobot.configs import parser
from lerobot.rl.train_rl import TrainRLServerPipelineConfig

sys.argv = ["lerobot-train-rl", *sys.argv[1:]]

@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig):
    print("NESTED", type(cfg.algorithm.policy_config).__name__, cfg.algorithm.policy_config.n_action_steps)

main()
"""

PLAIN_NESTED = """
import sys
from dataclasses import dataclass, field
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig

@dataclass
class Worker:
    policy: PreTrainedConfig | None = None

@dataclass
class Root:
    worker: Worker = field(default_factory=Worker)

@parser.wrap()
def main(cfg: Root):
    print("NESTED", type(cfg.worker.policy).__name__, getattr(cfg.worker.policy, "chunk_size", None))

sys.argv = ["prog", *sys.argv[1:]]
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

from lerobot.policies.factory import get_policy_class
print("POLICY_CLASS", get_policy_class("act").__name__)
"""

UNKNOWN_NAMES = """
import json, sys, tempfile
# As in a minimal install: no lookup here may need transformers, or peft, which imports it.
sys.modules["transformers"] = None
sys.modules["peft"] = None
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_policy_config

config_dir = tempfile.mkdtemp()
for name in ["pi_gemma", "nope", 5, None, ["act"]]:
    with open(f"{config_dir}/config.json", "w") as f:
        json.dump({"type": name}, f)
    from_config = (PreTrainedConfig.from_pretrained, config_dir)
    for lookup, arg in [(make_policy_config, name), (get_policy_class, name), from_config]:
        try:
            lookup(arg)
        except ValueError:
            print("VALUEERROR", lookup.__name__, name)
"""


def builtin_policies() -> dict[str, list[str]]:
    """The names each built-in policy package registers, read from its configuration module."""
    pattern = re.compile(r'PreTrainedConfig\.register_subclass\(\s*"([^"]+)"')
    root = Path(lerobot.policies.__path__[0])
    found = {
        path.parent.name: pattern.findall(path.read_text()) for path in root.glob("*/configuration_*.py")
    }
    return {package: names for package, names in found.items() if names}


def run_python(code: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, timeout=300)


def run_train_parse(*args: str) -> subprocess.CompletedProcess:
    return run_python(PARSE, *args)


@pytest.mark.parametrize("policy_type", [["--policy.type=act"], ["--policy.type", "act"]])
def test_policy_type_loads_only_that_policy(policy_type):
    result = run_train_parse(*policy_type, "--policy.device=cpu", "--dataset.repo_id=u/d")
    assert result.returncode == 0, result.stderr
    assert "POLICY ACTConfig cpu" in result.stdout
    assert "REGISTERED act\n" in result.stdout


def test_policy_loaded_from_a_directory_loads_only_that_policy(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"type": "act", "push_to_hub": False}))
    code = """
import sys
from lerobot.configs.policies import PreTrainedConfig
config = PreTrainedConfig.from_pretrained(sys.argv[1])
print("POLICY", type(config).__name__)
print("REGISTERED", ",".join(sorted(PreTrainedConfig.get_known_choices())))
"""
    result = run_python(code, str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "POLICY ACTConfig" in result.stdout
    assert "REGISTERED act\n" in result.stdout


def test_accelerator_dtype_fields_decode_without_a_policy_config():
    # No policy config class is defined here, so the codecs must come from the accelerator module itself.
    code = """
import draccus
from lerobot.configs.accelerator import MixedPrecisionPolicyConfig
config = draccus.decode(MixedPrecisionPolicyConfig, {"param_dtype": "bfloat16"})
print("DTYPE", config.param_dtype, draccus.encode(config)["param_dtype"])
"""
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert "DTYPE torch.bfloat16 bfloat16" in result.stdout


def test_each_builtin_policy_registers_the_name_of_its_package():
    # The policy named on the command line is found by importing the package of that name.
    policies = builtin_policies()
    assert "act" in policies
    assert {package: [package] for package in policies} == policies


def test_override_with_type_from_a_config_directory(tmp_path):
    (tmp_path / "train_config.json").write_text(
        json.dumps({"dataset": {"repo_id": "u/d"}, "policy": {"type": "act", "push_to_hub": False}})
    )
    result = run_train_parse(f"--config_path={tmp_path}", "--policy.device=cpu")
    assert result.returncode == 0, result.stderr
    assert "POLICY ACTConfig cpu" in result.stdout


def test_saved_train_config_with_a_policy_override(tmp_path):
    (tmp_path / "train_config.json").write_text(
        json.dumps({"dataset": {"repo_id": "u/d"}, "policy": {"type": "act", "push_to_hub": False}})
    )
    code = """
import sys
from lerobot.configs.train import TrainPipelineConfig
config = TrainPipelineConfig.from_pretrained(sys.argv[1], cli_args=["--policy.chunk_size=200"])
print("POLICY", type(config.policy).__name__, config.policy.chunk_size)
"""
    result = run_python(code, str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "POLICY ACTConfig 200" in result.stdout


def test_saved_train_config_reads_the_command_line_for_none(tmp_path):
    (tmp_path / "train_config.json").write_text(
        json.dumps({"dataset": {"repo_id": "u/d"}, "policy": {"type": "act", "push_to_hub": False}})
    )
    code = """
import sys
from lerobot.configs.train import TrainPipelineConfig
config_dir = sys.argv[1]
sys.argv = ["prog", "--policy.chunk_size=300"]
config = TrainPipelineConfig.from_pretrained(config_dir, cli_args=None)
print("POLICY", type(config.policy).__name__, config.policy.chunk_size)
"""
    result = run_python(code, str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "POLICY ACTConfig 300" in result.stdout


def test_policy_class_as_the_first_lookup():
    code = """
from lerobot.policies.factory import get_policy_class
print("POLICY_CLASS", get_policy_class("act").__name__)
"""
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert "POLICY_CLASS ACTPolicy" in result.stdout


@pytest.mark.parametrize("name", ["nope", "foo.bar"])
def test_unknown_policy_type_lists_every_policy(name):
    result = run_train_parse(f"--policy.type={name}", "--dataset.repo_id=u/d")
    error = result.stderr.strip().splitlines()[-1]
    assert result.returncode == 2
    assert "invalid choice" in error
    # Python versions differ on whether argparse quotes the choices.
    assert set(re.findall(r"\w+", error.split("choose from", 1)[1])) == set(builtin_policies())


def test_policy_class_of_a_non_string_name_lists_every_policy():
    code = """
from lerobot.policies.factory import get_policy_class

try:
    get_policy_class(5)
except ValueError as e:
    print(e)
"""
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert [name for name in builtin_policies() if f"'{name}'" not in result.stdout] == []


def test_policy_step_lookup_imports_only_the_module_that_registers_it():
    # These two policy packages do not import their processor module, so importing the package is not enough.
    steps = {
        "longwam_prompt": "lerobot.policies.longwam.processor_longwam",
        "fineart_vla_text_tokenizer": "lerobot.policies.fineart_vla.processor_fineart_vla",
    }
    lookup = """
import json, sys
from lerobot.processor.pipeline import ProcessorStepRegistry

modules = [ProcessorStepRegistry.get(name).__module__ for name in sys.argv[1:]]
print(json.dumps([modules, sorted(m for m in sys.modules if m.startswith("lerobot.policies."))]))
"""
    direct = """
import importlib, json, sys
import lerobot.processor.pipeline

for module in sys.argv[1:]:
    importlib.import_module(module)
print(json.dumps(sorted(m for m in sys.modules if m.startswith("lerobot.policies."))))
"""
    looked_up, imported = run_python(lookup, *steps), run_python(direct, *steps.values())
    assert looked_up.returncode == 0, looked_up.stderr
    assert imported.returncode == 0, imported.stderr
    modules, loaded = json.loads(looked_up.stdout.splitlines()[-1])
    assert modules == list(steps.values())
    assert loaded == json.loads(imported.stdout.splitlines()[-1])


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


def test_optimizer_and_scheduler_registered_by_another_policy():
    result = run_train_parse(
        "--policy.type=act",
        "--policy.device=cpu",
        "--dataset.repo_id=u/d",
        "--use_policy_training_preset=false",
        "--optimizer.type=molmoact2_adamw",
        "--scheduler.type=molmoact2_cosine_with_warmup",
        "--scheduler.num_warmup_steps=1",
        "--scheduler.num_decay_steps=10",
        "--scheduler.peak_lr=1e-4",
        "--scheduler.decay_lr=1e-6",
    )
    assert result.returncode == 0, result.stderr
    assert "OPTIMIZER MolmoAct2AdamWConfig" in result.stdout
    assert "SCHEDULER MolmoAct2CosineWithWarmupSchedulerConfig" in result.stdout
    assert "REGISTERED act,molmoact2\n" in result.stdout


@pytest.mark.parametrize(
    ("flag", "listed"),
    [
        ("--scheduler.type=molmoact2_adamw", "lawam_cosine_with_min_lr"),
        ("--optimizer.type=lawam_cosine_with_min_lr", "molmoact2_adamw"),
        ("--optimizer.type=nope", "molmoact2_adamw"),
    ],
)
def test_unknown_optimizer_or_scheduler_name_lists_every_choice(flag, listed):
    result = run_train_parse(
        "--policy.type=act",
        "--policy.device=cpu",
        "--dataset.repo_id=u/d",
        "--use_policy_training_preset=false",
        flag,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr
    assert listed in result.stderr


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_lists_every_policy(flag):
    result = run_train_parse(flag)
    assert result.returncode == 0, result.stderr
    choices = re.search(r"--policy\.type \{([^}]*)\}", result.stdout)
    assert choices, result.stdout
    assert {name.strip() for name in choices.group(1).split(",")} == set(builtin_policies())


def test_nested_policy_type_written_with_a_space():
    result = run_python(
        NESTED_PARSE,
        "--algorithm.type=sac",
        "--algorithm.policy_config.type",
        "act",
        "--policy.type=gaussian_actor",
        "--policy.device=cpu",
    )
    assert result.returncode == 0, result.stderr
    assert "NESTED ACTConfig" in result.stdout


def test_override_of_a_nested_policy_with_type_from_a_file(tmp_path):
    config_file = tmp_path / "train_config.json"
    config_file.write_text(
        json.dumps({"algorithm": {"type": "sac", "policy_config": {"type": "act", "push_to_hub": False}}})
    )
    result = run_python(
        NESTED_PARSE,
        f"--config_path={config_file}",
        "--policy.type=gaussian_actor",
        "--policy.device=cpu",
        "--algorithm.policy_config.n_action_steps=50",
    )
    assert result.returncode == 0, result.stderr
    assert "NESTED ACTConfig 50" in result.stdout


def test_policy_inside_a_registered_class_of_another_choice():
    code = """
import sys
from dataclasses import dataclass, field
import draccus
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig

@dataclass
class Algo(draccus.ChoiceRegistry):
    pass

@Algo.register_subclass("sub")
@dataclass
class SubAlgo(Algo):
    policy: PreTrainedConfig | None = None

@dataclass
class Root:
    algo: Algo = field(default_factory=SubAlgo)

@parser.wrap()
def main(cfg: Root):
    print("PARSED", type(cfg.algo).__name__, type(cfg.algo.policy).__name__)

sys.argv = ["prog", *sys.argv[1:]]
main()
"""
    result = run_python(code, "--algo.type=sub", "--algo.policy.type=act")
    assert result.returncode == 0, result.stderr
    assert "PARSED SubAlgo ACTConfig" in result.stdout


def test_policy_inside_a_plain_nested_config():
    result = run_python(PLAIN_NESTED, "--worker.policy.type=act")
    assert result.returncode == 0, result.stderr
    assert "NESTED ACTConfig" in result.stdout


def test_override_inside_a_plain_nested_config_with_type_from_a_file(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"worker": {"policy": {"type": "act", "push_to_hub": False}}}))
    result = run_python(PLAIN_NESTED, f"--config_path={config_file}", "--worker.policy.chunk_size=200")
    assert result.returncode == 0, result.stderr
    assert "NESTED ACTConfig 200" in result.stdout


def test_policy_owned_optimizer_inside_a_plugin_policy():
    code = """
import sys
from dataclasses import dataclass
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import OptimizerConfig
from lerobot.policies.act.configuration_act import ACTConfig

@PreTrainedConfig.register_subclass("plugin_with_optimizer")
@dataclass
class PluginWithOptimizerConfig(ACTConfig):
    optimizer: OptimizerConfig | None = None

@dataclass
class Root:
    policy: PreTrainedConfig | None = None

@parser.wrap()
def main(cfg: Root):
    print("PARSED", type(cfg.policy).__name__, type(cfg.policy.optimizer).__name__)

sys.argv = ["prog", *sys.argv[1:]]
main()
"""
    result = run_python(
        code, "--policy.type=plugin_with_optimizer", "--policy.optimizer.type=molmoact2_adamw"
    )
    assert result.returncode == 0, result.stderr
    assert "PARSED PluginWithOptimizerConfig MolmoAct2AdamWConfig" in result.stdout


def test_plugin_cannot_take_an_optimizer_name_a_policy_owns():
    code = """
from dataclasses import dataclass
from lerobot.optim.optimizers import OptimizerConfig

try:
    @OptimizerConfig.register_subclass("molmoact2_adamw")
    @dataclass
    class PluginOptimizer(OptimizerConfig):
        def build(self, params):
            return None
except ValueError:
    print("REJECTED")
print("OWNER", OptimizerConfig.get_choice_class("molmoact2_adamw").__module__)
"""
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert "REJECTED" in result.stdout
    assert "OWNER lerobot.policies.molmoact2." in result.stdout


def test_plugin_cannot_take_a_policy_processor_step_name():
    code = """
from lerobot.processor.pipeline import ProcessorStepRegistry

try:
    @ProcessorStepRegistry.register("pi0_new_line_processor")
    class PluginStep:
        pass
except ValueError:
    print("REJECTED")
from lerobot.configs.policies import PreTrainedConfig
print("PI0", PreTrainedConfig.get_choice_class("pi0").__name__)
print("STEP", ProcessorStepRegistry.get("pi0_new_line_processor").__module__)
"""
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert "REJECTED" in result.stdout
    assert "PI0 PI0Config" in result.stdout
    assert "STEP lerobot.policies.pi0.processor_pi0" in result.stdout


def test_plugin_can_replace_a_policy_processor_step_after_unregistering_it():
    code = """
from lerobot.processor.pipeline import ProcessorStepRegistry

ProcessorStepRegistry.unregister("pi0_new_line_processor")

@ProcessorStepRegistry.register("pi0_new_line_processor")
class PluginStep:
    pass

from lerobot.configs.policies import PreTrainedConfig
print("PI0", PreTrainedConfig.get_choice_class("pi0").__name__)
print("STEP", ProcessorStepRegistry.get("pi0_new_line_processor").__module__)
"""
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert "PI0 PI0Config" in result.stdout
    assert "STEP __main__" in result.stdout


def test_plugin_registering_new_names_loads_no_policy():
    code = """
import json, sys
from dataclasses import dataclass
from lerobot.optim.optimizers import OptimizerConfig
from lerobot.processor.pipeline import ProcessorStepRegistry

@ProcessorStepRegistry.register("my_plugin_step")
class PluginStep:
    pass

@OptimizerConfig.register_subclass("my_plugin_optimizer")
@dataclass
class PluginOptimizer(OptimizerConfig):
    def build(self, params):
        return None

print(json.dumps(sorted({m.split(".")[2] for m in sys.modules if m.startswith("lerobot.policies.")})))
"""
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    loaded = set(json.loads(result.stdout.splitlines()[-1]))
    assert not loaded & set(builtin_policies()), loaded


def test_reserved_names_match_what_policies_register():
    code = """
import importlib, json, pkgutil
import lerobot.policies
from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import ChoicesFromPolicies, OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.processor.pipeline import BUILTIN_STEP_MODULES, ProcessorStepRegistry

PreTrainedConfig.load_all_choices()
for module in pkgutil.walk_packages(lerobot.policies.__path__, "lerobot.policies."):
    if module.name.rsplit(".", 1)[-1].startswith("processor_"):
        importlib.import_module(module.name)
steps = {n: c.__module__ for n, c in ProcessorStepRegistry._registry.items() if c.__module__.startswith("lerobot.policies.")}
choices = {
    n: c.__module__.split(".")[2]
    for registry in (OptimizerConfig, LRSchedulerConfig)
    for n, c in registry.get_known_choices().items()
    if c.__module__.startswith("lerobot.policies.")
}
print(json.dumps([steps, {n: "lerobot." + m for n, m in BUILTIN_STEP_MODULES.items()}, choices, ChoicesFromPolicies.POLICY_CHOICES]))
"""
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    steps, step_table, choices, choice_table = json.loads(result.stdout.splitlines()[-1])
    assert steps == step_table
    assert choices == choice_table


def test_plugin_cannot_take_a_builtin_policy_name():
    result = run_python(PLUGIN)
    assert result.returncode == 0, result.stderr
    assert "REJECTED" in result.stdout
    assert "ACT ACTConfig" in result.stdout
    assert "POLICY_CLASS ACTPolicy" in result.stdout


def test_unknown_or_non_string_policy_type_raises_value_error():
    result = run_python(UNKNOWN_NAMES)
    assert result.returncode == 0, result.stderr
    # Five names, each through three lookups.
    assert len(result.stdout.split("VALUEERROR")) - 1 == 15, result.stdout
