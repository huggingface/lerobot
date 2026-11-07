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

# Only the tests that run lerobot-train need the dataset extra, so the others also run in the base tier,
# without it.
NEEDS_DATASETS = pytest.mark.skipif(not _datasets_available, reason="lerobot-train needs the dataset extra")

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

PLUGIN_WITH_OPTIMIZER = """
from dataclasses import dataclass
from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import OptimizerConfig
from lerobot.policies.act.configuration_act import ACTConfig

@PreTrainedConfig.register_subclass("plugin_with_optimizer")
@dataclass
class PluginWithOptimizerConfig(ACTConfig):
    optimizer: OptimizerConfig | None = None
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


def builtin_policies() -> set[str]:
    """The names the built-in policy packages register, read from their configuration modules."""
    pattern = re.compile(r'PreTrainedConfig\.register_subclass\(\s*"([^"]+)"')
    root = Path(lerobot.policies.__path__[0])
    return {
        name
        for path in root.glob("*/configuration_*.py")
        for name in pattern.findall(path.read_text(encoding="utf-8"))
    }


def run_python(code: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, timeout=300)


def run_train_parse(*args: str) -> subprocess.CompletedProcess:
    return run_python(PARSE, *args)


@pytest.mark.dataset
@NEEDS_DATASETS
def test_policy_type_loads_only_that_policy():
    result = run_train_parse("--policy.type", "act", "--policy.device=cpu", "--dataset.repo_id=u/d")
    assert result.returncode == 0, result.stderr
    assert "POLICY ACTConfig cpu" in result.stdout
    assert "REGISTERED act\n" in result.stdout


def test_policy_path_loads_only_that_policy(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"type": "act", "push_to_hub": False}))
    code = """
import sys
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.configs.policies import PreTrainedConfig

sys.argv = ["lerobot-eval", *sys.argv[1:]]

@parser.wrap()
def main(cfg: EvalPipelineConfig):
    print("POLICY", type(cfg.policy).__name__, cfg.policy.device)
    print("REGISTERED", ",".join(sorted(PreTrainedConfig.get_known_choices())))

main()
"""
    result = run_python(code, f"--policy.path={tmp_path}", "--policy.device=cpu", "--env.type=pusht")
    assert result.returncode == 0, result.stderr
    assert "POLICY ACTConfig cpu" in result.stdout
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


def test_policy_config_dtype_round_trips_without_the_accelerator_config(tmp_path):
    # The codecs must come from defining the policy config, since the accelerator module registers them too.
    code = """
import sys
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig

ACTConfig(device="cpu", dtype=torch.bfloat16, push_to_hub=False).save_pretrained(sys.argv[1])
config = PreTrainedConfig.from_pretrained(sys.argv[1])
print("DTYPE", config.dtype, "lerobot.configs.accelerator" in sys.modules)
"""
    result = run_python(code, str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "DTYPE torch.bfloat16 False" in result.stdout


def test_dtype_field_of_a_plugin_config_parses(tmp_path):
    # No policy or accelerator config is defined here, and torch loads only when the type lookup imports the
    # plugin, so the parser must register the codecs after its lookups.
    (tmp_path / "engines").mkdir()
    (tmp_path / "engines" / "__init__.py").write_text("")
    (tmp_path / "engines" / "fast.py").write_text(
        "from dataclasses import dataclass\n"
        "import torch\n"
        "from __main__ import Engine\n"
        "@Engine.register_subclass('fast')\n"
        "@dataclass\n"
        "class FastEngine(Engine):\n"
        "    dtype: torch.dtype | None = None\n"
    )
    code = """
import sys
from dataclasses import dataclass
import draccus
from lerobot.configs import parser

sys.path.insert(0, sys.argv[1])

@dataclass
class Engine(draccus.PluginRegistry, discover_packages_path="engines"):
    pass

@dataclass
class PluginConfig:
    engine: Engine

@parser.wrap()
def main(cfg: PluginConfig):
    print("DTYPE", cfg.engine.dtype)

sys.argv = ["prog", "--engine.type=fast", "--engine.dtype=bfloat16"]
main()
"""
    result = run_python(code, str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "DTYPE torch.bfloat16" in result.stdout


def test_saved_train_config_with_a_policy_override(tmp_path):
    (tmp_path / "train_config.json").write_text(
        json.dumps({"dataset": {"repo_id": "u/d"}, "policy": {"type": "act", "push_to_hub": False}})
    )
    code = """
import sys
from lerobot.configs.train import TrainPipelineConfig
config = TrainPipelineConfig.from_pretrained(sys.argv[1], cli_args=iter(["--policy.chunk_size=200"]))
print("POLICY", type(config.policy).__name__, config.policy.chunk_size)
"""
    result = run_python(code, str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "POLICY ACTConfig 200" in result.stdout


def test_saved_train_config_with_a_policy_flag_reports_a_failed_policy_once(tmp_path):
    # The flag has no type, so every policy loads, and a package that fails to import is skipped with a warning.
    (tmp_path / "train_config.json").write_text(
        json.dumps({"dataset": {"repo_id": "u/d"}, "policy": {"type": "act", "push_to_hub": False}})
    )
    code = """
import sys
sys.modules["lerobot.policies.vqbet.modeling_vqbet"] = None
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig

@parser.wrap()
def main(cfg: TrainPipelineConfig):
    print("POLICY", type(cfg.policy).__name__, cfg.policy.chunk_size)

sys.argv = ["lerobot-train", f"--config_path={sys.argv[1]}", "--policy.chunk_size=200"]
main()
"""
    result = run_python(code, str(tmp_path / "train_config.json"))
    assert result.returncode == 0, result.stderr
    assert "POLICY ACTConfig 200" in result.stdout
    assert result.stderr.count("Skipping lerobot.policies.vqbet") == 1, result.stderr


def test_saved_configs_read_the_command_line_for_none(tmp_path):
    (tmp_path / "train_config.json").write_text(
        json.dumps({"dataset": {"repo_id": "u/d"}, "policy": {"type": "act", "push_to_hub": False}})
    )
    (tmp_path / "config.json").write_text(json.dumps({"type": "act", "push_to_hub": False}))
    code = """
import sys
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
config_dir = sys.argv[1]
sys.argv = ["prog", "--policy.chunk_size=300"]
config = TrainPipelineConfig.from_pretrained(config_dir, cli_args=None)
print("POLICY", type(config.policy).__name__, config.policy.chunk_size)
sys.argv = ["prog", "--chunk_size=200"]
policy = PreTrainedConfig.from_pretrained(config_dir, cli_overrides=None)
print("SAVED_POLICY", type(policy).__name__, policy.chunk_size)
"""
    result = run_python(code, str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "POLICY ACTConfig 300\nSAVED_POLICY ACTConfig 200\n" in result.stdout


def test_path_fields_from_a_config_file_apply_only_to_their_own_parse(tmp_path):
    # Later parses in the same process, with or without a policy path of their own, must not see them.
    (tmp_path / "act").mkdir()
    (tmp_path / "act" / "config.json").write_text(json.dumps({"type": "act", "push_to_hub": False}))
    eval_config = {"env": {"type": "pusht"}, "policy": {"path": str(tmp_path / "act"), "n_action_steps": 7}}
    (tmp_path / "eval.json").write_text(json.dumps(eval_config))
    (tmp_path / "train_config.json").write_text(
        json.dumps({"dataset": {"repo_id": "u/d"}, "policy": {"type": "diffusion", "push_to_hub": False}})
    )
    code = """
import sys
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.configs.train import TrainPipelineConfig

@parser.wrap()
def evaluate(cfg: EvalPipelineConfig):
    print("EVAL", type(cfg.policy).__name__, cfg.policy.n_action_steps)

folder = sys.argv[1]
for args in [[f"--config_path={folder}/eval.json"], ["--env.type=pusht", f"--policy.path={folder}/act"]]:
    sys.argv = ["lerobot-eval", *args]
    evaluate()
config = TrainPipelineConfig.from_pretrained(folder, cli_args=["--policy.horizon=128"])
print("TRAIN", type(config.policy).__name__, config.policy.horizon)
"""
    result = run_python(code, str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "EVAL ACTConfig 7\nEVAL ACTConfig 100\nTRAIN DiffusionConfig 128\n" in result.stdout


def test_policy_path_given_with_a_type_is_rejected_before_the_type_is_imported():
    code = """
import importlib.abc, sys
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig

class BreakACT(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "lerobot.policies.act":
            raise ImportError("act cannot be imported")

sys.meta_path.insert(0, BreakACT())
sys.argv = ["lerobot-eval", "--policy.path=/nonexistent", "--policy.type=act"]

@parser.wrap()
def main(cfg: EvalPipelineConfig):
    pass

try:
    main()
except Exception as e:
    print(type(e).__name__, e)
"""
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert "ArgumentError Cannot specify both --policy.path and --policy.type" in result.stdout


def test_first_lookups_of_different_policies_from_several_threads():
    # Each first lookup imports a policy, and policies share dependencies such as torchvision.
    code = """
import threading
from lerobot.configs.policies import PreTrainedConfig

names = ["act", "diffusion", "groot", "lawam", "pi0", "smolvla", "tdmpc", "vqbet"]
barrier = threading.Barrier(len(names))
errors = []

def look_up(name):
    barrier.wait()
    try:
        PreTrainedConfig.get_choice_class(name)
    except Exception as e:
        errors.append(f"{name}: {e!r}")

threads = [threading.Thread(target=look_up, args=(name,)) for name in names]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
print("ERRORS", errors)
"""
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert "ERRORS []" in result.stdout


def test_lookup_of_an_imported_policy_does_not_wait_for_another_threads_import():
    # Another thread can hold the import lock for the seconds a large policy takes to import.
    code = """
import threading
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils import import_utils

PreTrainedConfig.get_choice_class("act")
found = []
thread = threading.Thread(target=lambda: found.append(PreTrainedConfig.get_choice_class("act").__name__))
with import_utils._on_demand_import_lock:
    thread.start()
    thread.join(60)
    print("WAITED" if thread.is_alive() else f"FOUND {found}")
"""
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert "FOUND ['ACTConfig']" in result.stdout


def test_names_registered_by_a_package_that_then_failed_to_import_raise_on_every_lookup():
    # The package registers its config and its scheduler, then fails to import its model. A later lookup must
    # not hand out the classes of a package that never loaded.
    code = """
import sys
sys.modules["lerobot.policies.lawam.modeling_lawam"] = None
from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.schedulers import LRSchedulerConfig

for registry, name in [(PreTrainedConfig, "lawam"), (LRSchedulerConfig, "lawam_cosine_with_min_lr")]:
    for _ in range(2):
        try:
            registry.get_choice_class(name)
            print("RETURNED", name)
        except ImportError:
            print("RAISED", name)
"""
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    expected = (
        "RAISED lawam\nRAISED lawam\nRAISED lawam_cosine_with_min_lr\nRAISED lawam_cosine_with_min_lr\n"
    )
    assert expected in result.stdout


def test_registries_can_be_read_while_another_thread_registers_into_them():
    # draccus walks a registry to find the name of a config, as cfg.type does, and callers walk
    # get_known_choices(). A first lookup in another thread adds to the registry the way these plugins do.
    code = """
import sys, threading
from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.schedulers import LRSchedulerConfig

sys.setswitchinterval(1e-6)
done, errors = threading.Event(), []

def read(registry):
    while not done.is_set():
        try:
            for _ in registry.get_known_choices():
                pass
            # The base class has no name, so this walks the whole registry.
            registry.get_choice_name(registry)
        except ValueError:
            pass
        except RuntimeError as e:
            errors.append(repr(e))
            return

readers = [threading.Thread(target=read, args=(r,)) for r in (PreTrainedConfig, LRSchedulerConfig)]
for reader in readers:
    reader.start()
for i in range(200):
    for registry in (PreTrainedConfig, LRSchedulerConfig):
        registry.register_subclass(f"plugin_{i}", type(f"Plugin{i}", (registry,), {}))
done.set()
for reader in readers:
    reader.join()
print("ERRORS", errors)
"""
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert "ERRORS []" in result.stdout


def test_policy_name_given_as_a_str_enum():
    code = """
from enum import Enum
from lerobot.policies.factory import get_policy_class

class PolicyName(str, Enum):
    ACT = "act"

print("POLICY_CLASS", get_policy_class(PolicyName.ACT).__name__)
"""
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert "POLICY_CLASS ACTPolicy" in result.stdout


@pytest.mark.dataset
@NEEDS_DATASETS
def test_unknown_policy_type_lists_every_policy():
    result = run_train_parse("--policy.type=foo.bar", "--dataset.repo_id=u/d")
    error = result.stderr.strip().splitlines()[-1]
    assert result.returncode == 2
    assert "invalid choice" in error
    # Python versions differ on whether argparse quotes the choices.
    assert set(re.findall(r"\w+", error.split("choose from", 1)[1])) == builtin_policies()


def test_config_with_a_non_string_type_lists_every_policy(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"type": 5}))
    code = """
import sys
from lerobot.configs.policies import PreTrainedConfig

try:
    PreTrainedConfig.from_pretrained(sys.argv[1])
except ValueError as e:
    print(e)
"""
    result = run_python(code, str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert [name for name in builtin_policies() if f"'{name}'" not in result.stdout] == []


def test_unknown_policy_type_when_another_policy_fails_to_import(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"type": "nope"}))
    # The package fails with a KeyError, as a lookup in its code could raise, which must not read as an unknown
    # name.
    code = """
import importlib.abc, importlib.util, sys
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class

class BreakVQBeT(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name == "lerobot.policies.vqbet.modeling_vqbet":
            return importlib.util.spec_from_loader(name, self)

    def exec_module(self, module):
        raise KeyError("vqbet_setting")

sys.meta_path.insert(0, BreakVQBeT())
lookups = [
    (PreTrainedConfig.get_choice_class, "nope"),
    (PreTrainedConfig.from_pretrained, sys.argv[1]),
    (get_policy_class, "nope"),
    (PreTrainedConfig.get_choice_class, "vqbet"),
]
for lookup, arg in lookups:
    try:
        lookup(arg)
    except (KeyError, ValueError, ImportError) as e:
        print("RAISED", type(e).__name__)
print("ACT", PreTrainedConfig.get_choice_class("act").__name__)
"""
    result = run_python(code, str(tmp_path))
    assert result.returncode == 0, result.stderr
    expected = "RAISED KeyError\nRAISED ValueError\nRAISED ValueError\nRAISED ImportError\nACT ACTConfig\n"
    assert expected in result.stdout
    # One warning per failed lookup of an unknown name: each imports every policy once.
    assert result.stderr.count("Skipping lerobot.policies.vqbet") == 3


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


def test_list_and_unknown_step_error_include_every_builtin_step_without_importing_it():
    # Both should show the built-in names, not only the imported ones, as for a mistyped saved step name.
    code = """
import sys
from lerobot.processor.pipeline import BUILTIN_STEP_MODULES, ProcessorStepRegistry

listed = ProcessorStepRegistry.list()
try:
    ProcessorStepRegistry.get("pi0_new_line_processr")
except KeyError as e:
    print("UNLISTED", [n for n in BUILTIN_STEP_MODULES if n not in listed or repr(n) not in e.args[0]])
print("IMPORTED", [m for m in BUILTIN_STEP_MODULES.values() if "lerobot." + m in sys.modules])
"""
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert "UNLISTED []" in result.stdout
    assert "IMPORTED []" in result.stdout


def test_step_lookup_waits_for_a_step_module_another_thread_is_importing(tmp_path):
    # The step registers before its module finishes running, so a lookup has to wait for the whole module.
    (tmp_path / "slow_step_module.py").write_text(
        "import __main__\n"
        "from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry\n"
        "@ProcessorStepRegistry.register('slow_step')\n"
        "class SlowStep(ProcessorStep):\n"
        "    pass\n"
        "__main__.registered.set()\n"
        "__main__.release.wait(60)\n"
        "LOADED = True\n"
    )
    code = """
import sys, threading
import lerobot.policies
from lerobot.processor.pipeline import BUILTIN_STEP_MODULES, ProcessorStepRegistry

lerobot.policies.__path__.append(sys.argv[1])
BUILTIN_STEP_MODULES["slow_step"] = "policies.slow_step_module"
registered, release, started, seen = threading.Event(), threading.Event(), threading.Event(), {}
importer = threading.Thread(target=__import__, args=("lerobot.policies.slow_step_module",))
importer.start()
assert registered.wait(60)

def look_up():
    started.set()
    ProcessorStepRegistry.get("slow_step")
    seen["loaded"] = getattr(sys.modules["lerobot.policies.slow_step_module"], "LOADED", False)

looker = threading.Thread(target=look_up)
looker.start()
assert started.wait(60)
looker.join(1)
seen["waiting"] = looker.is_alive()
release.set()
importer.join(60)
looker.join(60)
print("WAITING", seen["waiting"], "LOADED", seen["loaded"])
"""
    result = run_python(code, str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "WAITING True LOADED True" in result.stdout


def test_step_lookup_raises_the_error_of_a_step_module_that_failed_after_registering(tmp_path):
    # The failed import leaves the step registered, and the second lookup runs the module again.
    (tmp_path / "failing_step_module.py").write_text(
        "from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry\n"
        "@ProcessorStepRegistry.register('failing_step')\n"
        "class FailingStep(ProcessorStep):\n"
        "    pass\n"
        "raise RuntimeError('broken step module')\n"
    )
    code = """
import sys
import lerobot.policies
from lerobot.processor.pipeline import BUILTIN_STEP_MODULES, ProcessorStepRegistry

lerobot.policies.__path__.append(sys.argv[1])
BUILTIN_STEP_MODULES["failing_step"] = "policies.failing_step_module"
for _ in range(2):
    try:
        ProcessorStepRegistry.get("failing_step")
    except ImportError as e:
        print("RAISED", e)
"""
    result = run_python(code, str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("RAISED") == result.stdout.count("broken step module") == 2, result.stdout


@pytest.mark.dataset
@NEEDS_DATASETS
def test_repeated_policy_type_uses_the_last_one():
    result = run_train_parse(
        "--policy.type=act", "--policy.device=cpu", "--dataset.repo_id=u/d", "--policy.type=diffusion"
    )
    assert result.returncode == 0, result.stderr
    assert "POLICY DiffusionConfig cpu" in result.stdout


@pytest.mark.dataset
@NEEDS_DATASETS
def test_optimizer_flags_without_a_type_load_only_the_policies_that_add_optimizers(tmp_path):
    config = {
        "dataset": {"repo_id": "u/d"},
        "policy": {"type": "act", "device": "cpu", "push_to_hub": False},
        "use_policy_training_preset": False,
        "optimizer": {"type": "adamw"},
    }
    (tmp_path / "train_config.json").write_text(json.dumps(config))
    result = run_train_parse(f"--config_path={tmp_path / 'train_config.json'}", "--optimizer.lr=0.5")
    assert result.returncode == 0, result.stderr
    assert "OPTIMIZER AdamWConfig(lr=0.5," in result.stdout
    assert "REGISTERED act,molmoact2\n" in result.stdout


@pytest.mark.dataset
@NEEDS_DATASETS
def test_unknown_scheduler_name_lists_every_choice():
    result = run_train_parse(
        "--policy.type=act",
        "--policy.device=cpu",
        "--dataset.repo_id=u/d",
        "--use_policy_training_preset=false",
        "--scheduler.type=molmoact2_adamw",
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr
    assert "lawam_cosine_with_min_lr" in result.stderr


@pytest.mark.dataset
@NEEDS_DATASETS
def test_help_lists_every_policy():
    # argparse reads -help as -h followed by more short options, so it asks for help too.
    result = run_train_parse("-help")
    assert result.returncode == 0, result.stderr
    choices = re.search(r"--policy\.type \{([^}]*)\}", result.stdout)
    assert choices, result.stdout
    assert {name.strip() for name in choices.group(1).split(",")} == builtin_policies()
    from lerobot.optim.optimizers import OptimizerConfig
    from lerobot.optim.schedulers import LRSchedulerConfig

    for flag, registry in [("optimizer", OptimizerConfig), ("scheduler", LRSchedulerConfig)]:
        listed = {
            name.strip()
            for choices in re.findall(rf"--{flag}\.type \{{([^}}]*)\}}", result.stdout)
            for name in choices.split(",")
        }
        assert set(registry.POLICY_CHOICES) <= listed, flag


def test_help_skips_a_registered_choice_whose_annotations_cannot_be_resolved():
    # As in a plugin that imports a field type only for type checking, because its extra is not installed.
    code = """
import sys
from dataclasses import dataclass, field
import draccus
from lerobot.configs import parser

@dataclass
class CameraConfig(draccus.ChoiceRegistry):
    pass

@CameraConfig.register_subclass("plugin")
@dataclass
class PluginCameraConfig(CameraConfig):
    setting: "MissingExtraSetting | None" = None

@dataclass
class RobotConfig(draccus.ChoiceRegistry):
    pass

@RobotConfig.register_subclass("arm")
@dataclass
class ArmConfig(RobotConfig):
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

@dataclass
class Config:
    robot: RobotConfig

@parser.wrap()
def main(cfg: Config):
    pass

sys.argv = ["prog", "--help"]
main()
"""
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert "--robot.type" in result.stdout


def test_policy_inside_a_plain_nested_config():
    result = run_python(PLAIN_NESTED, "--worker.policy.type=act")
    assert result.returncode == 0, result.stderr
    assert "NESTED ACTConfig" in result.stdout


def test_policy_owned_optimizer_inside_a_policy_that_registers_on_demand(tmp_path):
    # The same policy as a package that registers only when --policy.type looks it up, as a built-in does.
    (tmp_path / "plugin_with_optimizer").mkdir()
    (tmp_path / "plugin_with_optimizer" / "__init__.py").write_text(PLUGIN_WITH_OPTIMIZER)
    code = """
import sys
from dataclasses import dataclass
import lerobot.policies
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig

lerobot.policies.__path__.append(sys.argv[1])

@dataclass
class Root:
    policy: PreTrainedConfig | None = None

@parser.wrap()
def main(cfg: Root):
    print("PARSED", type(cfg.policy).__name__, type(cfg.policy.optimizer).__name__)

sys.argv = ["prog", *sys.argv[2:]]
main()
"""
    result = run_python(
        code, str(tmp_path), "--policy.type=plugin_with_optimizer", "--policy.optimizer.type=molmoact2_adamw"
    )
    assert result.returncode == 0, result.stderr
    assert "PARSED PluginWithOptimizerConfig MolmoAct2AdamWConfig" in result.stdout


def test_policy_owned_optimizer_override_of_a_saved_plugin_policy(tmp_path):
    config = {"type": "plugin_with_optimizer", "push_to_hub": False, "optimizer": {"type": "molmoact2_adamw"}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    code = """
import sys

config = PreTrainedConfig.from_pretrained(sys.argv[1], cli_overrides=sys.argv[2:])
optimizer = config.optimizer
print("PARSED", type(config).__name__, type(optimizer).__name__, getattr(optimizer, "group_grad_clip_norm", None))
"""
    result = run_python(PLUGIN_WITH_OPTIMIZER + code, str(tmp_path), "--optimizer.group_grad_clip_norm=2.5")
    assert result.returncode == 0, result.stderr
    assert "PARSED PluginWithOptimizerConfig MolmoAct2AdamWConfig 2.5" in result.stdout


def test_policy_owned_optimizer_defined_in_its_policy_package_itself(tmp_path):
    # A stand-in for molmoact2 whose __init__.py defines its optimizer, as a policy's can define its config.
    (tmp_path / "molmoact2").mkdir()
    (tmp_path / "molmoact2" / "__init__.py").write_text(
        "from lerobot.optim.optimizers import OptimizerConfig\n"
        "@OptimizerConfig.register_subclass('molmoact2_adamw')\n"
        "class MolmoAct2AdamWConfig(OptimizerConfig):\n"
        "    pass\n"
    )
    code = """
import sys
import lerobot.policies
from lerobot.optim.optimizers import OptimizerConfig

lerobot.policies.__path__.insert(0, sys.argv[1])
print("OPTIMIZER", OptimizerConfig.get_choice_class("molmoact2_adamw").__module__)
"""
    result = run_python(code, str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "OPTIMIZER lerobot.policies.molmoact2\n" in result.stdout


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


def test_reserved_names_match_what_policies_register():
    code = """
import importlib, json, pkgutil
import lerobot.policies
from lerobot.configs.policies import BUILTIN_POLICY_PACKAGES, PreTrainedConfig
from lerobot.optim.optimizers import OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.processor.pipeline import BUILTIN_STEP_MODULES, ProcessorStepRegistry

PreTrainedConfig.load_all_choices()
packages = {n: c.__module__.split(".")[2] for n, c in PreTrainedConfig.get_known_choices().items()}
renamed = {n: package for n, package in packages.items() if n != package}
for module in pkgutil.walk_packages(lerobot.policies.__path__, "lerobot.policies."):
    if module.name.rsplit(".", 1)[-1].startswith("processor_"):
        importlib.import_module(module.name)
steps = {n: c.__module__ for n, c in ProcessorStepRegistry._registry.items() if c.__module__.startswith("lerobot.policies.")}
choices = {
    registry.__name__: {
        n: c.__module__.split(".")[2]
        for n, c in registry.get_known_choices().items()
        if c.__module__.startswith("lerobot.policies.")
    }
    for registry in (OptimizerConfig, LRSchedulerConfig)
}
tables = {registry.__name__: registry.POLICY_CHOICES for registry in (OptimizerConfig, LRSchedulerConfig)}
step_table = {n: "lerobot." + m for n, m in BUILTIN_STEP_MODULES.items()}
print(json.dumps([steps, step_table, choices, tables, renamed, BUILTIN_POLICY_PACKAGES]))
"""
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    steps, step_table, choices, choice_table, renamed, policy_table = json.loads(
        result.stdout.splitlines()[-1]
    )
    assert steps == step_table, "Keep BUILTIN_STEP_MODULES in src/lerobot/processor/pipeline.py in sync"
    assert choices == choice_table, "Keep the POLICY_CHOICES of OptimizerConfig and LRSchedulerConfig in sync"
    assert renamed == policy_table, "Keep BUILTIN_POLICY_PACKAGES in src/lerobot/configs/policies.py in sync"


def test_names_stay_reserved_when_their_owner_fails_to_import_until_unregistered():
    code = """
import importlib.abc, importlib.machinery, sys
from dataclasses import dataclass

class FailOnImport(importlib.abc.Loader):
    def exec_module(self, module):
        raise ImportError(f"{module.__name__} cannot be imported")

class BreakOwners(importlib.abc.MetaPathFinder):
    # Found, but failing when imported, as a package with a missing dependency does.
    def find_spec(self, name, path=None, target=None):
        if name in {
            "lerobot.policies.act",
            "lerobot.policies.cosmos3",
            "lerobot.policies.molmoact2",
            "lerobot.policies.pi0.processor_pi0",
        }:
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            spec.loader = FailOnImport()
            return spec

sys.meta_path.insert(0, BreakOwners())
from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import OptimizerConfig
from lerobot.processor.pipeline import ProcessorStepRegistry

def register_policy(name):
    @PreTrainedConfig.register_subclass(name)
    @dataclass
    class PluginConfig(PreTrainedConfig):
        pass

def register_optimizer(name):
    @OptimizerConfig.register_subclass(name)
    @dataclass
    class PluginOptimizer(OptimizerConfig):
        def build(self, params):
            return None

def register_step(name):
    @ProcessorStepRegistry.register(name)
    class PluginStep:
        pass

for register, name in [
    (register_policy, "act"),
    (register_policy, "cosmos3_nano"),
    (register_optimizer, "molmoact2_adamw"),
    (register_step, "pi0_new_line_processor"),
]:
    try:
        register(name)
    except ValueError as e:
        print("REJECTED", e)
ProcessorStepRegistry.unregister("pi0_new_line_processor")
print("LISTED", "pi0_new_line_processor" in ProcessorStepRegistry.list())
# An owner that imports fine: unregister imports it, so clear() alone must keep the name reserved.
ProcessorStepRegistry.unregister("longwam_prompt")

@ProcessorStepRegistry.register("pi0_new_line_processor")
class ReplacementStep:
    pass

print("REPLACED", ProcessorStepRegistry.get("pi0_new_line_processor").__name__)

# clear() forgets the unregisters, so both names are reserved again.
ProcessorStepRegistry.clear()
print("CLEARED")
for name in ["pi0_new_line_processor", "longwam_prompt"]:
    try:
        register_step(name)
    except ValueError as e:
        print("REJECTED", e)
"""
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    rejected = [line for line in result.stdout.splitlines() if line.startswith("REJECTED")]
    assert len(rejected) == 6, result.stdout
    assert all("is reserved for lerobot.policies." in line for line in rejected), rejected
    assert "LISTED False" in result.stdout
    assert "REPLACED ReplacementStep\nCLEARED\nREJECTED" in result.stdout


def test_a_directory_without_init_under_policies_does_not_reserve_its_name(tmp_path):
    # As git leaves a removed policy's __pycache__ behind. load_all_choices does not import such a directory.
    (tmp_path / "retired_policy" / "__pycache__").mkdir(parents=True)
    code = """
import sys
from dataclasses import dataclass
import lerobot.policies
from lerobot.configs.policies import PreTrainedConfig

lerobot.policies.__path__.append(sys.argv[1])

@PreTrainedConfig.register_subclass("retired_policy")
@dataclass
class PluginConfig(PreTrainedConfig):
    pass

print("POLICY", PreTrainedConfig.get_choice_class("retired_policy").__name__)
"""
    result = run_python(code, str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "POLICY PluginConfig" in result.stdout


def test_unknown_or_non_string_policy_type_raises_value_error():
    result = run_python(UNKNOWN_NAMES)
    assert result.returncode == 0, result.stderr
    # Five names, each through three lookups.
    assert len(result.stdout.split("VALUEERROR")) - 1 == 15, result.stdout
