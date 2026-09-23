"""Policy registration on demand, checked in fresh interpreters.

Test modules register policies as they are collected, so these run each case in a new Python process, the way a
command runs.
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
    loaded = sorted({m.split(".")[2] for m in sys.modules if m.startswith("lerobot.policies.") and m.count(".") >= 2})
    print("POLICY", type(cfg.policy).__name__, cfg.policy.device)
    print("OPTIMIZER", type(cfg.optimizer).__name__)
    print("LOADED", ",".join(loaded))

main()
"""


def run_train_parse(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", PARSE, *args], capture_output=True, text=True, timeout=300)


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
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stderr
    assert "STEPS ['Pi0NewLineProcessor']" in result.stdout
