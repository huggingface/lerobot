"""Robot-side processing without torch, checked in fresh interpreters.

Test modules import torch as they are collected, so each case runs in a new Python process.
"""

import subprocess
import sys

# The spec is still found, so code that only checks whether torch is installed keeps working. Each
# load attempt is recorded, so an import wrapped in try/except still fails the test.
RECORD_TORCH = """
import atexit, importlib.abc, importlib.machinery, os, sys, traceback
attempts = []
class RefuseLoader(importlib.abc.Loader):
    def exec_module(self, module):
        attempts.append("".join(traceback.format_stack()))
        raise RuntimeError("torch must not be imported here")
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != "torch":
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        spec.loader = RefuseLoader()
        return spec
sys.meta_path.insert(0, BlockTorch())
atexit.register(lambda: attempts and (print(attempts[0], file=sys.stderr), os._exit(3)))
"""


def run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)


def test_default_robot_processors_run_without_torch(tmp_path):
    result = run(
        RECORD_TORCH
        + f"""
import numpy as np
from lerobot.processor import make_default_processors

teleop_action, robot_action, robot_observation = make_default_processors()
action = {{"shoulder_pan.pos": 1.0, "gripper.pos": 0.5}}
observation = {{"shoulder_pan.pos": 0.9, "front": np.zeros((4, 4, 3), np.uint8)}}
assert robot_action((teleop_action((action, observation)), observation)) == action
assert robot_observation(observation).keys() == observation.keys()
robot_action.save_pretrained({str(tmp_path)!r})
"""
    )
    assert result.returncode == 0, result.stderr


def test_saved_pipeline_with_a_processor_step_loads(tmp_path):
    save = (
        "from lerobot.processor import DataProcessorPipeline, DeviceProcessorStep\n"
        f"DataProcessorPipeline([DeviceProcessorStep(device='cpu')]).save_pretrained({str(tmp_path)!r})\n"
    )
    load = (
        "import sys\n"
        "from lerobot.processor.pipeline import DataProcessorPipeline\n"
        f"p = DataProcessorPipeline.from_pretrained({str(tmp_path)!r}, config_filename='dataprocessorpipeline.json')\n"
        "print('STEPS', [type(s).__name__ for s in p.steps])\n"
        "print('POLICIES', sorted({m.split('.')[2] for m in sys.modules if m.startswith('lerobot.policies.')}))\n"
    )
    for code in (save, load):
        result = run(code)
        assert result.returncode == 0, result.stderr
    assert "STEPS ['DeviceProcessorStep']" in result.stdout
    # A step defined in lerobot.processor must not make the registry import every policy.
    assert "POLICIES []" in result.stdout
