# Hardware configs for Carl's XLeRobot

Configs for the two SO-101 arms, their leaders, and the head RealSense.
Run them with stock lerobot CLIs:

```bash
lerobot-teleoperate --config_path=config/left-arm.yaml
lerobot-teleoperate --config_path=config/right-arm.yaml
lerobot-teleoperate --config_path=config/bi-arms.yaml
```

`rosie.yaml` uses `--robot.type=xlerobot`, which comes from the XLeRobot
plugin rather than from this repo — see below.

## What this fork changes in lerobot

Deliberately almost nothing. Only `pyproject.toml`, so that merges from
upstream stay cheap:

- torch is routed to the CUDA wheel index only on `x86_64` linux. Upstream
  routes all linux there, and that index has no aarch64 wheels, so a Pi or
  Jetson cannot resolve torch at all.
- `pyrealsense2` may go to >=2.57.7 on ARM linux. Upstream pins <2.57.0, but
  the first aarch64 **cp312** wheels appear in 2.57.7, and lerobot requires
  Python >=3.12.
- `ipython` in dependencies, `[tool.uv] package = true` and
  `python-preference = "managed"`.

Expect `pyproject.toml` to conflict on upstream merges. Nothing else should.

## XLeRobot code is NOT in this repo

It used to be copied into `src/lerobot/robots/xlerobot/`,
`src/lerobot/model/SO101Robot.py` and `examples/`. That was removed — the
XLeRobot project now packages its robots as lerobot plugins, which is both
less work to maintain and free of merge conflicts.

lerobot auto-imports any installed package named `lerobot_robot_*`,
`lerobot_teleoperator_*`, etc. (`register_third_party_plugins()` in
`lerobot.utils.import_utils`), so an installed plugin makes its
`--robot.type=...` available on every standard CLI.

From a checkout of the XLeRobot repo:

```bash
pip install -e software/plugins/xlerobot_model
pip install -e software/plugins/lerobot_robot_xlerobot          # 3 omniwheels
pip install -e software/plugins/lerobot_robot_xlerobot_2wheels  # 2 wheels, Feetech servos
pip install -e software/plugins/lerobot_robot_xlerobot_mecanum
pip install -e software/plugins/lerobot_teleoperator_xlerobot_vr
```

Verify:

```bash
python -c "
from lerobot.utils.import_utils import register_third_party_plugins
register_third_party_plugins()
from lerobot.robots.config import RobotConfig
print(RobotConfig.get_choice_class('xlerobot'))
"
```

Examples live in the XLeRobot repo under `software/examples/`.

## Python version

`pyrealsense2` publishes aarch64 wheels for cp39/cp310/cp312 only, and no
sdist. lerobot requires >=3.12. So on ARM, **Python 3.12 exactly**:

```bash
uv python pin 3.12
uv sync --extra feetech --extra intelrealsense --extra viz
```

Leave out `--extra dataset` until you need to record; it pulls `torchcodec`,
which on aarch64 needs >=0.11.0 and torch>=2.11.
