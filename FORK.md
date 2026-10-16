# This fork of lerobot

Branch `feature/xlerobot`, used by Carl's XLeRobot cart. `main` tracks
upstream exactly; every change lives on this branch, and is kept small so
merges from upstream stay cheap. The table below is the complete list;
the smaller source changes also carry a `FORK-LOCAL` comment in place.

The robot itself - plugins, configuration, run scripts, docs - is in the
sibling XLeRobot repo (`../XLeRobot`, branch `rosie-config`), which is also
the uv workspace that installs this fork. **Do not `uv sync` from here.**
See `../XLeRobot/config/README.md`.


### Dependencies (`pyproject.toml`)

- torch is routed to the CUDA wheel index only on `x86_64` linux. Upstream
  routes all linux there, and `cu128` has no aarch64 wheels, so a Pi or
  Jetson cannot resolve torch at all.

  **This entry is live even when you sync from the XLeRobot workspace.**
  uv reads it as part of that resolution, so it is not dormant — adding a
  second index for `torch` without a disjoint marker makes uv refuse to
  resolve anything:

  ```
  Requirements contain conflicting indexes for package `torch` in split
  `... platform_machine == 'x86_64' and sys_platform == 'linux'`
  ```

  GPU torch for the Jetson is the `jetson` extra in the workspace
  (`uv sync --extra jetson`), scoped to `aarch64` for exactly that reason,
  routing to `cu132`. Note also that newer CUDA indexes **do** publish
  aarch64 wheels, so the premise above is specific to `cu128` rather than a
  permanent fact about ARM.
- `pyrealsense2` may go to >=2.57.7 on ARM linux. Upstream pins <2.57.0, but
  the first aarch64 **cp312** wheels appear in 2.57.7, and lerobot requires
  Python >=3.12.
- **ARM ceilings raised one minor version** for `torch` (<2.13),
  `torchvision` (<0.28) and `torchcodec` (<0.13). The cu132 index starts at
  torch 2.12.0 / torchvision 0.27.0 for aarch64 cp312 and publishes nothing
  older, so upstream's caps exclude every GPU build that exists for the
  Jetson — by exactly one notch each. torchcodec follows because upstream's
  own note says 0.12 needs torch 2.12, and `--extra dataset` cannot resolve
  on ARM otherwise.

  These **widen rather than shift**: 2.7 through 2.12 all stay legal on ARM,
  so rosie resolving from PyPI is unaffected. x86_64 keeps upstream's pins
  verbatim. The cost is that the Jetson now runs torch/torchcodec pairings
  lerobot has not tested — if dataset writing misbehaves, this is the first
  thing to suspect.
- `ipython` in dependencies, `[tool.uv] package = true` and
  `python-preference = "managed"`.

### Source (`src/lerobot/`)

| file | change |
|---|---|
| `utils/rerun_visualization.py` | the display off the control loop: async worker, latest-wins frame slot, decimation before queueing (`LEROBOT_RERUN_IMAGE_FPS`), capture timestamps, cv2 JPEG, accumulating blueprint, images never `static=True` (that wedged the viewer). **The big one; expect it to conflict.** |
| `utils/cycle_timer.py` | the slow-loop warning rate-limited (`LEROBOT_SLOW_LOOP_WARN_S`); statistics unchanged |
| `utils/import_utils.py` | video backend: probe `torchcodec.decoders`, fall back to PyAV, decide once |
| `cameras/opencv/*` | `buffer_size` (V4L2 `CAP_PROP_BUFFERSIZE`), applied before format negotiation |
| `cameras/realsense/camera_realsense.py` | frames copied out of librealsense's recycled buffer, so RGB is safe |
| `robots/utils.py`, `robots/__init__.py` | `stop_motion()`, `disconnect_quietly()` |
| `scripts/lerobot_teleoperate.py`, `scripts/lerobot_record.py` | shutdown order: stop motion, disconnect robot and teleop, then encode and flush the display |

`.gitignore` adds `.claude/`.

Expect `pyproject.toml` and `rerun_visualization.py` to conflict on upstream
merges. Nothing else should.
