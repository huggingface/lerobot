# Agentic manipulation (fork)

This branch extends [LeRobot](https://github.com/huggingface/lerobot) with an **agentic manipulation** stack: stereo RGB + depth, VLM detections, LLM reasoning, and motion execution (pick / place / done / retry). Entry point: `lerobot-agentic-manipulate`.

**Fork branch (source of truth for this work):**  
[https://github.com/arminforoughi/lerobot/tree/feature/agentic-manipulate](https://github.com/arminforoughi/lerobot/tree/feature/agentic-manipulate)

**Open a pull request from this branch:**  
[https://github.com/arminforoughi/lerobot/pull/new/feature/agentic-manipulate](https://github.com/arminforoughi/lerobot/pull/new/feature/agentic-manipulate)

---

## Install

From the repo root (editable install, with robot extras as needed, e.g. SO-100 follower):

```bash
pip install -e ".[feetech]"   # example: adjust extras for your hardware
```

Optional MCP server (stdio / HTTP-style SSE) for agentic tools:

```bash
pip install -e ".[agentic-mcp]"
```

---

## Main CLI

| Command | Purpose |
|--------|---------|
| `lerobot-agentic-manipulate` | Full loop: observe → LLM/VLM → act |
| `lerobot-agentic-mcp` | MCP server exposing agentic manipulation tools |
| `lerobot-auto-manipulate` | Simpler auto manipulation path |
| `lerobot-localize-objects` | Object localization utilities |
| `lerobot-track-tags` / `lerobot-make-tag` | AprilTag-style tracking / tag assets |
| `lerobot-realsense-mesh` / `lerobot-stereo-mesh` | Depth / stereo mesh helpers |
| `lerobot-scan-motors` | Motor bus discovery |

Detailed flags, coordinate frames, dry-run, Rerun 3D preview, and gripper-mounted camera notes are in the module docstring:

- `src/lerobot/scripts/lerobot_agentic_manipulate.py`

Perception pipeline notes:

- `src/lerobot/perception/PIPELINE.md`

---

## Minimal example (real robot)

Set your VLM cloud key (e.g. Gemini), then run with your robot type, camera JSON, URDF, and base–camera transform. Example shape (see the script docstring for full flags):

```bash
export GEMINI_API_KEY=your-key
lerobot-agentic-manipulate \
  --robot.type=so100_follower \
  --robot.cameras='{"front": {"type": "oakd", "fps": 30, "width": 640, "height": 480, "use_depth": true}}' \
  --task="pick up the red cube" \
  --urdf=./SO101/so101_new_calib.urdf \
  --vlm.backend=gemini \
  --camera_frame_convention=opencv \
  --camera_to_robot_tf="0.4,0,0.1,0,0,0"
```

**Dry-run (no arm):** use `--dry-run` and `--dry-run-image=/path/to/image.png` as documented in the script.

---

## Layout in this repo

- `src/lerobot/agent/` — agent flow and manipulation skills  
- `src/lerobot/perception/` — scene, VLM, depth, tags, grasp planning  
- `src/lerobot/processor/depth_perception_processor.py` — depth in the processing pipeline  
- `src/lerobot/utils/motion_executor.py`, `manipulation_sim3d.py`, `urdf_*` — motion and 3D helpers  
- `robot/` — small top-level helpers (`main.py` / `config.yaml` may drive local experiments)

Large third-party CAD trees (e.g. `SO-ARM100/` with its own `.git`) are not committed; add URDF/meshes you need under paths referenced by `--urdf`.
