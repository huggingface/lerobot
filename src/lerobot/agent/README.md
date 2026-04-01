# Agentic manipulation

Stereo RGB + depth, a **VLM** (vision-language model) for object boxes, and a **reasoning LLM** that chooses **which** object to act on and whether to **pick**, **place**, **retry**, or **stop**. Arm motion uses grasp planning, camera→robot transforms, IK, and the shared motion executor.

---

## How the agentic flow works

The runtime loop in `lerobot_agentic_manipulate.py` repeats roughly:

1. **Observe**  
   - Read RGB and depth from the robot’s configured camera (e.g. Oak-D, RealSense).  
   - Run the VLM on the **task string** (e.g. “pick up the red cube”) to get bounding boxes and labels.  
   - Refine masks (e.g. GrabCut), backproject masked depth with **camera intrinsics**, and produce a list of **`SceneObject`s**: index, label, 3D center/size, distance.  
   - That list is wrapped in a **`SceneObservation`** (task + objects).

2. **Reason**  
   - The scene is turned into short **natural language** (`build_scene_summary`) and sent to the reasoning LLM with a fixed system prompt (`ReasoningAgent` in `agentic_flow.py`).  
   - The model must answer with **one JSON line**, for example:  
     `{"action": "pick", "object_index": 0}` or `place` with `place_xyz`, or `done` / `fail` / `retry`.  
   - This is parsed into an **`AgentAction`**.

3. **Act**  
   - For **`pick`**, a grasp planner builds **waypoints** (pre-grasp, grasp, lift, etc.) from the chosen object’s 3D state.  
   - Poses are converted from **camera-related frame** to **robot base** using **`camera_to_robot_tf`** or, for a **gripper-mounted camera**, FK: `T_base_cam = FK(q) @ T_ee_cam`.  
   - IK + `motion_executor` send joint commands to the robot (unless **`--dry-run`**).

**Timing:** The main loop can run at camera rate. **VLM detection** and **LLM reasoning** are throttled by timers (`detect_period_s`, `reason_period_s`) so the expensive models are not called every frame. Optional **tracking** can update the box between VLM calls. **`agent_step_mode`**: **`macro`** runs a full pick/place plan per LLM step; **`micro`** advances **one waypoint per loop tick** so motion stays smooth without re-querying the LLM every tick.

**Logging:** Optional JSONL traces record scene + LLM action per reasoning step. Rerun can show camera streams and a 3D scene (`--display_data`, `--display_sim3d`).

For a diagram of perception → grasp → IK, see `../perception/PIPELINE.md`.

---

## How to run

### 1. Install

From the LeRobot repo root (editable install so scripts are on your PATH):

```bash
pip install -e ".[your-extras]"   # or: pip install -e .
```

Confirm the CLI is available:

```bash
lerobot-agentic-manipulate --help
```

### 2. Environment

For the default Gemini-style reasoning / VLM backends, set an API key, for example:

```bash
export GEMINI_API_KEY=your-key
```

(Other backends use their own keys or local models; see flags under `--vlm.*` and `--agent.*`.)

### 3. Example (robot + Oak-D, camera in front of workspace)

Typical layout: **camera → object → robot** (camera looks at the table; robot reaches forward). Use OpenCV-style camera axes and a **base-frame** translation from camera to robot (tune for your rig):

```bash
lerobot-agentic-manipulate \
  --robot.type=so100_follower \
  --robot.cameras='{"front": {"type": "oakd", "fps": 30, "width": 640, "height": 480, "use_depth": true}}' \
  --task="pick up the red cube" \
  --urdf=./SO101/so101_new_calib.urdf \
  --vlm.backend=gemini \
  --camera_frame_convention=opencv \
  --camera_to_robot_tf="0.4,0,0.1,0,0,0"
```

Adjust `robot.type`, camera JSON, `urdf` path, and `camera_to_robot_tf` to match your hardware.

**Gripper-mounted (eye-in-hand) camera:** use `--camera-mount=gripper` and `--gripper-camera-tf` (see the long docstring in `lerobot_agentic_manipulate.py`).

### 4. Example (no robot — perception + reasoning only)

```bash
lerobot-agentic-manipulate \
  --dry-run \
  --dry-run-image=/path/to/image.png \
  --task="pick up the red cube" \
  ...
```

Use **`--help`** and the module docstring at the top of `../scripts/lerobot_agentic_manipulate.py` for dry-run depth stubs, Claude/OpenAI backends, Rerun (`--display_data`, `--display_sim3d`), depth fusion, and micro-step options.

---

## Related code

| Area | Location |
|------|----------|
| Reasoning loop & actions | `agentic_flow.py`, `manipulation_skills.py` |
| Main CLI loop | `../scripts/lerobot_agentic_manipulate.py` |
| Perception (depth, VLM, grasp hints) | `../perception/` — `../perception/PIPELINE.md` |
| Depth processor | `../processor/depth_perception_processor.py` |
| 3D Rerun + motion | `../utils/manipulation_sim3d.py`, `../utils/motion_executor.py` |

## Other scripts

- `lerobot-agentic-mcp` — MCP server exposing observe / plan / pick / place as tools (if installed).
- `lerobot-auto-manipulate`, `lerobot-localize-objects`, `lerobot-track-tags`, mesh/tag helpers — under `../scripts/`.

## Requirements

Robot and camera configs follow LeRobot’s `make_robot_from_config` patterns (e.g. SO follower + Oak-D / RealSense / OpenCV). Cloud VLMs/LLMs need the matching API keys and `--vlm` / `--agent` flags.
