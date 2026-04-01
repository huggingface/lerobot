# Agentic manipulation

Stereo RGB + depth, VLM detections, and an LLM choose **which** object to act on and whether to pick, place, retry, or stop. Motion runs through the shared motion executor (waypoints / micro-steps).

## Main entry point

Install the package in editable mode, then run:

```bash
lerobot-agentic-manipulate --help
```

The script docstring has full flag documentation (dry-run, camera frames, gripper-mounted camera, Rerun, 3D preview). See `src/lerobot/scripts/lerobot_agentic_manipulate.py`.

## Related code

| Area | Location |
|------|----------|
| Reasoning loop & actions | `agentic_flow.py`, `manipulation_skills.py` |
| Perception (depth, tags, VLM, grasp hints) | `../perception/` — overview in `../perception/PIPELINE.md` |
| Depth processor | `../processor/depth_perception_processor.py` |
| 3D Rerun logging | `../utils/manipulation_sim3d.py`, `../utils/motion_executor.py` |

## Other scripts

- `lerobot-agentic-mcp` — MCP server wrapper (if installed).
- `lerobot-auto-manipulate`, `lerobot-localize-objects`, `lerobot-track-tags`, mesh/tag helpers — under `src/lerobot/scripts/`.

## Requirements

Cloud VLM backends need the corresponding API keys (e.g. `GEMINI_API_KEY`). Robot and camera configs follow the usual LeRobot `make_robot_from_config` patterns (e.g. SO follower + Oak-D / RealSense / OpenCV).
