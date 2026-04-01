#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
MCP server: DimOS-style discrete tools for agentic manipulation with OAK-D (stereo depth).

Expose observe → reason → act as separate MCP tools so Cursor, Claude Code, or other MCP
clients can drive the SO arm with the same perception stack as ``lerobot-agentic-manipulate``.

Install::

    pip install 'lerobot[agentic-mcp,perception,oakd,feetech]'

With the default ``stdio`` transport the process **blocks and waits** after connect: it does not
move the robot until an MCP client (Cursor, etc.) sends tool calls. Use ``mcp call …`` from the
terminal for a one-shot skill without a client.

Run (stdio transport, typical for MCP clients)::

    lerobot-agentic-mcp \\
        --robot.type=so101_follower \\
        --robot.port=/dev/tty.usbserial-* \\
        --robot.cameras='{"front": {"type": "oakd", "fps": 30, "width": 640, "height": 480, "use_depth": true}}' \\
        --urdf=./SO101/so101_new_calib.urdf \\
        --task=\"pick up the red cube\" \\
        --vlm.backend=gemini \\
        --camera_frame_convention=opencv \\
        --camera_to_robot_tf=\"0.4,0,0.1,0,0,0\"

Set ``GEMINI_API_KEY`` (or OpenAI-style keys if ``--agent.use-gemini=false``).

Skill names and flow follow the DimOS agent docs (``observe``, typed skills, optional HTTP MCP).
See ``lerobot.agent.manipulation_skills`` and https://github.com/dimensionalOS/dimos

Typical flow: ``set_task`` → ``observe`` → ``plan_next_action`` → ``pick_object`` / ``place_at``.
Use ``manipulation_step`` for one full observe→plan→act cycle. DimOS-style CLI::

    lerobot-agentic-mcp mcp list-tools
    lerobot-agentic-mcp mcp call observe --robot.type=so101_follower ...
    lerobot-agentic-mcp mcp call pick_object --json-args '{\"object_index\": 0}' --robot.type=...

HTTP/SSE (like DimOS ``--mcp-port``)::

    lerobot-agentic-mcp ... --mcp-transport=sse --mcp-host=127.0.0.1 --mcp-port=9990

Example Cursor MCP config (stdio): ``command`` = ``lerobot-agentic-mcp`` with the same robot/camera flags.

Rerun camera stream (opens viewer; logs each ``observe`` / loop frame)::

    ... --display_data=true
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import deque
from typing import Any

import draccus
import numpy as np
import rerun as rr

from lerobot.agent import AgentAction, ReasoningAgent, SceneObservation, build_scene_summary
from lerobot.agent.manipulation_skills import (
    MANIPULATION_SKILL_METAS,
    build_skills_system_appendix,
    list_skills_json,
    validate_tool_args,
)
from lerobot.cameras.oakd.configuration_oakd import OAKDCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower import SOFollowerRobotConfig  # noqa: F401
from lerobot.utils.motion_executor import MotionExecutionConfig, execute_waypoints
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun, log_stereo_pair_to_rerun

from lerobot.scripts.lerobot_agentic_manipulate import (
    SO100_MOTOR_NAMES,
    AgenticManipulateConfig,
    camera_opencv_to_robot_rotation,
    parse_tf_string,
    parse_xyz_string,
    _observe_from_images,
)

logger = logging.getLogger(__name__)


def _action_to_jsonable(action: AgentAction) -> dict[str, Any]:
    d: dict[str, Any] = {
        "action": action.action,
        "object_index": action.object_index,
        "success": action.success,
        "reason": action.reason,
    }
    if action.place_xyz is not None:
        d["place_xyz"] = [float(x) for x in action.place_xyz]
    else:
        d["place_xyz"] = None
    return d


def _scene_objects_json(scene: SceneObservation) -> list[dict[str, Any]]:
    out = []
    for o in scene.objects:
        out.append(
            {
                "index": o.index,
                "label": o.label,
                "center_xyz": [float(x) for x in o.center_xyz],
                "size_xyz": [float(x) for x in o.size_xyz],
                "distance_m": float(o.distance_m),
            }
        )
    return out


class AgenticMCPRuntime:
    """Holds robot, perception, and planner; updated by each observe call."""

    def __init__(self, cfg: AgenticManipulateConfig):
        self.cfg = cfg
        self.robot = None
        self.detector = None
        self.reasoning_agent = None
        self.planner = None
        self.kinematics = None
        self.motion_exec = MotionExecutionConfig()
        self.intrinsics: dict[str, float] = {}
        self.camera_key = cfg.camera_key
        self.cam_to_robot = np.eye(4, dtype=np.float64)
        self.depth_history: deque[np.ndarray] = deque(maxlen=max(1, int(cfg.depth_fusion_frames)))

        self.last_scene: SceneObservation | None = None
        self.last_detections: list = []
        self._connected = False
        self._rerun_frame_idx = 0
        self._rerun_compress: bool = False

    def setup(self) -> None:
        from lerobot.model.kinematics import RobotKinematics
        from lerobot.perception.grasp_planner import GraspPlanner
        from lerobot.perception.vlm_detector import VLMDetector
        from lerobot.processor.depth_perception_processor import fuse_depth_temporal_median

        self._fuse_depth = fuse_depth_temporal_median

        cfg = self.cfg
        self.cam_to_robot = parse_tf_string(cfg.camera_to_robot_tf)
        if (cfg.camera_frame_convention or "").lower() == "opencv":
            t = self.cam_to_robot[:3, 3].copy()
            R = camera_opencv_to_robot_rotation(flip_lateral=cfg.camera_flip_lateral)
            self.cam_to_robot = np.eye(4, dtype=np.float64)
            self.cam_to_robot[:3, :3] = R
            self.cam_to_robot[:3, 3] = t

        self.detector = VLMDetector(
            backend=cfg.vlm.backend,
            model_id=cfg.vlm.model_id,
            device=cfg.vlm.device or None,
            api_key=cfg.vlm.api_key or None,
            cloud_model=cfg.vlm.cloud_model,
            cloud_base_url=cfg.vlm.cloud_base_url or None,
        )
        self.reasoning_agent = ReasoningAgent(
            api_key=cfg.agent.api_key or None,
            base_url=cfg.agent.base_url or None,
            model=cfg.agent.model,
            is_gemini=cfg.agent.use_gemini,
        )
        self.planner = GraspPlanner(
            camera_to_robot_tf=self.cam_to_robot,
            pre_grasp_height_m=cfg.pre_grasp_height,
            lift_height_m=cfg.lift_height,
            arm_base_xyz_m=parse_xyz_string(cfg.arm_base_xyz),
            min_reach_m=cfg.min_reach_m,
            max_reach_m=cfg.max_reach_m,
            min_grasp_z_m=cfg.min_grasp_z_m,
            max_grasp_z_m=cfg.max_grasp_z_m,
            approach_xy_retract_m=cfg.approach_xy_retract_m,
        )
        self.kinematics = RobotKinematics(
            urdf_path=cfg.urdf,
            target_frame_name=cfg.ee_frame,
            joint_names=SO100_MOTOR_NAMES,
        )
        self.motion_exec = MotionExecutionConfig(
            use_cartesian_interp=cfg.motion_use_cartesian,
            cartesian_step_m=cfg.motion_cartesian_step_m,
            min_steps_per_segment=cfg.motion_min_steps_per_segment,
            inter_step_sleep_s=cfg.motion_inter_step_sleep_s,
            settle_threshold_deg=cfg.motion_settle_threshold_deg,
            settle_timeout_s=cfg.motion_settle_timeout_s,
        )

        if cfg.dry_run:
            self._connected = True
            h, w = 480, 640
            self.intrinsics = {
                "fx": 525.0,
                "fy": 525.0,
                "cx": w / 2.0,
                "cy": h / 2.0,
                "depth_scale": 0.001,
            }
            self._init_rerun_if_needed()
            return

        self.robot = make_robot_from_config(cfg.robot)
        self.robot.connect()
        depth_camera = self.robot.cameras.get(self.camera_key)
        if depth_camera is None:
            raise ValueError(
                f"Camera '{self.camera_key}' not found. Available: {list(self.robot.cameras.keys())}"
            )
        if hasattr(depth_camera, "get_depth_intrinsics"):
            try:
                self.intrinsics = depth_camera.get_depth_intrinsics()
            except Exception as e:
                logger.warning("Could not get depth intrinsics: %s", e)
        self._connected = True
        self._init_rerun_if_needed()

    def _init_rerun_if_needed(self) -> None:
        cfg = self.cfg
        if not cfg.display_data:
            return
        self._rerun_compress = (
            True
            if (cfg.display_ip is not None and cfg.display_port is not None)
            else bool(cfg.display_compressed_images)
        )
        init_rerun(session_name="agentic_mcp", ip=cfg.display_ip, port=cfg.display_port)
        logger.info(
            "Rerun logging on each observe() (observation.%s, observation.%s_depth_color).",
            self.camera_key,
            self.camera_key,
        )

    def disconnect(self) -> None:
        if self.cfg.display_data:
            try:
                rr.rerun_shutdown()
            except Exception:
                pass
        if self.robot is not None:
            try:
                self.robot.disconnect()
            except Exception:
                pass
            self.robot = None
        self._connected = False

    def _fused_depth(self, depth_curr: np.ndarray) -> np.ndarray:
        self.depth_history.append(np.asarray(depth_curr, dtype=np.uint16).copy())
        if self.cfg.depth_fusion_frames > 1:
            return self._fuse_depth(list(self.depth_history))
        return self.depth_history[-1]

    def observe(self) -> str:
        from lerobot.processor.depth_perception_processor import compute_object_state

        cfg = self.cfg
        color_fn = cfg.color_filter_min_fraction if cfg.color_filter_min_fraction > 0 else None
        task = cfg.task

        if cfg.dry_run:
            import cv2

            if not cfg.dry_run_image or not str(cfg.dry_run_image).strip():
                return json.dumps({"ok": False, "error": "dry_run requires --dry-run-image"})
            rgb_np = cv2.imread(str(cfg.dry_run_image).strip())
            if rgb_np is None:
                return json.dumps({"ok": False, "error": f"Could not load image: {cfg.dry_run_image}"})
            rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)
            if cfg.dry_run_depth and str(cfg.dry_run_depth).strip():
                depth_np = np.load(str(cfg.dry_run_depth).strip())
                if depth_np.ndim == 3:
                    depth_np = depth_np.squeeze()
            else:
                h, w = rgb_np.shape[:2]
                depth_np = np.full((h, w), 500, dtype=np.uint16)
            depth_fused = self._fused_depth(depth_np)
            scene, detections, _, _ = _observe_from_images(
                rgb_np,
                depth_fused,
                self.detector,
                compute_object_state,
                self.intrinsics,
                task,
                color_filter_min_fraction=color_fn,
            )
        else:
            assert self.robot is not None
            obs = self.robot.get_observation()
            rgb = obs.get(self.camera_key)
            depth = obs.get(f"{self.camera_key}_depth")
            if rgb is None or depth is None:
                self.last_scene = SceneObservation(objects=[], task=task)
                self.last_detections = []
                return json.dumps(
                    {
                        "ok": True,
                        "warning": "missing_rgb_or_depth",
                        "summary": build_scene_summary(self.last_scene),
                        "objects": [],
                    }
                )
            rgb_np = np.asarray(rgb)
            depth_fused = self._fused_depth(np.asarray(depth))
            scene, detections, _, _ = _observe_from_images(
                rgb_np,
                depth_fused,
                self.detector,
                compute_object_state,
                self.intrinsics,
                task,
                color_filter_min_fraction=color_fn,
            )

        self.last_scene = scene
        self.last_detections = detections

        if cfg.display_data and rgb_np.size > 0 and depth_fused.size > 0:
            self._rerun_frame_idx += 1
            log_stereo_pair_to_rerun(
                camera_key=self.camera_key,
                rgb_hwc=rgb_np,
                depth_u16_hw=depth_fused,
                frame_sequence=self._rerun_frame_idx,
                compress_images=self._rerun_compress,
            )

        payload = {
            "ok": True,
            "summary": build_scene_summary(scene),
            "objects": _scene_objects_json(scene),
            "task": task,
        }
        return json.dumps(payload, indent=2)

    def reason(self) -> str:
        if not self._connected:
            return json.dumps({"ok": False, "error": "Not connected; server misconfigured."})
        if self.last_scene is None:
            return json.dumps(
                {
                    "ok": False,
                    "error": "No scene yet. Call observe first.",
                }
            )
        action = self.reasoning_agent.reason(self.last_scene)
        return json.dumps({"ok": True, "action": _action_to_jsonable(action)}, indent=2)

    def pick(self, object_index: int) -> str:
        if not self._connected:
            return json.dumps({"ok": False, "error": "Not connected."})
        if self.last_scene is None or not self.last_detections:
            return json.dumps({"ok": False, "error": "No observations. Call observe first."})
        cfg = self.cfg
        idx = int(object_index)
        if idx < 0 or idx >= len(self.last_detections):
            return json.dumps(
                {
                    "ok": False,
                    "error": f"Invalid object_index {idx}; have {len(self.last_detections)} detections.",
                }
            )
        det, state = self.last_detections[idx]
        center = state["obj_center_xyz"]
        size = state["obj_size_xyz"]

        if cfg.dry_run:
            return json.dumps(
                {
                    "ok": True,
                    "dry_run": True,
                    "message": f"Would pick object {idx} '{det.label}' at {center.tolist()}",
                }
            )

        assert self.robot is not None
        waypoints = self.planner.plan_pick(center, size, det.label)
        execute_waypoints(self.robot, self.kinematics, waypoints, SO100_MOTOR_NAMES, self.motion_exec)
        result: dict[str, Any] = {"ok": True, "picked_index": idx, "label": det.label}
        if cfg.place_target:
            place_xyz = np.array([float(v) for v in cfg.place_target.split(",")], dtype=np.float64)
            place_wps = self.planner.plan_place(place_xyz)
            execute_waypoints(self.robot, self.kinematics, place_wps, SO100_MOTOR_NAMES, self.motion_exec)
            result["placed_after_pick"] = True
            result["place_target"] = place_xyz.tolist()
        return json.dumps(result, indent=2)

    def place(self, x: float, y: float, z: float) -> str:
        if not self._connected:
            return json.dumps({"ok": False, "error": "Not connected."})
        if self.cfg.dry_run:
            return json.dumps({"ok": True, "dry_run": True, "message": f"Would place at ({x},{y},{z})"})
        assert self.robot is not None
        place_xyz = np.array([x, y, z], dtype=np.float64)
        place_wps = self.planner.plan_place(place_xyz)
        execute_waypoints(self.robot, self.kinematics, place_wps, SO100_MOTOR_NAMES, self.motion_exec)
        return json.dumps({"ok": True, "place_xyz": [x, y, z]}, indent=2)

    def set_task(self, task: str) -> str:
        self.cfg.task = task
        if self.last_scene is not None:
            self.last_scene.task = task
        return json.dumps({"ok": True, "task": task})

    def wait(self, seconds: float) -> str:
        """DimOS-style pause skill."""
        time.sleep(max(0.0, float(seconds)))
        return f"waited {float(seconds):.3f}s"

    def mcp_status_json(self) -> str:
        names = [m.name for m in MANIPULATION_SKILL_METAS]
        return json.dumps(
            {
                "ok": True,
                "server": "lerobot-agentic-mcp",
                "connected": self._connected,
                "task": self.cfg.task,
                "mcp_transport": self.cfg.mcp_transport,
                "mcp_host": self.cfg.mcp_host,
                "mcp_port": self.cfg.mcp_port,
                "skills": names,
            },
            indent=2,
        )

    def manipulation_step(self) -> str:
        """One DimOS-like closed-loop step: observe → LLM plan → execute pick/place if applicable."""
        self.observe()
        if self.last_scene is None:
            return json.dumps({"ok": False, "error": "observe produced no scene"}, indent=2)
        action = self.reasoning_agent.reason(self.last_scene)
        plan = _action_to_jsonable(action)
        out: dict[str, Any] = {"ok": True, "planned_action": plan, "executed": None}

        if action.action in ("done", "fail", "retry"):
            out["terminal"] = action.action
            if action.reason:
                out["reason"] = action.reason
            if action.success is not None:
                out["success"] = action.success
            return json.dumps(out, indent=2)

        if action.is_pick() and action.object_index is not None:
            out["executed"] = "pick_object"
            out["motion_result"] = json.loads(self.pick(int(action.object_index)))
            return json.dumps(out, indent=2)

        if action.is_place() and action.place_xyz is not None:
            x, y, z = action.place_xyz
            out["executed"] = "place_at"
            out["motion_result"] = json.loads(self.place(x, y, z))
            return json.dumps(out, indent=2)

        out["ok"] = False
        out["error"] = f"Unhandled planner action: {action.action}"
        return json.dumps(out, indent=2)


def _require_mcp():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        raise SystemExit(
            "The 'mcp' package is required. Install with:\n"
            "  pip install 'lerobot[agentic-mcp]'\n"
            "For OAK-D and perception stack also add: perception, oakd extras."
        ) from e
    return FastMCP


def _register_manipulation_skills_mcp(mcp: Any, runtime: AgenticMCPRuntime) -> None:
    """Register DimOS-style @skill-equivalent tools with full descriptions."""
    by_name = {m.name: m for m in MANIPULATION_SKILL_METAS}

    @mcp.tool(name="observe", description=by_name["observe"].description.strip())
    def observe() -> str:
        return runtime.observe()

    @mcp.tool(name="plan_next_action", description=by_name["plan_next_action"].description.strip())
    def plan_next_action() -> str:
        return runtime.reason()

    @mcp.tool(name="pick_object", description=by_name["pick_object"].description.strip())
    def pick_object(object_index: int) -> str:
        return runtime.pick(object_index)

    @mcp.tool(name="place_at", description=by_name["place_at"].description.strip())
    def place_at(x: float, y: float, z: float) -> str:
        return runtime.place(x, y, z)

    @mcp.tool(name="set_task", description=by_name["set_task"].description.strip())
    def set_task(task: str) -> str:
        return runtime.set_task(task)

    @mcp.tool(name="wait", description=by_name["wait"].description.strip())
    def wait(seconds: float) -> str:
        return runtime.wait(seconds)

    @mcp.tool(name="mcp_status", description=by_name["mcp_status"].description.strip())
    def mcp_status() -> str:
        return runtime.mcp_status_json()

    @mcp.tool(name="manipulation_step", description=by_name["manipulation_step"].description.strip())
    def manipulation_step() -> str:
        return runtime.manipulation_step()

    @mcp.tool(name="list_registered_skills", description=by_name["list_registered_skills"].description.strip())
    def list_registered_skills() -> str:
        return list_skills_json()


@parser.wrap()
def agentic_mcp_main(cfg: AgenticManipulateConfig) -> None:
    init_logging()
    FastMCP = _require_mcp()

    runtime = AgenticMCPRuntime(cfg)
    try:
        runtime.setup()
    except ModuleNotFoundError as e:
        logger.exception("Failed to start MCP runtime")
        if e.name == "scservo_sdk":
            raise SystemExit(
                "Startup failed: SO / Feetech arms need the Feetech SDK (import scservo_sdk). "
                "Install: pip install 'lerobot[feetech]'   (PyPI: feetech-servo-sdk, not scservo-sdk)."
            ) from e
        raise SystemExit(f"Startup failed: {e}") from e
    except Exception as e:
        logger.exception("Failed to start MCP runtime")
        raise SystemExit(f"Startup failed: {e}") from e

    instructions = (
        "LeRobot SO-arm manipulation MCP, modeled after Dimensional DimOS skills + MCP server.\n"
        "The host LLM is the agent; this process exposes physical/perception skills as tools.\n\n"
        + build_skills_system_appendix()
    )
    mcp = FastMCP(
        name="lerobot-oakd-agentic",
        instructions=instructions,
        host=cfg.mcp_host,
        port=cfg.mcp_port,
        mount_path=cfg.mcp_mount_path,
    )
    _register_manipulation_skills_mcp(mcp, runtime)

    transport = (cfg.mcp_transport or "stdio").strip().lower()
    if transport == "sse":
        try:
            import uvicorn  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "mcp_transport=sse requires uvicorn. Install: pip install 'lerobot[agentic-mcp]'"
            ) from e
        logger.info(
            "MCP SSE server (DimOS-style): http://%s:%s/ — mount_path=%s",
            cfg.mcp_host,
            cfg.mcp_port,
            cfg.mcp_mount_path,
        )

    if transport not in ("stdio", "sse"):
        raise SystemExit(f"Unknown mcp_transport={transport!r}; use 'stdio' or 'sse'.")

    if transport == "stdio":
        skill_names = ", ".join(m.name for m in MANIPULATION_SKILL_METAS)
        logger.info(
            "MCP stdio server is ready and waiting on stdin for a client (e.g. Cursor MCP). "
            "This process will stay quiet until the client connects and calls tools (%s). "
            "To run one skill from the shell without a client: "
            "lerobot-agentic-mcp mcp call observe --robot.type=... (same flags as above).",
            skill_names,
        )

    try:
        mcp.run(transport=transport, mount_path=cfg.mcp_mount_path)
    finally:
        runtime.disconnect()


_MCP_CLI_HELP = """DimOS-style MCP helper (no long-running server).

  lerobot-agentic-mcp mcp list-tools
      Print JSON skill list (like dimos mcp list-tools).

  lerobot-agentic-mcp mcp status
      Print static server metadata.

  lerobot-agentic-mcp mcp call <skill> [--json-args '{{...}}'] [robot config flags...]
      One-shot skill invocation (connects robot, runs skill, exits).

Examples:
  lerobot-agentic-mcp mcp call observe --robot.type=so101_follower --robot.port=/dev/ttyUSB0 ...
  lerobot-agentic-mcp mcp call pick_object --json-args '{{\"object_index\": 0}}' --robot.type=...
"""


def _parse_mcp_call_argv(argv: list[str]) -> tuple[str, dict[str, Any], list[str]]:
    if not argv:
        raise SystemExit("mcp call <skill_name> ...\n" + _MCP_CLI_HELP)
    tool = argv[0]
    kwargs: dict[str, Any] = {}
    config_args: list[str] = []
    i = 1
    while i < len(argv):
        if argv[i] == "--json-args" and i + 1 < len(argv):
            kwargs.update(json.loads(argv[i + 1]))
            i += 2
        else:
            config_args.append(argv[i])
            i += 1
    return tool, kwargs, config_args


def _dispatch_skill(runtime: AgenticMCPRuntime, tool: str, kwargs: dict[str, Any]) -> str:
    if tool == "observe":
        return runtime.observe()
    if tool == "plan_next_action":
        return runtime.reason()
    if tool == "pick_object":
        return runtime.pick(int(kwargs["object_index"]))
    if tool == "place_at":
        return runtime.place(float(kwargs["x"]), float(kwargs["y"]), float(kwargs["z"]))
    if tool == "set_task":
        return runtime.set_task(str(kwargs["task"]))
    if tool == "wait":
        return runtime.wait(float(kwargs["seconds"]))
    if tool == "mcp_status":
        return runtime.mcp_status_json()
    if tool == "manipulation_step":
        return runtime.manipulation_step()
    if tool == "list_registered_skills":
        return list_skills_json()
    raise SystemExit(f"Unknown skill {tool!r}. Use: mcp list-tools")


def _run_dimos_style_mcp_cli(argv: list[str]) -> None:
    from lerobot.configs.parser import _normalize_cli_args

    if not argv or argv[0] in ("-h", "--help"):
        print(_MCP_CLI_HELP)
        return
    if argv[0] == "list-tools":
        print(list_skills_json())
        return
    if argv[0] == "status":
        print(
            json.dumps(
                {
                    "server": "lerobot-agentic-mcp",
                    "note": "Long-running MCP: run without the 'mcp' subcommand (stdio or --mcp-transport=sse).",
                    "default_sse_port": 9990,
                },
                indent=2,
            )
        )
        return
    if argv[0] != "call":
        raise SystemExit(f"Unknown subcommand {argv[0]!r}.\n{_MCP_CLI_HELP}")

    tool, kwargs, config_args = _parse_mcp_call_argv(argv[1:])
    err = validate_tool_args(tool, kwargs)
    if err:
        raise SystemExit(err)

    init_logging()
    cfg = draccus.parse(AgenticManipulateConfig, args=_normalize_cli_args(config_args))
    runtime = AgenticMCPRuntime(cfg)
    runtime.setup()
    try:
        print(_dispatch_skill(runtime, tool, kwargs))
    finally:
        runtime.disconnect()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "mcp":
        _run_dimos_style_mcp_cli(sys.argv[2:])
        return
    agentic_mcp_main()


if __name__ == "__main__":
    main()
