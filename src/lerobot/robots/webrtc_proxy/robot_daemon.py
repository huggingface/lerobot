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

"""robot-side daemon: the long-lived process that keeps the real robot on the user's
machine and serves cloud sessions over WebRTC.

It outlives any single cloud session: connect to the signaling relay, offer, serve
one controller until the link drops, **safe the arm**, then loop and wait for the
next session. This is the persistent counterpart to the cloud ``WebRTCProxyRobot``.

M3 ships the synthetic source (``CaptureAgent`` + ``SyntheticInventory``); M2 swaps
in a real ``so_follower`` + cameras + ``LocalDeviceInventory`` behind the same loop.

Run:
    # 1) start the relay (cloud, here localhost for a same-host demo)
    python -m lerobot.robots.webrtc_proxy.signaling_server --port 8765
    # 2a) synthetic source (no hardware):
    python -m lerobot.robots.webrtc_proxy.robot_daemon --signaling-url ws://127.0.0.1:8765/ws
    # 2b) a real robot (any registered type, draccus --robot.*):
    python -m lerobot.robots.webrtc_proxy.robot_daemon --signaling-url ws://127.0.0.1:8765/ws \
        --robot.type=so101_follower --robot.port=/dev/tty... --robot.cameras='{...}'
    # 3) cloud side: WebRTCProxyRobotConfig(signaling_url="ws://127.0.0.1:8765/ws").connect()
       (declare the same motors + cameras the daemon logs on connect)
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
from dataclasses import dataclass

from ..config import RobotConfig
from .capture_agent import CaptureAgent
from .configuration_webrtc_proxy import SO100_MOTORS
from .control import DeviceInventory, LocalDeviceInventory, SyntheticInventory, _silence_native_stderr
from .signaling import SignalingClosedError, WebSocketSignaling

logger = logging.getLogger(__name__)


@dataclass
class _RobotWrap:
    """draccus wrapper so ``--robot.type=… --robot.port=…`` parses into a RobotConfig.

    Module-level (not nested in ``_build_robot``) so draccus's get_type_hints can resolve
    the ``RobotConfig`` annotation under ``from __future__ import annotations``.
    """

    robot: RobotConfig


def _build_robot(robot_args: list[str]):
    """Parse ``--robot.type=… --robot.port=…`` (draccus) into a connected lerobot Robot.

    Robot-agnostic: any robot registered with ``RobotConfig`` works (so100/101, koch, omx,
    lekiwi, …) — the capture agent only uses the generic ``get_observation`` /
    ``send_action`` / ``bus`` interface. Returns ``(robot, motors, cameras)`` where motors
    and cameras (name -> (h, w)) are derived from the robot's own ``observation_features``,
    so the streamed schema always matches the real hardware. The cloud
    ``WebRTCProxyRobotConfig`` must declare the same motors + cameras.
    """
    import draccus

    # Robots register RobotConfig choices but NOT camera ones; import the camera configs so
    # `--robot.cameras="{...type: opencv...}"` resolves. Same set lerobot-record imports.
    from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
    from lerobot.cameras.reachy2_camera import Reachy2CameraConfig  # noqa: F401
    from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
    from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
    from lerobot.utils.import_utils import register_third_party_plugins

    from .. import (  # noqa: F401 - importing registers each RobotConfig draccus choice
        bi_openarm_follower,
        bi_rebot_b601_follower,
        bi_so_follower,
        earthrover_mini_plus,
        hope_jr,
        koch_follower,
        omx_follower,
        openarm_follower,
        reachy2,
        rebot_b601_follower,
        so_follower,
        unitree_g1,
    )
    from ..utils import make_robot_from_config

    register_third_party_plugins()  # let pip-installed robots register too

    cfg = draccus.parse(_RobotWrap, args=robot_args).robot
    robot = make_robot_from_config(cfg)
    robot.connect()
    feats = robot.observation_features
    motors = [k[: -len(".pos")] for k in feats if isinstance(k, str) and k.endswith(".pos")]
    cameras = {name: (shape[0], shape[1]) for name, shape in feats.items() if isinstance(shape, tuple)}
    logger.info("robot %s connected: motors=%s cameras=%s", cfg.type, motors, dict(cameras))
    return robot, motors, cameras


def _open_streaming_camera(index_or_path, fps: int, width: int, height: int):
    """Open an opencv camera for streaming, robustly.

    Try the requested capture size first (cheaper than native+resize for bandwidth/CPU),
    but fall back to the camera's native profile if it can't apply it — OpenCVCamera
    raises rather than silently using another size. Either way the capture loop's
    ``_fit_frame`` (and the cloud's get_observation) resize to the declared obs shape.
    """
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    with _silence_native_stderr():
        for kwargs in ({"fps": fps, "width": width, "height": height}, {}):
            cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=index_or_path, **kwargs))
            try:
                cam.connect(warmup=True)
                logger.info("opened camera %r (%s)", index_or_path, kwargs or "native resolution")
                return cam
            except Exception as e:
                logger.warning("camera %r open at %s failed (%s)", index_or_path, kwargs or "native", e)
                with contextlib.suppress(Exception):
                    cam.disconnect()
    raise RuntimeError(f"could not open camera {index_or_path!r}")


async def run_daemon(
    signaling_url: str,
    session_id: str = "default",
    motors: list[str] | None = None,
    cam_name: str = "front",
    cam_height: int = 480,
    cam_width: int = 640,
    capture_fps: int = 30,
    action_timeout_s: float = 0.5,
    ice_servers: list[str] | None = None,
    inventory: DeviceInventory | None = None,
    camera=None,
    cameras: dict[str, tuple[int, int]] | None = None,  # multi-cam: name -> (height, width)
    cameras_src: dict | None = None,  # multi-cam: name -> opened Camera
    robot=None,
    reliable_state: bool = False,
    reliable_action: bool = False,
    signaling_token: str | None = None,
    transport_backend: str = "aiortc",
    livekit_url: str | None = None,
    livekit_token: str | None = None,
    stop: asyncio.Event | None = None,
    on_agent=None,
) -> None:
    """Serve cloud sessions forever (until ``stop`` is set), one session at a time.

    The camera (if any) is owned by the caller and reused across sessions — only the
    per-session WebRTC peer is rebuilt each loop.
    """
    motors = list(motors or SO100_MOTORS)
    stop = stop or asyncio.Event()
    while not stop.is_set():
        # livekit does its own signaling (url+token); only aiortc needs the WS relay.
        sig = (
            None
            if transport_backend == "livekit"
            else WebSocketSignaling(signaling_url, session_id, role="robot", token=signaling_token)
        )
        agent = CaptureAgent(
            signaling=sig,
            motors=motors,
            cam_name=cam_name,
            cam_height=cam_height,
            cam_width=cam_width,
            capture_fps=capture_fps,
            action_timeout_s=action_timeout_s,
            inventory=inventory if inventory is not None else SyntheticInventory(),
            ice_servers=ice_servers,
            camera=camera,
            cameras=cameras,
            cameras_src=cameras_src,
            robot=robot,
            reliable_state=reliable_state,
            reliable_action=reliable_action,
            transport_backend=transport_backend,
            livekit_url=livekit_url,
            livekit_token=livekit_token,
        )
        if on_agent is not None:
            on_agent(agent)  # let a harness observe the live agent (watchdog/plan)
        try:
            await agent.run()  # blocks here until a controller answers the offer
            logger.info("daemon: session %s established", session_id)
            await agent.wait_closed()
            logger.info("daemon: session %s closed", session_id)
        except SignalingClosedError:
            logger.info("daemon: signaling closed before a session formed")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("daemon: session error")
        finally:
            agent.force_safe_stop()  # P0: never leave the arm live across sessions
            await agent.close()
        if not stop.is_set():
            await asyncio.sleep(0.2)  # brief backoff before waiting for the next session


def main() -> None:
    parser = argparse.ArgumentParser(description="WebRTCProxyRobot robot-side daemon")
    parser.add_argument(
        "--signaling-url", default=None, help="ws://host:port/ws (required for --transport aiortc)"
    )
    parser.add_argument("--session", default="default")
    parser.add_argument("--camera-name", default="front")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--action-timeout", type=float, default=0.5)
    parser.add_argument(
        "--ice-server", action="append", default=[], help="STUN/TURN url (repeatable); omit for same-host"
    )
    parser.add_argument(
        "--real-devices",
        action="store_true",
        help="enumerate the robot's actual serial ports + cameras (find_port/list_cameras return real ids)",
    )
    parser.add_argument(
        "--real-camera",
        default=None,
        help="open this opencv camera (index e.g. 0, or /dev/videoN) and stream it instead of synthetic frames",
    )
    parser.add_argument(
        "--profile",
        choices=["teleop", "eval", "record"],
        default="teleop",
        help="channel reliability: teleop/eval => unreliable (fresh); record => reliable state+action",
    )
    parser.add_argument("--reliable-state", action="store_true", help="override: reliable state channel")
    parser.add_argument("--reliable-action", action="store_true", help="override: reliable action channel")
    parser.add_argument("--auth-token", default=None, help="shared token for the signaling relay")
    parser.add_argument(
        "--transport", choices=["aiortc", "livekit"], default="aiortc", help="transport backend"
    )
    parser.add_argument(
        "--livekit-url",
        default=os.environ.get("LIVEKIT_URL"),
        help="LiveKit server URL (when --transport livekit; default $LIVEKIT_URL)",
    )
    parser.add_argument(
        "--livekit-token",
        default=None,
        help="pre-signed LiveKit JWT; omit to self-sign from --livekit-api-key/secret",
    )
    parser.add_argument(
        "--livekit-api-key",
        default=os.environ.get("LIVEKIT_API_KEY"),
        help="LiveKit API key for self-signing a token (default $LIVEKIT_API_KEY)",
    )
    parser.add_argument(
        "--livekit-api-secret",
        default=os.environ.get("LIVEKIT_API_SECRET"),
        help="LiveKit API secret for self-signing a token (default $LIVEKIT_API_SECRET)",
    )
    parser.add_argument(
        "--livekit-identity", default="robot", help="this daemon's LiveKit participant identity"
    )
    # Any leftover args (e.g. --robot.type=so101_follower --robot.port=/dev/...) are parsed
    # by draccus into a real robot. No --robot.* => synthetic source (or --real-camera).
    args, robot_args = parser.parse_known_args()
    # force=True: the livekit SDK configures the root logger on import, which would make a
    # plain basicConfig a no-op (hiding our INFO logs). Reset so daemon INFO lines show.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True)

    # Per-backend required args (livekit does its own signaling; aiortc needs the relay).
    livekit_token = args.livekit_token
    if args.transport == "aiortc":
        if not args.signaling_url:
            parser.error("--signaling-url is required for --transport aiortc")
    else:  # livekit: need a URL and a token (pre-signed, or self-signed from api key/secret)
        if not args.livekit_url:
            parser.error("--livekit-url (or $LIVEKIT_URL) is required for --transport livekit")
        if not livekit_token:
            if not args.livekit_api_key or not args.livekit_api_secret:
                parser.error(
                    "--transport livekit needs --livekit-token, or --livekit-api-key + "
                    "--livekit-api-secret (or $LIVEKIT_API_KEY/$LIVEKIT_API_SECRET) to self-sign"
                )
            from .transport_livekit import make_livekit_token

            # The LiveKit room is the session id; both ends must use the same one.
            livekit_token = make_livekit_token(
                api_key=args.livekit_api_key,
                api_secret=args.livekit_api_secret,
                identity=args.livekit_identity,
                room=args.session,
            )
            logger.info(
                "self-signed LiveKit token (identity=%s, room=%s)", args.livekit_identity, args.session
            )

    # record needs complete, ordered obs AND actions (no lost transitions); realtime loops
    # (teleop/eval) want freshness. Explicit flags override the profile.
    reliable_state = args.reliable_state or args.profile == "record"
    reliable_action = args.reliable_action or args.profile == "record"

    inventory: DeviceInventory = LocalDeviceInventory() if args.real_devices else SyntheticInventory()
    logger.info("daemon device inventory: %s", type(inventory).__name__)

    # A real robot (--robot.*) provides joints + its own cameras; otherwise fall back to a
    # standalone --real-camera, or a synthetic source. Robot and --real-camera are exclusive.
    robot = motors = cameras = camera = None
    if robot_args:
        robot, motors, cameras = _build_robot(robot_args)
    elif args.real_camera is not None:
        index_or_path = int(args.real_camera) if args.real_camera.isdigit() else args.real_camera
        camera = _open_streaming_camera(index_or_path, args.fps, args.width, args.height)

    try:
        asyncio.run(
            run_daemon(
                signaling_url=args.signaling_url,
                session_id=args.session,
                motors=motors,
                cam_name=args.camera_name,
                cam_height=args.height,
                cam_width=args.width,
                cameras=cameras,
                capture_fps=args.fps,
                action_timeout_s=args.action_timeout,
                ice_servers=args.ice_server,
                inventory=inventory,
                camera=camera,
                robot=robot,
                reliable_state=reliable_state,
                reliable_action=reliable_action,
                signaling_token=args.auth_token,
                transport_backend=args.transport,
                livekit_url=args.livekit_url,
                livekit_token=livekit_token,
            )
        )
    except KeyboardInterrupt:
        pass
    finally:
        if camera is not None:
            camera.disconnect()
        if robot is not None:
            with contextlib.suppress(Exception):
                robot.disconnect()


if __name__ == "__main__":
    main()
