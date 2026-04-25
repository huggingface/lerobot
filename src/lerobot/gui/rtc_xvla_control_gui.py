import ast
import contextlib
import json
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from queue import Empty, Queue
from tkinter import messagebox, ttk

import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.robots.so_follower import SO100FollowerConfig, SO101FollowerConfig
from lerobot.rtc_inference.configs import AGGREGATE_FUNCTIONS, RobotClientConfig
from lerobot.rtc_inference.robot_client import RobotClient
from lerobot.utils.import_utils import register_third_party_plugins


@dataclass
class _RuntimeState:
    connected: bool = False
    zero_pose_done: bool = False
    stream_running: bool = False
    busy: bool = False
    action_receiver_thread: threading.Thread | None = None
    control_loop_thread: threading.Thread | None = None


@dataclass
class _FieldSpec:
    key: str
    label: str
    default: str
    field_type: str
    options: tuple[str, ...] = ()


class RTCXVLAControlGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RTC XVLA Control Panel")
        self.geometry("1300x820")

        self._state = _RuntimeState()
        self._state_lock = threading.Lock()

        self._client: RobotClient | None = None
        self._client_cfg: RobotClientConfig | None = None

        self._vars: dict[str, tk.StringVar] = {}
        self._widgets: dict[str, ttk.Widget] = {}

        self._log_queue: Queue[str] = Queue()

        self._build_ui()
        self._start_log_pump()
        self._refresh_controls()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(root)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(root)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        self._build_config_panel(left)
        self._build_control_panel(right)

    def _build_config_panel(self, parent: ttk.Frame):
        title = ttk.Label(parent, text="Configuration", font=("Segoe UI", 12, "bold"))
        title.pack(anchor="w")

        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        form_frame = ttk.Frame(canvas)

        form_frame.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=form_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        row = 0
        for section_name, specs in self._field_sections().items():
            section_label = ttk.Label(form_frame, text=section_name, font=("Segoe UI", 10, "bold"))
            section_label.grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 4))
            row += 1

            for spec in specs:
                ttk.Label(form_frame, text=spec.label).grid(
                    row=row, column=0, sticky="w", padx=(0, 8), pady=3
                )
                var = tk.StringVar(value=spec.default)
                self._vars[spec.key] = var

                if spec.field_type in {"bool", "choice"}:
                    values = spec.options if spec.options else ("true", "false")
                    widget = ttk.Combobox(
                        form_frame, textvariable=var, values=values, state="readonly", width=48
                    )
                else:
                    widget = ttk.Entry(form_frame, textvariable=var, width=80)

                widget.grid(row=row, column=1, sticky="we", pady=3)
                self._widgets[spec.key] = widget
                row += 1

        form_frame.columnconfigure(1, weight=1)

    def _build_control_panel(self, parent: ttk.Frame):
        title = ttk.Label(parent, text="Control", font=("Segoe UI", 12, "bold"))
        title.pack(anchor="w", pady=(0, 8))

        self.status_var = tk.StringVar(value="Disconnected")
        status = ttk.Label(parent, textvariable=self.status_var, foreground="#1f4e79")
        status.pack(anchor="w", pady=(0, 10))

        self.btn_connect = ttk.Button(
            parent, text="Connect", command=lambda: self._run_async(self._on_connect)
        )
        self.btn_connect.pack(fill=tk.X, pady=4)

        self.btn_zero = ttk.Button(
            parent, text="To Zero Pose", command=lambda: self._run_async(self._on_zero_pose)
        )
        self.btn_zero.pack(fill=tk.X, pady=4)

        self.btn_call = ttk.Button(
            parent, text="Call to Server", command=lambda: self._run_async(self._on_call_server)
        )
        self.btn_call.pack(fill=tk.X, pady=4)

        self.btn_stop_stream = ttk.Button(
            parent,
            text="Stop Stream",
            command=lambda: self._run_async(self._on_stop_stream),
        )
        self.btn_stop_stream.pack(fill=tk.X, pady=4)

        self.btn_disconnect = ttk.Button(
            parent,
            text="Disconnect",
            command=lambda: self._run_async(self._on_disconnect),
        )
        self.btn_disconnect.pack(fill=tk.X, pady=4)

        log_label = ttk.Label(parent, text="Logs", font=("Segoe UI", 10, "bold"))
        log_label.pack(anchor="w", pady=(14, 4))

        self.log_text = tk.Text(parent, width=55, height=36, state="disabled", wrap="word")
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _field_sections(self) -> dict[str, list[_FieldSpec]]:
        return {
            "Connection": [
                _FieldSpec("server_address", "Server Address", "192.168.1.107:4567", "str"),
                _FieldSpec(
                    "task",
                    "Task",
                    "Pour sunflower seeds from the orange cup into the clean cup on the white cloth.",
                    "str",
                ),
            ],
            "Policy": [
                _FieldSpec("policy_type", "Policy Type", "xvla", "choice", ("xvla",)),
                _FieldSpec("pretrained_name_or_path", "Pretrained", "trietlm0306/xvla-poursing-v1", "str"),
                _FieldSpec("policy_device", "Policy Device", "cuda", "str"),
                _FieldSpec("client_device", "Client Device", "cpu", "str"),
                _FieldSpec(
                    "rename_map",
                    "Rename Map (JSON)",
                    '{"observation.images.camera1":"observation.images.image","observation.images.camera2":"observation.images.image2"}',
                    "json_dict",
                ),
            ],
            "Robot": [
                _FieldSpec(
                    "robot_type",
                    "Robot Type",
                    "so101_follower",
                    "choice",
                    ("so101_follower", "so100_follower"),
                ),
                _FieldSpec("robot_port", "Robot Port", "COM6", "str"),
                _FieldSpec("robot_id", "Robot ID", "DI_VLA_FOLLOWER", "str"),
                _FieldSpec("robot_use_degrees", "Robot Use Degrees", "true", "bool", ("true", "false")),
                _FieldSpec(
                    "robot_disable_torque_on_disconnect",
                    "Disable Torque On Disconnect",
                    "true",
                    "bool",
                    ("true", "false"),
                ),
                _FieldSpec("robot_max_relative_target", "Max Relative Target (empty|float|dict)", "", "str"),
                _FieldSpec(
                    "robot_cameras",
                    "Robot Cameras (JSON)",
                    '{"camera1":{"type":"opencv","index_or_path":1,"width":1280,"height":720,"fps":30},"camera2":{"type":"opencv","index_or_path":0,"width":640,"height":360,"fps":30}}',
                    "json_dict",
                ),
            ],
            "Runtime": [
                _FieldSpec("actions_per_chunk", "Actions Per Chunk", "30", "int"),
                _FieldSpec("chunk_size_threshold", "Chunk Size Threshold", "0.7", "float"),
                _FieldSpec(
                    "aggregate_fn_name",
                    "Aggregate Function",
                    "latest_only",
                    "choice",
                    tuple(AGGREGATE_FUNCTIONS.keys()),
                ),
                _FieldSpec("fps", "FPS", "15", "int"),
                _FieldSpec(
                    "obs_timestep_independent", "Obs Timestep Independent", "false", "bool", ("true", "false")
                ),
                _FieldSpec(
                    "image_compress_enable", "Image Compress Enable", "true", "bool", ("true", "false")
                ),
                _FieldSpec("image_compress_quality", "Image Compress Quality", "90", "int"),
                _FieldSpec("interpolation_multiplier", "Interpolation Multiplier", "1", "int"),
                _FieldSpec(
                    "debug_visualize_queue_size", "Debug Visualize Queue", "true", "bool", ("true", "false")
                ),
            ],
            "RTC": [
                _FieldSpec("rtc_execution_horizon", "RTC Execution Horizon", "10", "int"),
                _FieldSpec("rtc_max_guidance_weight", "RTC Max Guidance Weight", "10.0", "float"),
                _FieldSpec(
                    "rtc_prefix_attention_schedule",
                    "RTC Prefix Schedule",
                    "EXP",
                    "choice",
                    tuple(s.name for s in RTCAttentionSchedule),
                ),
                _FieldSpec("rtc_debug", "RTC Debug", "false", "bool", ("true", "false")),
                _FieldSpec("rtc_debug_maxlen", "RTC Debug Maxlen", "100", "int"),
                _FieldSpec("inference_delay_steps", "Inference Delay Steps (optional)", "2", "optional_int"),
                _FieldSpec("xvla_domain_id", "XVLA Domain ID (optional)", "15", "optional_int"),
            ],
            "Homing": [
                _FieldSpec("homing_duration_start", "Homing Duration Start", "8.0", "float"),
                _FieldSpec("homing_duration_after_stop", "Homing Duration After Stop", "8.0", "float"),
                _FieldSpec("gripper_home_value", "Gripper Home Value", "0.0", "float"),
                _FieldSpec("zero_pose_left_offset_deg", "Zero Pose Left Offset (deg)", "20.0", "float"),
            ],
        }

    def _start_log_pump(self):
        def pump():
            try:
                while True:
                    msg = self._log_queue.get_nowait()
                    self.log_text.configure(state="normal")
                    self.log_text.insert("end", msg + "\n")
                    self.log_text.see("end")
                    self.log_text.configure(state="disabled")
            except Empty:
                pass
            self.after(120, pump)

        self.after(120, pump)

    def _log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self._log_queue.put(f"[{timestamp}] {message}")

    def _set_busy(self, busy: bool):
        with self._state_lock:
            self._state.busy = busy
        self.after(0, self._refresh_controls)

    def _set_status(self, text: str):
        self.after(0, lambda: self.status_var.set(text))

    def _refresh_controls(self):
        with self._state_lock:
            connected = self._state.connected
            zero_pose_done = self._state.zero_pose_done
            stream_running = self._state.stream_running
            busy = self._state.busy

        self.btn_connect.configure(state=("normal" if not connected and not busy else "disabled"))
        self.btn_zero.configure(
            state=("normal" if connected and not stream_running and not busy else "disabled")
        )
        self.btn_call.configure(
            state=(
                "normal" if connected and zero_pose_done and not stream_running and not busy else "disabled"
            )
        )
        self.btn_stop_stream.configure(
            state=("normal" if connected and stream_running and not busy else "disabled")
        )
        self.btn_disconnect.configure(state=("normal" if connected and not busy else "disabled"))

    def _run_async(self, fn):
        def worker():
            self._set_busy(True)
            try:
                fn()
            except Exception as exc:
                error_text = str(exc)
                self._log(f"[ERROR] {exc}")
                self.after(0, lambda: messagebox.showerror("Error", error_text))
            finally:
                self._set_busy(False)

        threading.Thread(target=worker, daemon=True).start()

    def _parse_bool(self, key: str) -> bool:
        value = self._vars[key].get().strip().lower()
        if value in {"true", "1", "yes", "y", "on"}:
            return True
        if value in {"false", "0", "no", "n", "off"}:
            return False
        raise ValueError(f"{key}: invalid bool '{value}'")

    def _parse_int(self, key: str) -> int:
        return int(self._vars[key].get().strip())

    def _parse_float(self, key: str) -> float:
        return float(self._vars[key].get().strip())

    def _parse_optional_int(self, key: str) -> int | None:
        text = self._vars[key].get().strip()
        if text == "":
            return None
        return int(text)

    def _parse_dict_like(self, key: str) -> dict:
        text = self._vars[key].get().strip()
        if text == "":
            return {}

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except Exception as exc:
                raise ValueError(f"{key}: invalid dict/json format") from exc

        if not isinstance(parsed, dict):
            raise ValueError(f"{key}: expected a dict-like object")
        return parsed

    def _parse_optional_float_or_dict(self, key: str):
        text = self._vars[key].get().strip()
        if text == "":
            return None
        try:
            return float(text)
        except ValueError:
            parsed = self._parse_dict_like(key)
            for name, value in parsed.items():
                parsed[name] = float(value)
            return parsed

    @staticmethod
    def _coerce_bool_value(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y", "on"}:
            return True
        if text in {"false", "0", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid boolean value: {value}")

    def _build_camera_configs(self) -> dict:
        raw = self._parse_dict_like("robot_cameras")
        cameras = {}

        for name, cfg in raw.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"robot_cameras.{name}: expected object")

            camera_type = str(cfg.get("type", "opencv")).lower()
            if camera_type == "opencv":
                if "index_or_path" not in cfg:
                    raise ValueError(f"robot_cameras.{name}: missing index_or_path")

                index_or_path = cfg["index_or_path"]
                if isinstance(index_or_path, str) and index_or_path.strip().isdigit():
                    index_or_path = int(index_or_path.strip())

                cameras[name] = OpenCVCameraConfig(
                    index_or_path=index_or_path,
                    fps=int(cfg["fps"]),
                    width=int(cfg["width"]),
                    height=int(cfg["height"]),
                    color_mode=cfg.get("color_mode", "rgb"),
                    rotation=int(cfg.get("rotation", 0)),
                    warmup_s=int(cfg.get("warmup_s", 1)),
                    fourcc=cfg.get("fourcc"),
                    backend=int(cfg.get("backend", 0)),
                )
                continue

            if camera_type in {"intelrealsense", "realsense"}:
                serial = cfg.get("serial_number_or_name")
                if not serial:
                    raise ValueError(f"robot_cameras.{name}: missing serial_number_or_name for realsense")

                cameras[name] = RealSenseCameraConfig(
                    serial_number_or_name=str(serial),
                    fps=int(cfg["fps"]),
                    width=int(cfg["width"]),
                    height=int(cfg["height"]),
                    color_mode=cfg.get("color_mode", "rgb"),
                    use_depth=self._coerce_bool_value(cfg.get("use_depth", False)),
                    rotation=int(cfg.get("rotation", 0)),
                    warmup_s=int(cfg.get("warmup_s", 1)),
                )
                continue

            raise ValueError(f"robot_cameras.{name}: unsupported camera type '{camera_type}'")

        return cameras

    def _build_robot_config(self):
        robot_type = self._vars["robot_type"].get().strip().lower()
        common_kwargs = {
            "port": self._vars["robot_port"].get().strip(),
            "id": self._vars["robot_id"].get().strip() or None,
            "cameras": self._build_camera_configs(),
            "use_degrees": self._parse_bool("robot_use_degrees"),
            "disable_torque_on_disconnect": self._parse_bool("robot_disable_torque_on_disconnect"),
            "max_relative_target": self._parse_optional_float_or_dict("robot_max_relative_target"),
        }

        if robot_type == "so101_follower":
            return SO101FollowerConfig(**common_kwargs)
        if robot_type == "so100_follower":
            return SO100FollowerConfig(**common_kwargs)

        raise ValueError(
            f"GUI currently supports robot_type in {{so101_follower, so100_follower}}. Received: {robot_type}"
        )

    def _build_client_config(self) -> RobotClientConfig:
        schedule_name = self._vars["rtc_prefix_attention_schedule"].get().strip().upper()
        schedule = RTCAttentionSchedule[schedule_name]

        cfg = RobotClientConfig(
            policy_type=self._vars["policy_type"].get().strip(),
            pretrained_name_or_path=self._vars["pretrained_name_or_path"].get().strip(),
            robot=self._build_robot_config(),
            actions_per_chunk=self._parse_int("actions_per_chunk"),
            task=self._vars["task"].get().strip(),
            rename_map=self._parse_dict_like("rename_map"),
            server_address=self._vars["server_address"].get().strip(),
            policy_device=self._vars["policy_device"].get().strip(),
            client_device=self._vars["client_device"].get().strip(),
            chunk_size_threshold=self._parse_float("chunk_size_threshold"),
            fps=self._parse_int("fps"),
            obs_timestep_independent=self._parse_bool("obs_timestep_independent"),
            image_compress_enable=self._parse_bool("image_compress_enable"),
            image_compress_quality=self._parse_int("image_compress_quality"),
            interpolation_multiplier=self._parse_int("interpolation_multiplier"),
            aggregate_fn_name=self._vars["aggregate_fn_name"].get().strip(),
            debug_visualize_queue_size=self._parse_bool("debug_visualize_queue_size"),
            rtc_enabled=True,
            rtc_execution_horizon=self._parse_int("rtc_execution_horizon"),
            rtc_max_guidance_weight=self._parse_float("rtc_max_guidance_weight"),
            rtc_prefix_attention_schedule=schedule,
            rtc_debug=self._parse_bool("rtc_debug"),
            rtc_debug_maxlen=self._parse_int("rtc_debug_maxlen"),
            inference_delay_steps=self._parse_optional_int("inference_delay_steps"),
            xvla_domain_id=self._parse_optional_int("xvla_domain_id"),
        )
        return cfg

    @staticmethod
    def _to_float(value) -> float:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)

    def _extract_current_action_from_observation(
        self,
        current_obs: dict,
        joint_names: list[str],
    ) -> dict[str, float]:
        missing = [k for k in joint_names if k not in current_obs]
        if not missing:
            return {k: self._to_float(current_obs[k]) for k in joint_names}

        state = current_obs.get("observation.state")
        if state is not None:
            if hasattr(state, "detach"):
                state = state.detach()
            if hasattr(state, "cpu"):
                state = state.cpu()
            if hasattr(state, "numpy"):
                state = state.numpy()

            state_array = np.asarray(state, dtype=np.float64).reshape(-1)
            if state_array.size == len(joint_names):
                return {k: float(state_array[i]) for i, k in enumerate(joint_names)}

        raise KeyError(f"Missing joint keys in observation and invalid observation.state fallback: {missing}")

    def _home_to_zero(
        self,
        robot,
        homing_duration: float,
        start_action: dict[str, float],
        joint_names: list[str],
        gripper_home_value: float,
        left_turn_offset_deg: float,
        use_degrees: bool,
    ):
        target_action = dict.fromkeys(joint_names, 0.0)
        gripper_name = next((k for k in joint_names if "gripper" in k.lower() or "jaw" in k.lower()), None)
        if gripper_name:
            target_action[gripper_name] = gripper_home_value

        left_turn_joint = self._find_left_turn_joint(joint_names)
        if left_turn_joint is not None and abs(left_turn_offset_deg) > 1e-9:
            offset_value = left_turn_offset_deg if use_degrees else left_turn_offset_deg * np.pi / 180.0
            target_action[left_turn_joint] += offset_value
            unit = "deg" if use_degrees else "rad"
            self._log(
                f"[INFO] Zero-pose left offset applied on '{left_turn_joint}': {offset_value:.4f} {unit}"
            )
        elif left_turn_joint is None and abs(left_turn_offset_deg) > 1e-9:
            self._log("[WARN] No base/yaw joint detected. Left offset was skipped.")

        hz = 50.0
        steps = max(1, int(homing_duration * hz))
        sleep_time = 1.0 / hz

        for i in range(1, steps + 1):
            alpha = i / steps
            smooth_alpha = (1.0 - np.cos(alpha * np.pi)) / 2.0
            interp_action = {
                k: start_action[k] + smooth_alpha * (target_action[k] - start_action[k]) for k in joint_names
            }
            robot.send_action(interp_action)
            time.sleep(sleep_time)

        robot.send_action(target_action)

    def _find_left_turn_joint(self, joint_names: list[str]) -> str | None:
        candidates = [
            "shoulder_pan",
            "base",
            "waist",
            "yaw",
            "joint_1",
            "joint1",
        ]

        lower_names = {name.lower(): name for name in joint_names}
        for token in candidates:
            for lower_name, original in lower_names.items():
                if token in lower_name and "gripper" not in lower_name and "jaw" not in lower_name:
                    return original

        for name in joint_names:
            lname = name.lower()
            if "gripper" not in lname and "jaw" not in lname:
                return name
        return None

    def _home_robot_to_zero(
        self, homing_duration: float, gripper_home_value: float, left_turn_offset_deg: float
    ):
        if self._client is None:
            raise RuntimeError("Client is not connected")

        robot = self._client.robot
        current_obs = robot.get_observation()
        joint_names = list(robot.action_features.keys())
        use_degrees = True
        if self._client_cfg is not None and hasattr(self._client_cfg.robot, "use_degrees"):
            use_degrees = bool(self._client_cfg.robot.use_degrees)

        try:
            start_action = self._extract_current_action_from_observation(current_obs, joint_names)
        except KeyError:
            start_action = dict.fromkeys(joint_names, 0.0)

        self._home_to_zero(
            robot,
            homing_duration,
            start_action,
            joint_names,
            gripper_home_value,
            left_turn_offset_deg,
            use_degrees,
        )

    def _stop_client_stream_preserve_robot(self):
        if self._client is None:
            return

        with self._state_lock:
            running = self._state.stream_running
            control_thread = self._state.control_loop_thread
            receiver_thread = self._state.action_receiver_thread

        if not running:
            return

        self._client.shutdown_event.set()
        if control_thread is not None:
            control_thread.join(timeout=2.0)
        if receiver_thread is not None:
            receiver_thread.join(timeout=2.0)

        if receiver_thread is not None and receiver_thread.is_alive():
            with contextlib.suppress(Exception):
                self._client.channel.close()
            receiver_thread.join(timeout=1.0)

        with self._state_lock:
            self._state.stream_running = False

    def _on_connect(self):
        with self._state_lock:
            if self._state.connected:
                self._log("[INFO] Already connected")
                return

        self._log("[INFO] Building config and connecting...")
        cfg = self._build_client_config()
        client = RobotClient(cfg)

        self._client = client
        self._client_cfg = cfg

        with self._state_lock:
            self._state.connected = True
            self._state.zero_pose_done = False
            self._state.stream_running = False
            self._state.action_receiver_thread = None
            self._state.control_loop_thread = None

        self._set_status("Connected")
        self._log("[OK] Connected to robot. Press 'To Zero Pose' before 'Call to Server'.")
        self.after(0, self._refresh_controls)

    def _on_zero_pose(self):
        with self._state_lock:
            connected = self._state.connected
            stream_running = self._state.stream_running
        if not connected:
            raise RuntimeError("Please connect first")
        if stream_running:
            raise RuntimeError("Cannot home while stream is running")

        homing_duration = self._parse_float("homing_duration_start")
        gripper_home_value = self._parse_float("gripper_home_value")
        left_turn_offset_deg = self._parse_float("zero_pose_left_offset_deg")

        self._log(f"[INFO] Homing to zero pose ({homing_duration:.1f}s)...")
        self._home_robot_to_zero(
            homing_duration=homing_duration,
            gripper_home_value=gripper_home_value,
            left_turn_offset_deg=left_turn_offset_deg,
        )

        with self._state_lock:
            self._state.zero_pose_done = True

        self._set_status("Connected | Zero Pose Done")
        self._log("[OK] Robot is at zero pose")
        self.after(0, self._refresh_controls)

    def _on_call_server(self):
        with self._state_lock:
            connected = self._state.connected
            zero_pose_done = self._state.zero_pose_done
            stream_running = self._state.stream_running

        if not connected:
            raise RuntimeError("Please connect first")
        if not zero_pose_done:
            raise RuntimeError("Call to Server requires robot at zero pose. Press 'To Zero Pose' first")
        if stream_running:
            raise RuntimeError("Stream is already running")
        if self._client is None or self._client_cfg is None:
            raise RuntimeError("Client is not initialized")

        self._log("[INFO] Starting client-server stream...")
        if not self._client.start():
            raise RuntimeError("Failed to connect to policy server")

        receiver = threading.Thread(target=self._client.receive_actions, daemon=True)
        control = threading.Thread(
            target=self._client.control_loop,
            kwargs={"task": self._client_cfg.task},
            daemon=True,
        )
        receiver.start()
        control.start()

        with self._state_lock:
            self._state.stream_running = True
            # Require a fresh zero-pose check before next call cycle.
            self._state.zero_pose_done = False
            self._state.action_receiver_thread = receiver
            self._state.control_loop_thread = control

        self._set_status("Connected | Stream Running")
        self._log("[OK] Stream started")
        self.after(0, self._refresh_controls)

    def _on_stop_stream(self):
        with self._state_lock:
            connected = self._state.connected
            stream_running = self._state.stream_running
        if not connected:
            raise RuntimeError("Please connect first")
        if not stream_running:
            raise RuntimeError("Stream is not running")

        self._log("[INFO] Stopping stream (preserve robot connection)...")
        self._stop_client_stream_preserve_robot()

        homing_duration = self._parse_float("homing_duration_after_stop")
        gripper_home_value = self._parse_float("gripper_home_value")
        left_turn_offset_deg = self._parse_float("zero_pose_left_offset_deg")
        self._home_robot_to_zero(
            homing_duration=homing_duration,
            gripper_home_value=gripper_home_value,
            left_turn_offset_deg=left_turn_offset_deg,
        )

        with self._state_lock:
            self._state.zero_pose_done = True

        self._set_status("Connected | Zero Pose Done")
        self._log("[OK] Stream stopped and robot returned to zero pose")
        self.after(0, self._refresh_controls)

    def _on_disconnect(self):
        with self._state_lock:
            connected = self._state.connected
        if not connected:
            self._log("[INFO] Already disconnected")
            return
        if self._client is None:
            return

        self._log("[INFO] Disconnecting...")
        self._stop_client_stream_preserve_robot()

        try:
            self._client.stop()
        except Exception:
            with contextlib.suppress(Exception):
                self._client.channel.close()
            with contextlib.suppress(Exception):
                self._client.robot.disconnect()

        self._client = None
        self._client_cfg = None

        with self._state_lock:
            self._state.connected = False
            self._state.zero_pose_done = False
            self._state.stream_running = False
            self._state.action_receiver_thread = None
            self._state.control_loop_thread = None

        self._set_status("Disconnected")
        self._log("[OK] Disconnected")
        self.after(0, self._refresh_controls)

    def _on_close(self):
        def close_worker():
            with contextlib.suppress(Exception):
                self._on_disconnect()
            self.after(0, self.destroy)

        threading.Thread(target=close_worker, daemon=True).start()


def run_gui() -> None:
    register_third_party_plugins()
    app = RTCXVLAControlGUI()
    app.mainloop()
