import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from threading import Event, Lock, Thread

import draccus
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc import ActionInterpolator, ActionQueue, LatencyTracker, RTCConfig
from lerobot.processor.factory import (
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    koch_follower,
    so_follower,
    unitree_g1,
)
from lerobot.robots.utils import make_robot_from_config


@dataclass
class OrchestratorRTCXVLAConfig:
    # Kept for CLI compatibility with legacy orchestrator; not used in local RTC mode.
    server_address: str = "192.168.1.107:4567"

    # Legacy-style policy args
    policy_type: str = "xvla"
    pretrained_name_or_path: str = ""
    policy_device: str = "cuda"

    # Robot / task args
    robot: RobotConfig | None = None
    task: str = field(default="", metadata={"help": "Task instruction"})

    # Runtime args
    duration: float = 120.0
    fps: float = 30.0
    interpolation_multiplier: int = 1

    # RTC args
    rtc_enabled: bool = True
    rtc_execution_horizon: int = 10
    rtc_max_guidance_weight: float = 10.0
    rtc_prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.EXP
    rtc_debug: bool = False
    rtc_debug_maxlen: int = 100

    # How many actions can remain in queue before requesting a new chunk.
    # Keep this larger than inference_delay + execution_horizon.
    action_queue_size_to_get_new_actions: int = 30

    # If None, script estimates inference_delay from measured latency.
    # If set, this fixed value is passed to predict_action_chunk.
    inference_delay_steps: int | None = None

    # Optional override. If None, domain_id from pretrained config.json is used.
    xvla_domain_id: int | None = None

    # Safe exit homing
    safe_homing_on_exit: bool = True
    safe_homing_duration_s: float = 8.0
    safe_homing_gripper_open_value: float = 60.0
    safe_homing_gripper_home_value: float = 0.0
    safe_homing_gripper_open_wait_s: float = 1.5
    rename_map: dict[str, str] = field(
        default_factory=lambda: {
            "observation.images.camera1": "observation.images.image",
            "observation.images.camera2": "observation.images.image2",
        }
    )

    def __post_init__(self):
        if self.robot is None:
            raise ValueError("robot configuration is required")
        if self.policy_type.lower() != "xvla":
            raise ValueError(
                "This script is dedicated to RTC + X-VLA. Please set --policy_type=xvla"
            )
        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path is required")
        if self.fps <= 0:
            raise ValueError("fps must be > 0")
        if self.action_queue_size_to_get_new_actions < 0:
            raise ValueError("action_queue_size_to_get_new_actions must be >= 0")
        if self.inference_delay_steps is not None and self.inference_delay_steps < 0:
            raise ValueError("inference_delay_steps must be >= 0 when provided")
        if self.rtc_debug_maxlen <= 0:
            raise ValueError("rtc_debug_maxlen must be > 0")


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
    current_obs: dict,
    joint_names: list[str],
) -> dict[str, float]:
    missing = [k for k in joint_names if k not in current_obs]
    if missing:
        raise KeyError(f"Missing joint keys in observation: {missing}")
    return {k: _to_float(current_obs[k]) for k in joint_names}


def _open_gripper_if_present(
    action_dict: dict[str, float],
    joint_names: list[str],
    gripper_open_value: float,
) -> bool:
    gripper_name = next((k for k in joint_names if "gripper" in k.lower() or "jaw" in k.lower()), None)
    if gripper_name:
        print(f"[SAFE EXIT] Opening gripper before homing (target: {gripper_open_value})")
        action_dict[gripper_name] = gripper_open_value
        return True
    return False


def _home_to_zero(
    robot,
    homing_duration_s: float,
    start_action: dict[str, float],
    joint_names: list[str],
    gripper_home_value: float,
) -> None:
    print(f"[SAFE EXIT] Homing to zero pose over {homing_duration_s:.1f}s")

    target_action = {k: 0.0 for k in joint_names}
    gripper_name = next((k for k in joint_names if "gripper" in k.lower() or "jaw" in k.lower()), None)
    if gripper_name:
        target_action[gripper_name] = gripper_home_value

    hz = 50.0
    steps = max(1, int(homing_duration_s * hz))
    sleep_time = 1.0 / hz

    for i in range(1, steps + 1):
        alpha = i / steps
        smooth_alpha = (1.0 - math.cos(alpha * math.pi)) / 2.0
        interp_action = {
            k: start_action[k] + smooth_alpha * (target_action[k] - start_action[k])
            for k in joint_names
        }
        robot.send_action(interp_action)
        time.sleep(sleep_time)


class RobotWrapper:
    def __init__(self, robot: Robot):
        self.robot = robot
        self.lock = Lock()

    def get_observation(self):
        with self.lock:
            return self.robot.get_observation()

    def send_action(self, action):
        with self.lock:
            self.robot.send_action(action)

    def observation_features(self):
        with self.lock:
            return self.robot.observation_features

    def action_features(self):
        with self.lock:
            return self.robot.action_features


def _load_xvla_policy(cfg: OrchestratorRTCXVLAConfig):
    policy_class = get_policy_class("xvla")
    policy_cfg = PreTrainedConfig.from_pretrained(cfg.pretrained_name_or_path)

    policy_cfg.device = cfg.policy_device
    if cfg.xvla_domain_id is not None:
        policy_cfg.domain_id = cfg.xvla_domain_id
    policy_cfg.rtc_config = RTCConfig(
        enabled=cfg.rtc_enabled,
        execution_horizon=cfg.rtc_execution_horizon,
        max_guidance_weight=cfg.rtc_max_guidance_weight,
        prefix_attention_schedule=cfg.rtc_prefix_attention_schedule,
        debug=cfg.rtc_debug,
        debug_maxlen=cfg.rtc_debug_maxlen,
    )

    policy = policy_class.from_pretrained(cfg.pretrained_name_or_path, config=policy_cfg)
    if hasattr(policy, "init_rtc_processor"):
        policy.init_rtc_processor()

    policy = policy.to(cfg.policy_device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=cfg.pretrained_name_or_path,
        dataset_stats=None,
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy_device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
        postprocessor_overrides={"device_processor": {"device": cfg.policy_device}},
    )

    return policy, preprocessor, postprocessor


def _get_actions_thread(
    cfg: OrchestratorRTCXVLAConfig,
    robot: RobotWrapper,
    robot_observation_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
):
    policy, preprocessor, postprocessor = _load_xvla_policy(cfg)
    latency_tracker = LatencyTracker()
    model_dtype_name = getattr(policy.config, "dtype", "float32")
    obs_dtype = torch.bfloat16 if model_dtype_name == "bfloat16" else torch.float32
    amp_dtype = torch.bfloat16 if model_dtype_name == "bfloat16" else None
    device_type = cfg.policy_device.split(":")[0]

    observation_features_hw = {
        key: value
        for key, value in robot.observation_features().items()
        if key.endswith(".pos") or isinstance(value, tuple)
    }
    dataset_features = hw_to_dataset_features(observation_features_hw, "observation")

    while not shutdown_event.is_set():
        if action_queue.qsize() > cfg.action_queue_size_to_get_new_actions:
            time.sleep(0.01)
            continue

        start_time = time.perf_counter()
        action_index_before = action_queue.get_action_index()
        prev_actions = action_queue.get_left_over() if cfg.rtc_enabled else None

        raw_obs = robot.get_observation()
        obs_processed = robot_observation_processor(raw_obs)
        obs = build_dataset_frame(dataset_features, obs_processed, prefix="observation")

        for name in obs:
            obs[name] = torch.from_numpy(obs[name])
            if "image" in name:
                obs[name] = obs[name].type(torch.float32) / 255.0
                obs[name] = obs[name].permute(2, 0, 1).contiguous()
            elif obs[name].is_floating_point():
                obs[name] = obs[name].to(obs_dtype)

            if obs[name].is_floating_point() and "image" in name:
                obs[name] = obs[name].to(obs_dtype)

            obs[name] = obs[name].unsqueeze(0).to(cfg.policy_device)

        obs["task"] = [cfg.task]
        obs["robot_type"] = robot.robot.name if hasattr(robot.robot, "name") else ""

        policy_input = preprocessor(obs)

        time_per_step = 1.0 / cfg.fps
        if cfg.inference_delay_steps is None:
            current_delay = math.ceil(latency_tracker.max() / time_per_step)
        else:
            current_delay = cfg.inference_delay_steps

        if cfg.action_queue_size_to_get_new_actions < cfg.rtc_execution_horizon + current_delay:
            print(
                "[WARN] action_queue_size_to_get_new_actions is small. "
                "Recommended > inference_delay_steps + rtc_execution_horizon"
            )

        use_autocast = amp_dtype is not None and device_type in {"cuda", "cpu"}
        autocast_ctx = (
            torch.autocast(device_type=device_type, dtype=amp_dtype)
            if use_autocast
            else nullcontext()
        )

        with torch.inference_mode():
            with autocast_ctx:
                actions = policy.predict_action_chunk(
                    policy_input,
                    inference_delay=current_delay,
                    prev_chunk_left_over=prev_actions,
                )

        original_actions = actions.squeeze(0).clone()
        processed_actions = postprocessor(actions).squeeze(0)

        new_latency = time.perf_counter() - start_time
        real_delay = math.ceil(new_latency / time_per_step)
        latency_tracker.add(new_latency)

        action_queue.merge(original_actions, processed_actions, real_delay, action_index_before)


def _actor_thread(
    cfg: OrchestratorRTCXVLAConfig,
    robot: RobotWrapper,
    robot_action_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
):
    action_keys = [k for k in robot.action_features() if k.endswith(".pos")]
    interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)
    action_interval = interpolator.get_control_interval(cfg.fps)

    while not shutdown_event.is_set():
        start = time.perf_counter()

        if interpolator.needs_new_action():
            new_action = action_queue.get()
            if new_action is not None:
                interpolator.add(new_action.cpu())

        action = interpolator.get()
        if action is not None:
            action = action.cpu()
            action_dict = {k: action[i].item() for i, k in enumerate(action_keys)}
            action_processed = robot_action_processor((action_dict, None))
            robot.send_action(action_processed)

        dt = time.perf_counter() - start
        time.sleep(max(0.0, action_interval - dt))


def run_orchestrator_rtc_xvla(cfg: OrchestratorRTCXVLAConfig):
    print("[INFO] Running local RTC + X-VLA orchestrator (server_address is ignored).")
    print(
        f"[INFO] policy={cfg.pretrained_name_or_path} | device={cfg.policy_device} | "
        f"rtc_enabled={cfg.rtc_enabled} | schedule={cfg.rtc_prefix_attention_schedule.name} | "
        f"inference_delay_steps={'auto' if cfg.inference_delay_steps is None else cfg.inference_delay_steps}"
    )

    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    robot_wrapper = RobotWrapper(robot)

    robot_observation_processor = make_default_robot_observation_processor()
    robot_action_processor = make_default_robot_action_processor()
    action_queue = ActionQueue(
        RTCConfig(
            enabled=cfg.rtc_enabled,
            execution_horizon=cfg.rtc_execution_horizon,
            max_guidance_weight=cfg.rtc_max_guidance_weight,
            prefix_attention_schedule=cfg.rtc_prefix_attention_schedule,
        )
    )
    shutdown_event = Event()

    get_actions_worker = Thread(
        target=_get_actions_thread,
        args=(cfg, robot_wrapper, robot_observation_processor, action_queue, shutdown_event),
        daemon=True,
        name="GetActionsRTCXvla",
    )
    actor_worker = Thread(
        target=_actor_thread,
        args=(cfg, robot_wrapper, robot_action_processor, action_queue, shutdown_event),
        daemon=True,
        name="ActorRTCXvla",
    )

    get_actions_worker.start()
    actor_worker.start()

    start = time.time()
    try:
        while not shutdown_event.is_set() and (time.time() - start) < cfg.duration:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received, shutting down...")
    finally:
        shutdown_event.set()
        get_actions_worker.join(timeout=2.0)
        actor_worker.join(timeout=2.0)

        if cfg.safe_homing_on_exit:
            try:
                current_obs = robot.get_observation()
                joint_names = list(robot.action_features.keys())

                action_dict = _extract_current_action_from_observation(current_obs, joint_names)
                gripper_opened = _open_gripper_if_present(
                    action_dict,
                    joint_names,
                    cfg.safe_homing_gripper_open_value,
                )
                if gripper_opened:
                    robot.send_action(action_dict)
                    time.sleep(cfg.safe_homing_gripper_open_wait_s)

                _home_to_zero(
                    robot=robot,
                    homing_duration_s=cfg.safe_homing_duration_s,
                    start_action=action_dict,
                    joint_names=joint_names,
                    gripper_home_value=cfg.safe_homing_gripper_home_value,
                )
                print("[SAFE EXIT] Homing complete.")
            except Exception as exc:
                print(f"[WARN] Safe homing failed: {exc}")

        robot.disconnect()
        print("[INFO] Robot disconnected safely.")


@draccus.wrap()
def main(cfg: OrchestratorRTCXVLAConfig):
    run_orchestrator_rtc_xvla(cfg)


if __name__ == "__main__":
    main()
