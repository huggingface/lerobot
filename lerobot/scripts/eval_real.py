import argparse
import logging
import time
from contextlib import nullcontext

import torch
from torch import nn

from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.policy_protocol import Policy
from lerobot.common.policies.rollout_wrapper import PolicyRolloutWrapper
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.koch import KochRobot
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, init_logging, set_global_seed
from lerobot.scripts.eval import get_pretrained_policy_path


def busy_wait(seconds: float):
    # Significantly more accurate than `time.sleep`, and mandatory for our use case,
    # but it consumes CPU cycles.
    # TODO(rcadene): find an alternative: from python 11, time.sleep is precise
    end_time = time.perf_counter() + seconds
    while time.perf_counter() < end_time:
        time.sleep(0.0001)


def rollout(robot: KochRobot, policy: Policy, fps: float, warmup_s: float = 5.0):
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    device = get_device_from_parameters(policy)
    policy_rollout_wrapper = PolicyRolloutWrapper(policy, fps=fps, n_action_buffer=0)

    policy_rollout_wrapper.reset()

    step = 0
    start_t = time.perf_counter()
    period = 1 / fps
    buffer = 0.001
    while True:
        start_step_t = time.perf_counter() - start_t

        observation: dict[str, torch.Tensor] = robot.capture_observation()

        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if name.startswith("observation.image"):
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # Compute the next action with the policy
        # based on the current observation
        with torch.inference_mode():
            action_sequence = policy_rollout_wrapper.provide_observation_get_actions(
                observation,
                observation_timestamp=start_step_t,
                first_action_timestamp=start_step_t,
                strict_observation_timestamps=step > 0,
                timeout=period - (time.perf_counter() - start_step_t) - buffer if step > 0 else None,
            )
            if action_sequence is not None:
                action_sequence = action_sequence.numpy()
                action_sequence = action_sequence.squeeze(0)

        if step == 0:
            # On the first step we should just use the first action. We are guaranteed that action_sequence is
            # not None.
            action = action_sequence[0]
            # We also need to store the next action. If the next action is not available, we adopt the
            # strategy of repeating the current action.
            if len(action_sequence) > 1:
                next_action = action_sequence[1].copy()
            else:
                next_action = action.copy()
                is_dropped_cycle = True
        else:
            # All steps after  the first must use the `next_action` from the previous step.
            action = next_action.copy()
            if action_sequence is not None and len(action_sequence) > 1:
                next_action = action_sequence[1].copy()
            else:
                next_action = action.copy()
                is_dropped_cycle = True

        # Order the robot to move
        if start_step_t - start_t >= warmup_s:
            robot.send_action(action)

        this_step_t = time.perf_counter() - start_step_t
        if this_step_t > period:
            logging.warning(f"Step took too long! {this_step_t=}")

        busy_wait(period - this_step_t)
        step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fps", type=float)
    parser.add_argument(
        "--robot-path",
        type=str,
        default="lerobot/configs/robot/koch_.yaml",
        help="Path to robot yaml file used to instantiate the robot using `make_robot` factory function.",
    )
    parser.add_argument(
        "--warmup-time-s",
        type=int,
        default=5,
        help="Number of seconds before starting data collection. It allows the robot devices to warmup and synchronize.",
    )
    parser.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        type=str,
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`."
        ),
    )
    parser.add_argument(
        "--policy-overrides",
        type=str,
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )

    args = parser.parse_args()

    init_logging()

    pretrained_policy_path = get_pretrained_policy_path(args.pretrained_policy_name_or_path)

    robot_cfg = init_hydra_config(args.robot_path)
    robot = make_robot(robot_cfg)

    try:
        if not robot.is_connected:
            robot.connect()
        hydra_cfg = init_hydra_config(str(pretrained_policy_path / "config.yaml"), args.policy_overrides)

        # Check device is available
        device = get_safe_torch_device(hydra_cfg.device, log=True)

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        set_global_seed(hydra_cfg.seed)

        policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=str(pretrained_policy_path))

        assert isinstance(policy, nn.Module)
        policy.eval()

        with torch.no_grad(), torch.autocast(device_type=device.type) if hydra_cfg.use_amp else nullcontext():
            rollout(robot, policy, args.fps, warmup_s=args.warmup_time_s)

        logging.info("End of eval")
    finally:
        if robot.is_connected:
            # Disconnect manually to avoid a "Core dump" during process
            # termination due to camera threads not properly exiting.
            robot.disconnect()
