#!/usr/bin/env python
"""
Evaluate a trained manipulation policy on the real robot with OAK-D perception.

Loads a trained ACT/Diffusion Policy checkpoint and runs it in a closed loop
with the OAK-D camera and SO-100 arm.

Usage::

    # Run trained policy
    python -m robot.evaluate --checkpoint ./checkpoints/act_final --task "pick up the red cube"

    # Evaluate with recording (saves episodes for analysis)
    python -m robot.evaluate --checkpoint ./checkpoints/act_final --task "pick up the red cube" --record

    # Dry run (no robot, static image)
    python -m robot.evaluate --checkpoint ./checkpoints/act_final --dry-run --dry-run-image sample.png
"""

from __future__ import annotations

import argparse
import logging
import time

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


def load_policy(checkpoint_path: str, device: str = "cpu"):
    """Load a trained LeRobot policy from checkpoint.

    Args:
        checkpoint_path: Path to the saved checkpoint directory.
        device: Torch device (cpu, cuda, mps).

    Returns:
        (policy, preprocessor, postprocessor) tuple.
    """
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig

    config = PreTrainedConfig(path=checkpoint_path)
    policy = make_policy(config)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(config, dataset=None)

    logger.info("Loaded policy from %s", checkpoint_path)
    return policy, preprocessor, postprocessor


def evaluate_policy(
    cfg: dict,
    checkpoint_path: str,
    task: str,
    num_episodes: int = 10,
    max_steps_per_episode: int = 300,
    dry_run: bool = False,
    dry_run_image: str = "",
    record: bool = False,
    device: str = "cpu",
) -> dict:
    """Run a trained policy on the robot and report success metrics.

    Args:
        cfg: Config dict from config.yaml.
        checkpoint_path: Path to trained policy checkpoint.
        task: Task description.
        num_episodes: Number of evaluation episodes.
        max_steps_per_episode: Max steps per episode.
        dry_run: Skip robot, use static image.
        dry_run_image: Path to RGB image for dry-run.
        record: Whether to record episodes.
        device: Torch device.

    Returns:
        Dict with evaluation metrics.
    """
    import torch

    # Load policy
    policy, preprocessor, postprocessor = load_policy(checkpoint_path, device)

    # Init camera
    camera = None
    intrinsics = None
    if not dry_run:
        from lerobot.cameras.oakd import OAKDCamera, OAKDCameraConfig

        cam_cfg = cfg["camera"]
        w, h = cam_cfg["rgb_resolution"]
        camera_config = OAKDCameraConfig(
            device_id=cam_cfg.get("device_id", ""),
            fps=cam_cfg["fps"],
            width=w,
            height=h,
            use_depth=cam_cfg.get("use_depth", True),
            warmup_s=cam_cfg.get("warmup_s", 2),
        )
        camera = OAKDCamera(camera_config)
        camera.connect()
        intrinsics = camera.get_depth_intrinsics()
    else:
        intrinsics = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0,
                      "width": 640, "height": 480, "depth_scale": 0.001}

    # Init arm
    arm = None
    if not dry_run:
        from robot.arm_controller import ArmController, ArmControllerConfig
        from lerobot.robots import make_robot_from_config
        from lerobot.robots.so_follower import SOFollowerRobotConfig

        robot_cfg = cfg["robot"]
        so_config = SOFollowerRobotConfig(
            port=robot_cfg.get("port", ""),
            cameras={},
            use_degrees=robot_cfg.get("use_degrees", True),
        )
        robot = make_robot_from_config(so_config)
        arm_config = ArmControllerConfig(
            urdf_path=robot_cfg.get("urdf", "SO101/so101_new_calib.urdf"),
            ee_frame=robot_cfg.get("ee_frame", "gripper_frame_link"),
            home_joints_deg=robot_cfg.get("home_joints"),
        )
        arm = ArmController(robot=robot, config=arm_config)
        arm.connect()

    # Init recorder
    collector = None
    if record:
        from robot.dataset_collector import DatasetCollector
        collector = DatasetCollector(task=task)

    # Run evaluation episodes
    results = {
        "num_episodes": num_episodes,
        "successes": 0,
        "failures": 0,
        "episodes": [],
    }

    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

    for ep_idx in range(num_episodes):
        logger.info("=== Episode %d/%d ===", ep_idx + 1, num_episodes)

        if arm is not None:
            arm.home()

        if collector is not None:
            collector.start_episode(task)

        episode_reward = 0.0
        for step in range(max_steps_per_episode):
            # Get observation
            if dry_run:
                rgb = cv2.imread(dry_run_image)
                if rgb is None:
                    raise FileNotFoundError(f"Cannot read: {dry_run_image}")
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                depth = np.full(rgb.shape[:2], 500, dtype=np.uint16)
            else:
                rgb = camera.read()
                depth = camera.read_depth()

            # Build observation dict for the policy
            obs = {}
            if arm is not None:
                robot_obs = arm.robot.get_observation()
                for m in motor_names:
                    obs[f"{m}.pos"] = float(robot_obs[f"{m}.pos"])
                obs["gripper.pos"] = float(robot_obs.get("gripper.pos", 100.0))

            # Add image observation
            obs["front"] = rgb

            # Preprocess and run policy
            if preprocessor is not None:
                processed = preprocessor(obs)
            else:
                processed = obs

            with torch.no_grad():
                policy_output = policy(processed)

            if postprocessor is not None:
                action = postprocessor(policy_output)
            else:
                action = policy_output

            # Execute action on robot
            if arm is not None and not dry_run:
                action_dict = {}
                if isinstance(action, dict):
                    action_dict = action
                else:
                    # Assume flat tensor/array of joint positions
                    action_np = np.asarray(action).flatten()
                    for i, m in enumerate(motor_names):
                        if i < len(action_np):
                            action_dict[f"{m}.pos"] = float(action_np[i])
                    if len(action_np) > len(motor_names):
                        action_dict["gripper.pos"] = float(action_np[len(motor_names)])

                arm.robot.send_action(action_dict)

            # Record step
            if collector is not None and arm is not None:
                joints = arm.get_joint_positions()
                gripper = arm.get_gripper_position()
                action_joints = np.array([action.get(f"{m}.pos", 0.0) for m in motor_names]
                                         if isinstance(action, dict) else joints)
                gripper_act = float(action.get("gripper.pos", gripper) if isinstance(action, dict) else gripper)
                collector.record_step(rgb, depth, joints, action_joints, gripper, gripper_act)

            time.sleep(1.0 / cfg.get("lerobot", {}).get("fps", 30))

        # End episode — ask user for success label
        if not dry_run:
            success_input = input("Was this episode successful? (y/n): ").strip().lower()
            success = success_input in ("y", "yes", "1")
        else:
            success = False
            logger.info("[DRY RUN] Episode complete.")
            break

        if collector is not None:
            collector.end_episode(success=success)

        if success:
            results["successes"] += 1
        else:
            results["failures"] += 1

        results["episodes"].append({
            "episode_id": ep_idx,
            "success": success,
        })

    # Compute metrics
    results["success_rate"] = results["successes"] / max(results["num_episodes"], 1)
    logger.info(
        "Evaluation complete: %d/%d successful (%.1f%%)",
        results["successes"],
        results["num_episodes"],
        results["success_rate"] * 100,
    )

    # Cleanup
    if arm is not None:
        arm.home()
        arm.disconnect()
    if camera is not None:
        camera.disconnect()

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained manipulation policy")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to policy checkpoint")
    parser.add_argument("--task", default="pick up the red cube", help="Task description")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-image", default="")
    parser.add_argument("--record", action="store_true", help="Record episodes during evaluation")
    parser.add_argument("--device", default="cpu", help="Torch device (cpu, cuda, mps)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results = evaluate_policy(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        task=args.task,
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        dry_run=args.dry_run,
        dry_run_image=args.dry_run_image,
        record=args.record,
        device=args.device,
    )

    print(f"\nResults: {results['successes']}/{results['num_episodes']} "
          f"({results['success_rate']*100:.1f}% success rate)")


if __name__ == "__main__":
    main()
