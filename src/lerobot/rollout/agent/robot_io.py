# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Real-robot adapters using the existing rollout context and dataset writer."""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from pathlib import Path

import draccus
import numpy as np
import torch

from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.language import language_feature_info
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.rollout import RolloutConfig, build_rollout_context
from lerobot.utils.feature_utils import build_dataset_frame


class RealRobotIO:
    def __init__(self, rollout_args, *, root, repo_id, fps, resume=False):
        if not math.isfinite(fps) or fps <= 0 or int(fps) != fps:
            raise ValueError("Recorded rollout fps must be a positive integer")
        argv = [*rollout_args, "--strategy.type=base", f"--fps={fps}", "--return_to_initial_position=false"]
        previous = sys.argv
        try:
            # RolloutConfig resolves policy paths/overrides through parser helpers reading sys.argv.
            sys.argv = [previous[0], *argv]
            cfg = draccus.parse(
                RolloutConfig, args=parser.filter_path_args(RolloutConfig.__get_path_fields__(), argv)
            )
        finally:
            sys.argv = previous
        if cfg.robot is None:
            raise ValueError("The physical agent requires a real --robot.type")
        if getattr(cfg.policy, "compile_model", False):
            cfg.policy.compile_model = False
        if getattr(cfg.policy, "gradient_checkpointing", False):
            cfg.policy.gradient_checkpointing = False
        self.ctx = build_rollout_context(cfg, threading.Event(), build_inference=False)
        self.robot = self.ctx.hardware.robot_wrapper
        self._revision = None
        self._closed = False
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.journal = (self.root / "events.jsonl").open("a", buffering=1)
        self.results = (self.root / "episodes.jsonl").open("a", buffering=1)
        self.fps = fps
        self.frames = 0
        self.intervention_frames = 0
        self.policy_frames = 0
        features = {
            **self.ctx.data.dataset_features,
            **language_feature_info(),
            "intervention": {"dtype": "bool", "shape": (1,), "names": None},
        }
        try:
            if resume:
                self.dataset = LeRobotDataset.resume(
                    repo_id, root=self.root / "dataset", image_writer_threads=6
                )
                if self.dataset.fps != fps or self.dataset.meta.robot_type != self.robot.robot_type:
                    raise ValueError("Recording metadata does not match this robot/control rate")
            else:
                self.dataset = LeRobotDataset.create(
                    repo_id,
                    root=self.root / "dataset",
                    fps=int(fps),
                    robot_type=self.robot.robot_type,
                    features=features,
                    image_writer_threads=6,
                )
        except Exception:
            self.journal.close()
            self.results.close()
            self.robot.inner.disconnect()
            raise

    def observe(self):
        # Robot I/O is called only on the runtime thread.
        return self.robot.get_observation()

    def predict(self, snapshot):
        ctx = self.ctx
        if snapshot["revision"] != self._revision:
            ctx.policy.policy.reset()
            ctx.policy.preprocessor.reset()
            ctx.policy.postprocessor.reset()
            self._revision = snapshot["revision"]
        observation = dict(snapshot["raw"]["_processed"])
        observation = build_dataset_frame(ctx.data.dataset_features, observation, prefix="observation")
        observation = prepare_observation_for_inference(
            observation, torch.device(ctx.runtime.cfg.device), snapshot["task"], self.robot.robot_type
        )
        with torch.inference_mode():
            observation = ctx.policy.preprocessor(observation)
            chunk = ctx.policy.policy.predict_action_chunk(observation)
            # Process the complete chunk against its originating state, including relative actions.
            chunk = ctx.policy.postprocessor(chunk)
        if not isinstance(chunk, torch.Tensor):
            raise TypeError("This controller requires a tensor-valued action chunk")
        if chunk.ndim == 3:
            if chunk.shape[0] != 1:
                raise ValueError("Real robot inference requires batch size one")
            chunk = chunk[0]
        if chunk.ndim == 1:
            chunk = chunk.unsqueeze(0)
        if chunk.ndim != 2 or not len(chunk) or not torch.isfinite(chunk).all():
            raise ValueError("Invalid action chunk")
        return [make_robot_action(action.cpu().unsqueeze(0), ctx.data.dataset_features) for action in chunk]

    def execute(self, action):
        # The Robot.send_action return value includes driver-side clipping.
        processed = self.ctx.processors.robot_action_processor((action, self._latest_raw))
        return self.robot.send_action(processed)

    def record(self, snapshot, proposed, applied, source, instruction):
        obs = dict(snapshot["raw"]["_processed"])
        frame = build_dataset_frame(self.ctx.data.dataset_features, obs, prefix="observation")
        frame.update(build_dataset_frame(self.ctx.data.dataset_features, applied, prefix="action"))
        frame.update(
            task=snapshot["task"],
            intervention=np.array([source == "tool"], dtype=bool),
            language_persistent=[
                {
                    "role": "assistant",
                    "content": instruction,
                    "style": "subtask",
                    "timestamp": self.frames / self.fps,
                    "camera": None,
                    "tool_calls": None,
                }
            ],
            language_events=[],
        )
        self.dataset.add_frame(frame)
        self.event(
            {
                "kind": "action",
                "frame": self.frames,
                "observation_id": snapshot["id"],
                "observed_at": snapshot["observed_at"],
                "source": source,
                "proposed": proposed,
                "applied": applied,
                "instruction": instruction,
            }
        )
        self.frames += 1
        self.intervention_frames += source == "tool"
        self.policy_frames += source == "policy"

    def event(self, event):
        self.journal.write(json.dumps(event, allow_nan=False) + "\n")

    def finish(self, result):
        if self.frames:
            self.dataset.save_episode()
        self.results.write(
            json.dumps(
                {
                    **result,
                    "episode_index": self.dataset.num_episodes - 1 if self.frames else None,
                    "frames": self.frames,
                    "intervention_frames": self.intervention_frames,
                    "policy_frames": self.policy_frames,
                    "finished_at": time.time(),
                }
            )
            + "\n"
        )
        self.frames = self.intervention_frames = self.policy_frames = 0

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.dataset.finalize()
        finally:
            self.journal.close()
            self.results.close()
            self.robot.inner.disconnect()

    def limits(self):
        config = self.robot.inner.config
        if hasattr(config, "left_arm_config"):
            return {
                f"{side}_{key}.pos": bounds
                for side in ("left", "right")
                for key, bounds in getattr(config, f"{side}_arm_config").joint_limits.items()
            }
        return {f"{key}.pos": bounds for key, bounds in config.joint_limits.items()}

    def observation_provider(self):
        self._latest_raw = self.observe()
        return {
            **self._latest_raw,
            "_processed": self.ctx.processors.robot_observation_processor(dict(self._latest_raw)),
        }


class CartesianTools:
    """Robot-frame IK with explicit, installation-specific joint mappings and URDFs."""

    def __init__(self, arms):
        from lerobot.model.kinematics import RobotKinematics

        self.arms = {}
        for name, config in arms.items():
            solver = RobotKinematics(
                config["urdf_path"],
                target_frame_name=config["target_frame"],
                joint_names=config["urdf_joint_names"],
            )
            if len(config["robot_joint_names"]) != len(config["urdf_joint_names"]):
                raise ValueError("IK joint mappings must have the same length")
            self.arms[name] = (config, solver)

    def __call__(self, arguments, joints):
        config, solver = self.arms[arguments["arm"]]
        if arguments["reference_frame"] != config["reference_frame"]:
            raise ValueError("Target frame does not match the calibrated IK frame")
        pose = np.asarray(arguments["pose"], dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("pose must be a finite 4x4 transform in metres")
        if (
            not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
            or not math.isclose(np.linalg.det(pose[:3, :3]), 1, abs_tol=1e-4)
        ):
            raise ValueError("pose must be a rigid transform")
        current = np.array([joints[key] for key in config["robot_joint_names"]])
        target = solver.inverse_kinematics(current, pose)
        achieved = solver.forward_kinematics(target)
        if np.linalg.norm(achieved[:3, 3] - pose[:3, 3]) > config.get("position_tolerance_m", 0.01):
            raise ValueError("IK did not reach the requested position")
        if not np.allclose(achieved[:3, :3], pose[:3, :3], atol=config.get("rotation_tolerance", 0.05)):
            raise ValueError("IK did not reach the requested orientation")
        return dict(zip(config["robot_joint_names"], target.tolist(), strict=True))
