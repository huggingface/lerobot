#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
Quasi-static contact and wrench estimation processor step.

This processor estimates contact events using motor effort residuals and optionally
provides wrench estimates using Modern Robotics formulations (TODO implementation).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature

from .core import EnvTransition, TransitionKey
from .pipeline import ProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("contact/quasistatic_wrench")
@dataclass
class QuasiStaticWrenchEstimatorStep(ProcessorStep):
    """
    Processor step for quasi-static contact detection and wrench estimation.
    
    This step detects contact events using motor effort residuals and optionally
    estimates end-effector wrench using Modern Robotics formulations.
    
    Contact detection uses EMA baseline and residual scoring with hysteresis.
    Wrench estimation is disabled by default and contains Modern Robotics TODOs.
    
    Args:
        effort_keys_candidates: List of candidate keys to find effort data in observations.
        strict: If True, raises error when effort data is not found. If False, returns gracefully.
        ema_alpha: Exponential moving average alpha for baseline estimation.
        score_mode: Scoring mode for contact detection ("l2" or "l1mean").
        threshold_on: Contact detection threshold for turning on.
        threshold_off: Contact detection threshold for turning off (defaults to threshold_on).
        min_consecutive_frames: Minimum consecutive frames above threshold to assert contact.
        debug: If True, includes debug outputs like effort_residual.
        enable_wrench: If True, attempts wrench estimation (requires TODO implementation).
        damping: Damping factor for wrench estimation near singularities.
        tau_scale: Scaling factor for external torque estimation from residuals.
        q_key: Override key for joint positions (auto-detected if None).
    
    Inputs:
        transition["observation"]: Must contain motor effort telemetry under one of the candidate keys.
        
    Outputs (added to transition["observation"]):
        contact_score: Contact detection score (B, 1)
        contact_flag: Boolean contact flag (B, 1)
        effort_residual: Effort residuals (B, n_joints) - only if debug=True
        joint_tau_ext_hat: Scaled residual torques (B, n_joints) - optional
        ee_wrench_hat: End-effector wrench estimate (B, 6) - only if enable_wrench=True and TODOs implemented
    """
    
    effort_keys_candidates: list[str] = field(
        default_factory=lambda: ["Present_Current", "Present_Load", "motor_current", "motor_load", "effort", "current", "load"]
    )
    strict: bool = False
    ema_alpha: float = 0.01
    score_mode: str = "l2"
    threshold_on: float = 0.1
    threshold_off: float | None = None
    min_consecutive_frames: int = 1
    debug: bool = False
    enable_wrench: bool = False
    damping: float = 1e-6
    tau_scale: float = 1.0
    q_key: str | None = None
    
    # Internal state - per-sample tracking
    _ema_baseline: Tensor | None = field(default=None, init=False, repr=False)
    _consecutive_counters: Tensor | None = field(default=None, init=False, repr=False)
    _contact_active: Tensor | None = field(default=None, init=False, repr=False)
    _warned_wrench: bool = field(default=False, init=False, repr=False)
    
    def __post_init__(self):
        """Initialize processor parameters."""
        if self.threshold_off is None:
            self.threshold_off = self.threshold_on
            
        if self.score_mode not in ["l2", "l1mean"]:
            raise ValueError(f"score_mode must be 'l2' or 'l1mean', got {self.score_mode}")
            
        if not (0.0 <= self.ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in [0, 1], got {self.ema_alpha}")
    
    def _extract_effort(self, observation: dict[str, Any]) -> Tensor | None:
        """
        Extract effort tensor from observation using candidate keys.
        
        Args:
            observation: Observation dictionary containing effort data.
            
        Returns:
            Effort tensor of shape (B, n_joints) or None if not found.
        """
        for key in self.effort_keys_candidates:
            if key in observation:
                effort = observation[key]
                if isinstance(effort, Tensor):
                    return effort
                elif isinstance(effort, dict):
                    # Handle per-joint dict format: {"joint_1": value, "joint_2": value, ...}
                    if all(isinstance(v, (Tensor, int, float)) for v in effort.values()):
                        # Sort keys for deterministic ordering
                        vals = []
                        for k in sorted(effort.keys()):
                            v = effort[k]
                            if isinstance(v, Tensor):
                                if v.dim() == 0:  # Scalar tensor
                                    vals.append(v.unsqueeze(0).unsqueeze(0))  # (1, 1)
                                elif v.dim() == 1:  # 1D tensor
                                    vals.append(v.view(-1, 1))  # (B, 1)
                                else:  # 2D+ tensor - take first column
                                    vals.append(v.reshape(v.shape[0], -1)[:, :1])  # (B, 1)
                            else:  # scalar int/float
                                vals.append(torch.tensor([[v]]))  # (1, 1)
                        
                        # Concatenate along joint dimension
                        effort = torch.cat(vals, dim=1)  # (B, n_joints)
                        assert effort.ndim == 2, f"Expected 2D effort tensor, got {effort.shape}"
                        return effort
        
        return None
    
    def _extract_joint_positions(self, observation: dict[str, Any]) -> Tensor | None:
        """
        Extract joint positions from observation.
        
        Args:
            observation: Observation dictionary containing joint positions.
            
        Returns:
            Joint position tensor of shape (B, n_joints) or None if not found.
        """
        if self.q_key is not None:
            if self.q_key in observation:
                return observation[self.q_key]
            return None
        
        # Common joint position keys in LeRobot
        q_candidates = ["q", "joint_positions", "joint_pos", "positions", "state"]
        for key in q_candidates:
            if key in observation:
                q = observation[key]
                if isinstance(q, Tensor) and q.dim() >= 2:
                    return q
        
        return None
    
    def _compute_contact_score(self, effort: Tensor, baseline: Tensor) -> Tensor:
        """
        Compute contact score from effort residual.
        
        Args:
            effort: Current effort tensor (B, n_joints).
            baseline: Baseline effort tensor (B, n_joints).
            
        Returns:
            Contact score tensor (B, 1).
        """
        residual = effort - baseline
        
        if self.score_mode == "l2":
            score = torch.norm(residual, dim=1, keepdim=True)
        else:  # l1mean
            score = torch.mean(torch.abs(residual), dim=1, keepdim=True)
        
        return score
    
    def _update_ema_baseline(self, effort: Tensor):
        """Update EMA baseline with current effort."""
        if self._ema_baseline is None:
            if self.ema_alpha == 0.0:
                # Initialize to zero for exact residual computation
                self._ema_baseline = torch.zeros_like(effort)
            else:
                # Initialize to current effort for zero residual on first call
                self._ema_baseline = effort.clone()
        else:
            if self.ema_alpha == 0.0:
                # No EMA update - baseline stays at zero
                pass
            else:
                self._ema_baseline = (1 - self.ema_alpha) * self._ema_baseline + self.ema_alpha * effort
    
    def _apply_hysteresis(self, score: Tensor) -> Tensor:
        """
        Apply hysteresis and debounce to contact detection.
        
        Args:
            score: Contact score tensor (B, 1).
            
        Returns:
            Contact flag tensor (B, 1) of bool type.
        """
        batch_size = score.shape[0]
        device = score.device
        
        # Initialize or resize state tensors if needed
        if (self._consecutive_counters is None or 
            self._contact_active is None or 
            self._consecutive_counters.shape[0] != batch_size):
            
            self._consecutive_counters = torch.zeros(batch_size, 1, dtype=torch.int32, device=device)
            self._contact_active = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        
        # Vectorized hysteresis logic
        above_threshold = score > self.threshold_on
        below_threshold = score < self.threshold_off
        
        # Update consecutive counters
        self._consecutive_counters = torch.where(
            above_threshold & ~self._contact_active,
            self._consecutive_counters + 1,
            torch.where(
                below_threshold,
                torch.zeros_like(self._consecutive_counters),
                self._consecutive_counters
            )
        )
        
        # Update contact flags
        self._contact_active = torch.where(
            above_threshold & (self._consecutive_counters >= self.min_consecutive_frames),
            torch.ones_like(self._contact_active),
            torch.where(
                below_threshold,
                torch.zeros_like(self._contact_active),
                self._contact_active
            )
        )
        
        return self._contact_active
    
    # TODO: Modern Robotics functions - implement these following the book references
    def _compute_body_jacobian(self, q: Tensor) -> Tensor | None:
        """
        Compute body Jacobian using Modern Robotics formulation.
        
        TODO: Implement using Modern Robotics book Chapter 5 (Velocity Kinematics).
        - Body Jacobian J_b(θ) via Product of Exponentials
        - ModernRobotics library oracle: modern_robotics.JacobianBody(Blist, thetalist)
        
        Args:
            q: Joint positions (B, n) in radians.
            
        Returns:
            Body Jacobian J of shape (B, 6, n) or None if not implemented.
        """
        if not self._warned_wrench:
            warnings.warn("Body Jacobian computation not implemented - returning None", UserWarning)
            self._warned_wrench = True
        return None
    
    def _solve_wrench_from_tau(self, J: Tensor, tau: Tensor, damping: float) -> Tensor | None:
        """
        Solve for wrench from joint torques using damped least squares.
        
        TODO: Implement using Modern Robotics book Chapter 5 (Statics).
        - τ = J^T F
        - Objective: min_F ||J^T F - τ||^2 + λ^2||F||^2
        - Closed form: F = (J J^T + λ^2 I)^{-1} (J τ)
        - Use torch.linalg.solve on (6x6) per batch.
        
        Args:
            J: Jacobian matrix (B, 6, n).
            tau: Joint torques (B, n).
            damping: Damping factor λ.
            
        Returns:
            Wrench vector F of shape (B, 6) or None if not implemented.
        """
        if not self._warned_wrench:
            warnings.warn("Wrench solving not implemented - returning None", UserWarning)
            self._warned_wrench = True
        return None
    
    def _manipulability_or_cond(self, J: Tensor) -> Tensor | None:
        """
        Compute manipulability or condition number metric.
        
        TODO: Implement using Modern Robotics book Chapter 5 (Manipulability ellipsoids).
        - Suggested metrics: cond(J J^T) or det(J J^T)^(1/2)
        
        Args:
            J: Jacobian matrix (B, 6, n).
            
        Returns:
            Manipulability metric or None if not implemented.
        """
        if not self._warned_wrench:
            warnings.warn("Manipulability computation not implemented - returning None", UserWarning)
            self._warned_wrench = True
        return None
    
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Process transition to detect contact and optionally estimate wrench.
        
        Args:
            transition: Input environment transition.
            
        Returns:
            Modified transition with contact detection outputs.
        """
        self._current_transition = transition
        new_transition = transition.copy()
        
        observation = transition.get(TransitionKey.OBSERVATION, {})
        if observation is None:
            observation = {}
            new_transition[TransitionKey.OBSERVATION] = observation
        
        # Extract effort data
        effort = self._extract_effort(observation)
        if effort is None:
            if self.strict:
                available_keys = list(observation.keys())
                raise ValueError(
                    f"No effort data found in observation. Tried keys: {self.effort_keys_candidates}. "
                    f"Available keys: {available_keys}"
                )
            # Graceful fallback: set outputs to zeros/false
            batch_size = 1  # Default batch size for empty observation
            observation["contact_score"] = torch.zeros(batch_size, 1)
            observation["contact_flag"] = torch.zeros(batch_size, 1, dtype=torch.bool)
            return new_transition
        
        batch_size, n_joints = effort.shape
        
        # Get baseline for current computation (baseline_{t-1})
        if self._ema_baseline is None:
            if self.ema_alpha == 0.0:
                baseline_prev = torch.zeros_like(effort)
            else:
                baseline_prev = effort.clone()  # Zero residual on first call
        else:
            baseline_prev = self._ema_baseline
        
        # Compute contact score using baseline_{t-1}
        contact_score = self._compute_contact_score(effort, baseline_prev)
        
        # Apply hysteresis for contact flag
        contact_flag = self._apply_hysteresis(contact_score)
        
        # Add outputs to observation
        observation["contact_score"] = contact_score
        observation["contact_flag"] = contact_flag
        
        # Debug outputs (use same residual as scoring)
        if self.debug:
            effort_residual = effort - baseline_prev
            observation["effort_residual"] = effort_residual
            observation["joint_tau_ext_hat"] = self.tau_scale * effort_residual
        
        # Update EMA baseline after computing outputs
        self._update_ema_baseline(effort)
        
        # Wrench estimation (if enabled)
        if self.enable_wrench:
            q = self._extract_joint_positions(observation)
            if q is not None:
                # Compute Jacobian
                J = self._compute_body_jacobian(q)
                
                if J is not None:
                    # Estimate external torques from residuals (same as debug)
                    tau_ext_hat = self.tau_scale * (effort - baseline_prev)
                    
                    # Solve for wrench
                    ee_wrench_hat = self._solve_wrench_from_tau(J, tau_ext_hat, self.damping)
                    if ee_wrench_hat is not None:
                        observation["ee_wrench_hat"] = ee_wrench_hat
                        
                        # Optional: add manipulability metric
                        manip_metric = self._manipulability_or_cond(J)
                        if manip_metric is not None:
                            observation["manipulability_metric"] = manip_metric
                    else:
                        # Wrench solving failed
                        observation["ee_wrench_hat"] = torch.zeros(batch_size, 6)
                        observation["wrench_unavailable"] = True
                else:
                    # Jacobian computation failed
                    observation["ee_wrench_hat"] = torch.zeros(batch_size, 6)
                    observation["wrench_unavailable"] = True
            else:
                # Joint positions not available
                observation["ee_wrench_hat"] = torch.zeros(batch_size, 6)
                observation["wrench_unavailable"] = True
        
        return new_transition
    
    def get_config(self) -> dict[str, Any]:
        """Return processor configuration."""
        return {
            "effort_keys_candidates": self.effort_keys_candidates,
            "strict": self.strict,
            "ema_alpha": self.ema_alpha,
            "score_mode": self.score_mode,
            "threshold_on": self.threshold_on,
            "threshold_off": self.threshold_off,
            "min_consecutive_frames": self.min_consecutive_frames,
            "debug": self.debug,
            "enable_wrench": self.enable_wrench,
            "damping": self.damping,
            "tau_scale": self.tau_scale,
            "q_key": self.q_key,
        }
    
    def state_dict(self) -> dict[str, Tensor]:
        """Return processor state for serialization."""
        state = {}
        if self._ema_baseline is not None:
            state["_ema_baseline"] = self._ema_baseline
        if self._consecutive_counters is not None:
            state["_consecutive_counters"] = self._consecutive_counters
        if self._contact_active is not None:
            state["_contact_active"] = self._contact_active
        return state
    
    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        """Load processor state from dictionary."""
        if "_ema_baseline" in state:
            self._ema_baseline = state["_ema_baseline"]
        if "_consecutive_counters" in state:
            self._consecutive_counters = state["_consecutive_counters"]
        if "_contact_active" in state:
            self._contact_active = state["_contact_active"]
    
    def reset(self) -> None:
        """Reset processor internal state."""
        self._ema_baseline = None
        self._consecutive_counters = None
        self._contact_active = None
        self._warned_wrench = False
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Transform feature descriptions to include new outputs.
        
        Args:
            features: Input feature descriptions.
            
        Returns:
            Modified feature descriptions with new outputs.
        """
        new_features = features.copy()
        
        # Add observation features if they exist
        if "observation" in new_features:
            obs_features = new_features["observation"].copy()
            
            # Always add contact outputs
            obs_features["contact_score"] = PolicyFeature(
                type="continuous",
                shape=(1,),
                dtype="float32"
            )
            obs_features["contact_flag"] = PolicyFeature(
                type="binary",
                shape=(1,),
                dtype="bool"
            )
            
            # Add debug outputs if enabled
            if self.debug:
                # Note: We don't know the exact shape without knowing n_joints
                # This would typically be inferred from actual data
                pass
            
            # Add wrench outputs if enabled
            if self.enable_wrench:
                obs_features["ee_wrench_hat"] = PolicyFeature(
                    type="continuous",
                    shape=(6,),
                    dtype="float32"
                )
            
            new_features["observation"] = obs_features
        
        return new_features
