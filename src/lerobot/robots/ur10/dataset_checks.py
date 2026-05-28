"""
Production-grade dataset verification checks for UR10 HIL-SERL demos.

Each check is a self-contained class that accumulates per-frame stats during a single
pass over the dataset, then produces a `CheckResult` describing pass/fail/warn status,
counts, and (where useful) a list of offending frames for triage.

The orchestrator (`verify_dataset.py`) loads the dataset once, iterates frames in order,
calls `process_frame` on every enabled check, and finally prints a unified report.

Severity vocabulary:
    BLOCKER  : violations that will silently break training (NaN/Inf, schema mismatch,
               action outside declared bounds). The runner exits non-zero.
    WARN     : violations that degrade training but won't crash it (state outside
               dataset_stats — saturates normalization; missing toggles per episode).
    INFO     : statistics only; never fails.

Adding a new check:
    1. Subclass `Check`, set `name` and `severity`, implement `setup` + `process_frame` +
       `finalize`.
    2. Register it in `ALL_CHECKS` in `verify_dataset.py`.
    3. Document the invariant it enforces in its docstring.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Severity + result types
# ---------------------------------------------------------------------------


SEVERITY_BLOCKER = "BLOCKER"
SEVERITY_WARN = "WARN"
SEVERITY_INFO = "INFO"

STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"
STATUS_SKIP = "SKIP"


@dataclass
class CheckResult:
    name: str
    severity: str
    status: str           # one of STATUS_*
    summary: str          # one-line message
    n_violations: int = 0
    details: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def _scalar(v: Any) -> float:
    if isinstance(v, torch.Tensor):
        try:
            return float(v.item())
        except (ValueError, RuntimeError):
            return float(v.flatten()[0].item())
    if isinstance(v, np.ndarray):
        return float(v.flatten()[0])
    try:
        return float(v[0])
    except (TypeError, IndexError):
        return float(v)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Check(ABC):
    name: str = "check"
    severity: str = SEVERITY_WARN

    def setup(self, ds, ctx: dict[str, Any]) -> bool:
        """Validate config / dataset compatibility. Return False to skip this check."""
        return True

    @abstractmethod
    def process_frame(self, frame_idx: int, episode_idx: int, sample: dict) -> None:
        ...

    @abstractmethod
    def finalize(self) -> CheckResult:
        ...


# ---------------------------------------------------------------------------
# Schema check
# ---------------------------------------------------------------------------


class SchemaCheck(Check):
    """Dataset feature shapes match the train_config's declared shapes.

    Catches the kind of regression that produced the rc10 16-vs-22 bug — env emits a
    22-D `observation.state` while the train config still declares `[16]`. With this
    check, that mismatch fails loudly before training starts.

    Invariants enforced:
      - `observation.state.shape` matches train_config `env.features.observation.state.shape`
        AND `policy.input_features.observation.state.shape`.
      - `action.shape` matches `env.features.action.shape`.
      - `dataset_stats."observation.state".min/max` length equals state shape.
      - `dataset_stats.action.min/max` length equals action-continuous-channels (for
        UR10: 3 in the no-yaw layout, 4 in the yaw-enabled layout, ignoring the
        discrete gripper channel handled by `num_discrete_actions`).

    Also internal-consistency: when `complementary_info.discrete_penalty` is present
    (gripper enabled flow), action.shape is either (4,) for the standard layout
    (`[dx, dy, dz, gripper]`) or (5,) for the yaw-enabled layout
    (`[dx, dy, dz, dyaw, gripper]`). The gripper stays the last element in both.
    """

    name = "schema"
    severity = SEVERITY_BLOCKER

    def __init__(self, train_config: dict | None = None):
        self.train_config = train_config
        self.violations: list[str] = []
        self._setup_done = False
        self._ds_features: dict | None = None

    def setup(self, ds, ctx: dict[str, Any]) -> bool:
        self._ds_features = dict(ds.features)
        self._setup_done = True
        return True

    def process_frame(self, frame_idx, episode_idx, sample) -> None:
        # Schema check is static — no per-frame work.
        pass

    def finalize(self) -> CheckResult:
        feats = self._ds_features or {}

        def _shape(entry):
            if entry is None:
                return None
            if isinstance(entry, dict):
                return tuple(entry.get("shape", ()))
            return tuple(getattr(entry, "shape", ()))

        ds_state = _shape(feats.get("observation.state"))
        ds_action = _shape(feats.get("action"))
        ds_pen = _shape(feats.get("complementary_info.discrete_penalty"))

        if ds_state is None:
            self.violations.append("dataset is missing observation.state feature")
        if ds_action is None:
            self.violations.append("dataset is missing action feature")

        # Internal consistency: gripper present ⇒ action is (4,) [xyz+gripper] or
        # (5,) [xyz+yaw+gripper]. Gripper always at the last index regardless of layout.
        if ds_pen is not None and ds_action is not None and ds_action not in {(4,), (5,)}:
            self.violations.append(
                f"complementary_info.discrete_penalty is present but action.shape={ds_action}; "
                "expected (4,) for [dx, dy, dz, gripper] or (5,) for [dx, dy, dz, dyaw, gripper]"
            )

        if self.train_config is not None:
            tc = self.train_config
            try:
                tc_env_state = tuple(tc["env"]["features"]["observation.state"]["shape"])
            except (KeyError, TypeError):
                tc_env_state = None
            try:
                tc_pol_state = tuple(tc["policy"]["input_features"]["observation.state"]["shape"])
            except (KeyError, TypeError):
                tc_pol_state = None
            try:
                tc_action = tuple(tc["env"]["features"]["action"]["shape"])
            except (KeyError, TypeError):
                tc_action = None
            try:
                ds_stats = tc["policy"]["dataset_stats"]["observation.state"]
                tc_state_min = list(ds_stats["min"])
                tc_state_max = list(ds_stats["max"])
            except (KeyError, TypeError):
                tc_state_min = tc_state_max = None
            try:
                ds_stats_a = tc["policy"]["dataset_stats"]["action"]
                tc_action_min = list(ds_stats_a["min"])
                tc_action_max = list(ds_stats_a["max"])
            except (KeyError, TypeError):
                tc_action_min = tc_action_max = None

            if tc_env_state and ds_state and tc_env_state != ds_state:
                self.violations.append(
                    f"train_config env.features.observation.state.shape={tc_env_state} != "
                    f"dataset observation.state.shape={ds_state}"
                )
            if tc_pol_state and ds_state and tc_pol_state != ds_state:
                self.violations.append(
                    f"train_config policy.input_features.observation.state.shape={tc_pol_state} "
                    f"!= dataset observation.state.shape={ds_state}"
                )
            if tc_action and ds_action and tc_action != ds_action:
                # UR10 ships ds_action=(4,) (gamepad) while train_config declares (3,) for
                # the SAC continuous head — gripper is handled by num_discrete_actions.
                # That mismatch is INTENTIONAL when num_discrete_actions == 3.
                ndisc = tc.get("policy", {}).get("num_discrete_actions", None)
                if ndisc is None or ds_action != (4,) or tc_action != (3,):
                    self.violations.append(
                        f"train_config env.features.action.shape={tc_action} != "
                        f"dataset action.shape={ds_action}"
                    )
            if tc_state_min and ds_state and len(tc_state_min) != ds_state[0]:
                self.violations.append(
                    f"dataset_stats.observation.state.min has {len(tc_state_min)} entries, "
                    f"expected {ds_state[0]}"
                )
            if tc_state_max and ds_state and len(tc_state_max) != ds_state[0]:
                self.violations.append(
                    f"dataset_stats.observation.state.max has {len(tc_state_max)} entries, "
                    f"expected {ds_state[0]}"
                )
            # action stats expected to match the *continuous* action dim (3 for UR10).
            if tc_action_min and tc_action and len(tc_action_min) != tc_action[0]:
                self.violations.append(
                    f"dataset_stats.action.min has {len(tc_action_min)} entries, "
                    f"expected {tc_action[0]} (train_config action.shape)"
                )

        if self.violations:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status=STATUS_FAIL,
                summary=f"{len(self.violations)} schema violation(s)",
                n_violations=len(self.violations),
                details=self.violations,
                stats={"ds_state": ds_state, "ds_action": ds_action},
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status=STATUS_PASS,
            summary=f"observation.state {ds_state}, action {ds_action} consistent",
            stats={"ds_state": ds_state, "ds_action": ds_action},
        )


# ---------------------------------------------------------------------------
# Finite check
# ---------------------------------------------------------------------------


class FiniteCheck(Check):
    """No NaN / Inf in `action`, `observation.state`, or any image tensor.

    A single bad frame poisons SAC's loss and silently corrupts checkpoints. We scan
    every numeric tensor in every frame.
    """

    name = "finite"
    severity = SEVERITY_BLOCKER

    def __init__(self, max_details: int = 20):
        self.max_details = max_details
        self.n_violations = 0
        self.details: list[str] = []
        self.checked_keys: set[str] = set()

    def setup(self, ds, ctx) -> bool:
        return True

    def process_frame(self, frame_idx, episode_idx, sample) -> None:
        for key, val in sample.items():
            if not isinstance(val, torch.Tensor):
                continue
            if not torch.is_floating_point(val):
                continue
            self.checked_keys.add(key)
            if not torch.isfinite(val).all():
                self.n_violations += 1
                if len(self.details) < self.max_details:
                    n_nan = int(torch.isnan(val).sum().item())
                    n_inf = int(torch.isinf(val).sum().item())
                    self.details.append(
                        f"frame {frame_idx} ep {episode_idx}: {key} has "
                        f"{n_nan} NaN, {n_inf} Inf"
                    )

    def finalize(self) -> CheckResult:
        if self.n_violations == 0:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status=STATUS_PASS,
                summary=f"all tensors finite across {len(self.checked_keys)} float keys",
                stats={"checked_keys": sorted(self.checked_keys)},
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status=STATUS_FAIL,
            summary=f"{self.n_violations} non-finite tensor(s) found",
            n_violations=self.n_violations,
            details=self.details,
        )


# ---------------------------------------------------------------------------
# Action bounds check
# ---------------------------------------------------------------------------


class ActionBoundsCheck(Check):
    """Continuous action channels in [-1, +1]; discrete gripper in {0, 1, 2}.

    A teleop bug that ships malformed actions (e.g. [-2, 0, 0, 1]) would silently land
    in the dataset and be normalized to garbage by MIN_MAX clamping at training time.
    Catch it at audit time.

    Auto-detects yaw-enabled vs no-yaw layouts from action_dim on the first frame
    when `action_continuous_dims=None`:
        action_dim == 3  → all continuous (no gripper)
        action_dim == 4  → 3 continuous + 1 gripper (no yaw)
        action_dim == 5  → 4 continuous + 1 gripper (yaw enabled)
    Pass `action_continuous_dims` explicitly to override.
    """

    name = "action_bounds"
    severity = SEVERITY_BLOCKER

    def __init__(self, action_continuous_dims: int | None = None, max_details: int = 20):
        # None ⇒ derive per-frame from `action_dim` (xyz+yaw+gripper layout-aware).
        self.cont_dims = action_continuous_dims
        self.max_details = max_details
        self.n_violations = 0
        self.details: list[str] = []
        self._action_dim: int | None = None

    def process_frame(self, frame_idx, episode_idx, sample) -> None:
        a = sample.get("action")
        if a is None:
            return
        if self._action_dim is None:
            self._action_dim = int(a.shape[-1]) if hasattr(a, "shape") else len(a)
        if self.cont_dims is None:
            # Heuristic: dim ≥ 4 implies the last channel is the discrete gripper.
            # dim == 3 is the no-gripper xyz-only layout (all continuous).
            self.cont_dims = (self._action_dim - 1) if self._action_dim >= 4 else self._action_dim
        if isinstance(a, torch.Tensor):
            a_np = a.detach().cpu().numpy()
        else:
            a_np = np.asarray(a)

        # Continuous channels
        cont = a_np[: self.cont_dims]
        if (cont < -1.0 - 1e-6).any() or (cont > 1.0 + 1e-6).any():
            self.n_violations += 1
            if len(self.details) < self.max_details:
                self.details.append(
                    f"frame {frame_idx} ep {episode_idx}: continuous action "
                    f"{cont.tolist()} outside [-1, 1]"
                )

        # Discrete gripper channel (if present)
        if a_np.shape[-1] > self.cont_dims:
            g = a_np[-1]
            g_int = int(round(float(g)))
            if g_int not in (0, 1, 2) or abs(float(g) - g_int) > 1e-3:
                self.n_violations += 1
                if len(self.details) < self.max_details:
                    self.details.append(
                        f"frame {frame_idx} ep {episode_idx}: gripper action "
                        f"{float(g):.4f} not in {{0, 1, 2}}"
                    )

    def finalize(self) -> CheckResult:
        if self.n_violations == 0:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status=STATUS_PASS,
                summary=(
                    f"action_dim={self._action_dim} (cont_dims={self.cont_dims}): "
                    "all continuous in [-1,1], gripper in {0,1,2}"
                ),
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status=STATUS_FAIL,
            summary=f"{self.n_violations} out-of-bounds action(s)",
            n_violations=self.n_violations,
            details=self.details,
        )


# ---------------------------------------------------------------------------
# State bounds check
# ---------------------------------------------------------------------------


class StateBoundsCheck(Check):
    """`observation.state[i]` falls inside the train_config `dataset_stats` min/max.

    MIN_MAX normalization in the SAC policy clips out-of-bound values, saturating the
    state representation and starving the gradient. If `find_limits` ran on a workspace
    different from the recording workspace, this check catches it before training.

    Counts per-dimension violations and lists the top offenders.
    """

    name = "state_bounds"
    severity = SEVERITY_WARN

    def __init__(self, train_config: dict | None = None, max_details: int = 20):
        self.train_config = train_config
        self.max_details = max_details
        self.min: np.ndarray | None = None
        self.max: np.ndarray | None = None
        self.per_dim_lo = None  # int counts per dim
        self.per_dim_hi = None
        self.n_violations = 0
        self.details: list[str] = []

    def setup(self, ds, ctx) -> bool:
        if self.train_config is None:
            return False
        try:
            stats = self.train_config["policy"]["dataset_stats"]["observation.state"]
            self.min = np.asarray(stats["min"], dtype=np.float64)
            self.max = np.asarray(stats["max"], dtype=np.float64)
        except (KeyError, TypeError):
            return False
        n = self.min.shape[0]
        self.per_dim_lo = np.zeros(n, dtype=np.int64)
        self.per_dim_hi = np.zeros(n, dtype=np.int64)
        return True

    def process_frame(self, frame_idx, episode_idx, sample) -> None:
        s = sample.get("observation.state")
        if s is None or self.min is None:
            return
        if isinstance(s, torch.Tensor):
            s = s.detach().cpu().numpy()
        else:
            s = np.asarray(s)
        s = s.astype(np.float64).flatten()
        if s.shape[0] != self.min.shape[0]:
            return  # let SchemaCheck handle this
        lo_mask = s < self.min - 1e-6
        hi_mask = s > self.max + 1e-6
        if lo_mask.any() or hi_mask.any():
            self.n_violations += 1
            self.per_dim_lo += lo_mask.astype(np.int64)
            self.per_dim_hi += hi_mask.astype(np.int64)
            if len(self.details) < self.max_details:
                # report first offending dim for compactness
                idxs = np.where(lo_mask | hi_mask)[0]
                for d in idxs[:3]:
                    side = "below" if lo_mask[d] else "above"
                    bound = self.min[d] if side == "below" else self.max[d]
                    self.details.append(
                        f"frame {frame_idx} ep {episode_idx}: state[{d}]={s[d]:.4f} "
                        f"{side} {bound:.4f}"
                    )

    def finalize(self) -> CheckResult:
        if self.min is None:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status=STATUS_SKIP,
                summary="no train_config / dataset_stats — skipped",
            )
        if self.n_violations == 0:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status=STATUS_PASS,
                summary="all state values within dataset_stats bounds",
            )
        # Build per-dim offender summary, top 3 worst dims
        worst = np.argsort(-(self.per_dim_lo + self.per_dim_hi))[:3]
        worst_summary = ", ".join(
            f"dim {int(d)}: {int(self.per_dim_lo[d])} below + {int(self.per_dim_hi[d])} above"
            for d in worst
            if (self.per_dim_lo[d] + self.per_dim_hi[d]) > 0
        )
        stats = {
            "per_dim_below": self.per_dim_lo.tolist(),
            "per_dim_above": self.per_dim_hi.tolist(),
        }
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status=STATUS_WARN,
            summary=f"{self.n_violations} frames out-of-bounds; worst dims: {worst_summary}",
            n_violations=self.n_violations,
            details=self.details,
            stats=stats,
        )


# ---------------------------------------------------------------------------
# Gripper penalty check (refactored from verify_gripper_penalty.py)
# ---------------------------------------------------------------------------


class GripperPenaltyCheck(Check):
    """Recorded `complementary_info.discrete_penalty` matches gym-hil-style
    state-change semantics computed against the previous frame's gripper state.

    See `UR10GripperPenaltyProcessorStep` for the rule itself.
    """

    name = "gripper_penalty"
    severity = SEVERITY_BLOCKER

    def __init__(
        self,
        penalty: float = -0.02,
        gripper_state_index: int = -1,
        gripper_action_index: int = -1,
        max_details: int = 20,
    ):
        self.penalty = penalty
        self.sa_idx = gripper_state_index
        self.aa_idx = gripper_action_index
        self.max_details = max_details
        self.has_pen_col = False
        self.prev_state_by_ep: dict[int, float | None] = {}
        self.n_frames = 0
        self.classes: dict[str, int] = {
            "stay": 0,
            "toggle_close": 0,
            "toggle_open": 0,
            "redundant_close": 0,
            "redundant_open": 0,
            "first_frame": 0,
        }
        self.exp_sum = 0.0
        self.rec_sum = 0.0
        self.n_mismatch = 0
        self.details: list[str] = []
        self.per_ep: dict[int, dict] = {}

    def setup(self, ds, ctx) -> bool:
        self.has_pen_col = "complementary_info.discrete_penalty" in ds.features
        return True

    @staticmethod
    def _expected(prev_state: float | None, cmd: int, penalty: float) -> tuple[float, str]:
        if cmd == 1:
            return 0.0, "stay"
        if prev_state is None:
            return 0.0, "first_frame"
        prev_open = prev_state >= 0.5
        if cmd == 0:
            if prev_open:
                return penalty, "toggle_close"
            return 0.0, "redundant_close"
        if cmd == 2:
            if not prev_open:
                return penalty, "toggle_open"
            return 0.0, "redundant_open"
        return 0.0, "first_frame"  # unknown cmd → silent

    def process_frame(self, frame_idx, episode_idx, sample) -> None:
        a = sample.get("action")
        s = sample.get("observation.state")
        if a is None or s is None:
            return
        cmd = int(round(_scalar(a[self.aa_idx] if hasattr(a, "__getitem__") else a)))
        cur_state = _scalar(s[self.sa_idx] if hasattr(s, "__getitem__") else s)
        prev_state = self.prev_state_by_ep.get(episode_idx)
        exp, cls = self._expected(prev_state, cmd, self.penalty)
        rec = _scalar(sample["complementary_info.discrete_penalty"]) if self.has_pen_col else 0.0

        self.classes[cls] = self.classes.get(cls, 0) + 1
        self.n_frames += 1
        self.exp_sum += exp
        self.rec_sum += rec
        if self.has_pen_col and not math.isclose(exp, rec, abs_tol=1e-6):
            self.n_mismatch += 1
            if len(self.details) < self.max_details:
                self.details.append(
                    f"frame {frame_idx} ep {episode_idx}: cmd={cmd}, prev_state="
                    f"{prev_state if prev_state is not None else 'none'}, "
                    f"expected={exp:+.4f}, recorded={rec:+.4f}"
                )

        ep = self.per_ep.setdefault(
            episode_idx, {"toggle_close": 0, "toggle_open": 0, "exp_sum": 0.0, "rec_sum": 0.0}
        )
        if cls in ("toggle_close", "toggle_open"):
            ep[cls] += 1
        ep["exp_sum"] += exp
        ep["rec_sum"] += rec

        # Update tracker for next frame in this episode.
        self.prev_state_by_ep[episode_idx] = cur_state

    def finalize(self) -> CheckResult:
        toggles = self.classes["toggle_close"] + self.classes["toggle_open"]
        stats = {
            "toggles": toggles,
            "expected_sum": self.exp_sum,
            "recorded_sum": self.rec_sum,
            "classes": self.classes,
            "per_ep": self.per_ep,
        }
        if not self.has_pen_col:
            return CheckResult(
                name=self.name,
                severity=SEVERITY_WARN,
                status=STATUS_WARN,
                summary=(
                    f"`complementary_info.discrete_penalty` column missing — "
                    f"would have recorded {toggles} toggles, sum {self.exp_sum:+.4f}"
                ),
                stats=stats,
            )
        if self.n_mismatch == 0:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status=STATUS_PASS,
                summary=f"{toggles} toggles, exp_sum={self.exp_sum:+.4f}, rec_sum={self.rec_sum:+.4f}",
                stats=stats,
            )
        # Special case: every recorded penalty is 0 — strongly suggests a pre-fix dataset.
        old_dataset_hint = self.classes["toggle_close"] + self.classes["toggle_open"] > 0 and self.rec_sum == 0.0
        msg = f"{self.n_mismatch} frames mismatched; exp_sum={self.exp_sum:+.4f} vs rec_sum={self.rec_sum:+.4f}"
        if old_dataset_hint:
            msg += " (recorded all-zero — pre-fix dataset; re-record to populate)"
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status=STATUS_FAIL,
            summary=msg,
            n_violations=self.n_mismatch,
            details=self.details,
            stats=stats,
        )


# ---------------------------------------------------------------------------
# Gripper activity check
# ---------------------------------------------------------------------------


class GripperActivityCheck(Check):
    """Every demo episode contains at least one close-toggle and one open-toggle.

    For pick/place and insertion tasks an episode without any grasp event is almost
    certainly a botched recording (operator forgot to hold intervention, or the gripper
    USB cable wasn't connected, etc). Don't let those pollute training.

    Implementation note: piggybacks on `GripperPenaltyCheck`'s tracker via shared
    detection logic — kept independent here so it can run standalone too.
    """

    name = "gripper_activity"
    severity = SEVERITY_WARN

    def __init__(
        self,
        gripper_state_index: int = -1,
        gripper_action_index: int = -1,
        min_toggles_per_episode: int = 2,
    ):
        self.sa_idx = gripper_state_index
        self.aa_idx = gripper_action_index
        self.min_per_ep = min_toggles_per_episode
        self.prev_state_by_ep: dict[int, float | None] = {}
        self.toggles_by_ep: dict[int, int] = {}

    def process_frame(self, frame_idx, episode_idx, sample) -> None:
        a = sample.get("action")
        s = sample.get("observation.state")
        if a is None or s is None:
            return
        cmd = int(round(_scalar(a[self.aa_idx])))
        cur_state = _scalar(s[self.sa_idx])
        prev = self.prev_state_by_ep.get(episode_idx)
        if prev is not None and cmd in (0, 2):
            prev_open = prev >= 0.5
            if (cmd == 0 and prev_open) or (cmd == 2 and not prev_open):
                self.toggles_by_ep[episode_idx] = self.toggles_by_ep.get(episode_idx, 0) + 1
        self.prev_state_by_ep[episode_idx] = cur_state
        # Make sure every episode appears in the result map even with 0 toggles
        self.toggles_by_ep.setdefault(episode_idx, 0)

    def finalize(self) -> CheckResult:
        bad = {ep: n for ep, n in self.toggles_by_ep.items() if n < self.min_per_ep}
        total_toggles = sum(self.toggles_by_ep.values())
        per_ep_str = ", ".join(f"ep{ep}:{n}" for ep, n in sorted(self.toggles_by_ep.items()))
        if not bad:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status=STATUS_PASS,
                summary=f"every episode has ≥{self.min_per_ep} toggles ({total_toggles} total)",
                stats={"toggles_by_ep": dict(self.toggles_by_ep)},
            )
        details = [
            f"episode {ep} has only {n} toggle(s) (expected ≥ {self.min_per_ep})"
            for ep, n in sorted(bad.items())
        ]
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status=STATUS_WARN,
            summary=f"{len(bad)} episode(s) below toggle threshold (per-ep: {per_ep_str})",
            n_violations=len(bad),
            details=details,
            stats={"toggles_by_ep": dict(self.toggles_by_ep)},
        )


# ---------------------------------------------------------------------------
# Stationary frames check
# ---------------------------------------------------------------------------


class StationaryFramesCheck(Check):
    """Detect long runs of frames where TCP xyz didn't move.

    A stationary run usually means the operator released the joystick mid-demo. The
    frames are still "valid" demo data but they teach the policy nothing — and at
    pathological lengths they bias the buffer toward "do nothing".

    INFO-only: reports per-episode counts of stationary runs above `run_threshold`
    frames. Doesn't fail the audit.
    """

    name = "stationary_frames"
    severity = SEVERITY_INFO

    def __init__(
        self,
        tcp_xyz_slice: slice = slice(12, 15),
        movement_threshold_m: float = 1e-4,
        run_threshold_frames: int = 20,
    ):
        self.slice = tcp_xyz_slice
        self.move_th = movement_threshold_m
        self.run_th = run_threshold_frames
        self.prev_xyz_by_ep: dict[int, np.ndarray | None] = {}
        self.run_len_by_ep: dict[int, int] = {}
        self.long_runs_by_ep: dict[int, int] = {}

    def process_frame(self, frame_idx, episode_idx, sample) -> None:
        s = sample.get("observation.state")
        if s is None:
            return
        if isinstance(s, torch.Tensor):
            xyz = s[self.slice].detach().cpu().numpy().astype(np.float64)
        else:
            xyz = np.asarray(s[self.slice], dtype=np.float64)

        prev = self.prev_xyz_by_ep.get(episode_idx)
        if prev is not None:
            if np.linalg.norm(xyz - prev) < self.move_th:
                self.run_len_by_ep[episode_idx] = self.run_len_by_ep.get(episode_idx, 0) + 1
            else:
                # Run ended — check if it was long enough to count
                run = self.run_len_by_ep.get(episode_idx, 0)
                if run >= self.run_th:
                    self.long_runs_by_ep[episode_idx] = self.long_runs_by_ep.get(episode_idx, 0) + 1
                self.run_len_by_ep[episode_idx] = 0
        self.prev_xyz_by_ep[episode_idx] = xyz

    def finalize(self) -> CheckResult:
        # Close out any open run at end of dataset
        for ep, run in self.run_len_by_ep.items():
            if run >= self.run_th:
                self.long_runs_by_ep[ep] = self.long_runs_by_ep.get(ep, 0) + 1
        total = sum(self.long_runs_by_ep.values())
        if total == 0:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status=STATUS_PASS,
                summary=f"no stationary runs ≥ {self.run_th} frames",
            )
        per_ep = ", ".join(f"ep{ep}:{n}" for ep, n in sorted(self.long_runs_by_ep.items()))
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status=STATUS_WARN,  # INFO-severity but flagged for visibility
            summary=f"{total} stationary run(s) ≥ {self.run_th} frames ({per_ep})",
            n_violations=total,
            stats={"long_runs_by_ep": dict(self.long_runs_by_ep)},
        )


# ---------------------------------------------------------------------------
# Timestamp monotonicity check
# ---------------------------------------------------------------------------


class TimestampMonotonicityCheck(Check):
    """`timestamp` strictly increasing within each episode.

    Catches dropped frames, duplicate writes, or clock skew. Lerobot stores per-frame
    timestamps in seconds since episode start.
    """

    name = "timestamp_monotonicity"
    severity = SEVERITY_BLOCKER

    def __init__(self, max_details: int = 10):
        self.max_details = max_details
        self.last_ts_by_ep: dict[int, float] = {}
        self.n_violations = 0
        self.details: list[str] = []
        self.has_ts = False

    def setup(self, ds, ctx) -> bool:
        self.has_ts = "timestamp" in ds.features
        return self.has_ts

    def process_frame(self, frame_idx, episode_idx, sample) -> None:
        ts = sample.get("timestamp")
        if ts is None:
            return
        ts_v = _scalar(ts)
        last = self.last_ts_by_ep.get(episode_idx)
        if last is not None and ts_v <= last:
            self.n_violations += 1
            if len(self.details) < self.max_details:
                self.details.append(
                    f"frame {frame_idx} ep {episode_idx}: ts={ts_v:.4f} ≤ prev {last:.4f}"
                )
        self.last_ts_by_ep[episode_idx] = ts_v

    def finalize(self) -> CheckResult:
        if not self.has_ts:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status=STATUS_SKIP,
                summary="no timestamp column",
            )
        if self.n_violations == 0:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status=STATUS_PASS,
                summary=f"timestamps monotonic across {len(self.last_ts_by_ep)} episodes",
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status=STATUS_FAIL,
            summary=f"{self.n_violations} non-monotonic timestamp(s)",
            n_violations=self.n_violations,
            details=self.details,
        )


# ---------------------------------------------------------------------------
# Episode length check
# ---------------------------------------------------------------------------


class DatasetStatsEmitter(Check):
    """Emit empirically-correct `dataset_stats."observation.state".min/max` arrays.

    Not a pass/fail check — informational. Scans every frame of `observation.state` and
    reports per-dimension absolute min/max plus an optional percentile-clipped range
    that suppresses single-frame contact spikes. The resulting JSON-ready arrays should
    be pasted into `ur10_train_*.json`'s `policy.dataset_stats."observation.state"`.

    Why this exists: `find_limits` derives bounds while you free-jog the arm across the
    workspace, so its joint-velocity range reflects jogging dynamics rather than the
    quieter velocities of a task demo. The dataset_stats that drive the policy's MIN_MAX
    normalization should come from the *recorded demos* — the same data distribution the
    policy trains on. This emitter computes that.

    Reports two ranges:
      - **abs**: pure min/max (the `find_limits`-equivalent number, but on real demos).
      - **p{pct}**: percentile-clipped — uses the `pct/2`-th and `100 - pct/2`-th
        percentiles per dim. Default pct=99 gives the 0.5%-99.5% range, which clips
        single-frame contact transients while preserving meaningful task signal.

    Use the percentile range as your `dataset_stats` if the absolute range looks blown
    up by occasional contact spikes; use the absolute range otherwise.
    """

    name = "dataset_stats"
    severity = SEVERITY_INFO

    def __init__(self, percentile: float = 99.0):
        self.percentile = float(percentile)
        self.samples: list[np.ndarray] = []

    def process_frame(self, frame_idx, episode_idx, sample) -> None:
        s = sample.get("observation.state")
        if s is None:
            return
        if isinstance(s, torch.Tensor):
            s = s.detach().cpu().numpy()
        else:
            s = np.asarray(s)
        self.samples.append(s.astype(np.float64).flatten())

    def finalize(self) -> CheckResult:
        if not self.samples:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status=STATUS_SKIP,
                summary="no observation.state samples",
            )
        arr = np.stack(self.samples, axis=0)  # (N, D)
        abs_min = arr.min(axis=0)
        abs_max = arr.max(axis=0)
        pct_lo = np.percentile(arr, (100.0 - self.percentile) / 2.0, axis=0)
        pct_hi = np.percentile(arr, 100.0 - (100.0 - self.percentile) / 2.0, axis=0)

        def _fmt(v):
            return [round(float(x), 4) for x in v]

        details = [
            f"observation.state shape: {arr.shape[1]}-D over {arr.shape[0]} frames",
            "",
            "# Empirical absolute min/max (raw):",
            f'"min": {_fmt(abs_min)},',
            f'"max": {_fmt(abs_max)},',
            "",
            f"# {self.percentile:.1f}-percentile-clipped min/max (recommended for normalization):",
            f'"min": {_fmt(pct_lo)},',
            f'"max": {_fmt(pct_hi)},',
        ]
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status=STATUS_PASS,
            summary=f"dataset_stats over {arr.shape[0]} frames "
                    f"(absolute range + p{self.percentile} clipped). See details.",
            details=details,
            stats={
                "abs_min": abs_min.tolist(),
                "abs_max": abs_max.tolist(),
                "pct_min": pct_lo.tolist(),
                "pct_max": pct_hi.tolist(),
                "percentile": self.percentile,
                "n_frames": int(arr.shape[0]),
            },
        )


class EpisodeLengthCheck(Check):
    """Episode length distribution — sanity stats only (INFO).

    Reports min / max / mean episode lengths and flags episodes that finished in fewer
    than `min_frames` (likely operator pressed terminate too early) or more than
    `max_frames * fps` (time-limit truncation).
    """

    name = "episode_lengths"
    severity = SEVERITY_INFO

    def __init__(self, min_frames: int = 5, max_frames: int = 10_000):
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.lens: dict[int, int] = {}

    def process_frame(self, frame_idx, episode_idx, sample) -> None:
        self.lens[episode_idx] = self.lens.get(episode_idx, 0) + 1

    def finalize(self) -> CheckResult:
        if not self.lens:
            return CheckResult(
                name=self.name, severity=self.severity, status=STATUS_SKIP, summary="no frames",
            )
        ls = list(self.lens.values())
        too_short = [ep for ep, n in self.lens.items() if n < self.min_frames]
        too_long = [ep for ep, n in self.lens.items() if n > self.max_frames]
        summary = (
            f"{len(self.lens)} episodes; len min/mean/max = "
            f"{min(ls)}/{sum(ls)/len(ls):.1f}/{max(ls)}"
        )
        details = []
        if too_short:
            details.append(f"{len(too_short)} too-short episode(s): {sorted(too_short)}")
        if too_long:
            details.append(f"{len(too_long)} too-long episode(s): {sorted(too_long)}")
        status = STATUS_WARN if (too_short or too_long) else STATUS_PASS
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status=status,
            summary=summary,
            n_violations=len(too_short) + len(too_long),
            details=details,
            stats={"lens": dict(self.lens)},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_train_config(path: str | Path | None) -> dict | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"train_config not found: {p}")
    with p.open() as f:
        return json.load(f)
