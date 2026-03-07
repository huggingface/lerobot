from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from .pipeline import PolicyActionProcessorStep, ProcessorStepRegistry


def optimize_actions_qp(
    actions: torch.Tensor,
    dt: float = 1.0,
    vel_limits: tuple[float, float] | None = None,
    acc_limits: tuple[float, float] | None = None,
    w_data: float = 1.0,
    w_acc: float = 1.0,
    w_jerk: float = 0.0,
    fix_ends: bool = True,
    eps: float = 1e-6,
    verbose: bool = False,
):
    """Solve QP with smoothing, optional fixed endpoints, and optional velocity/acceleration constraints.

    Minimizes: 0.5 * w_data * ||x - a||^2 + 0.5 * w_acc * ||D2 x||^2 + 0.5 * w_jerk * ||D3 x||^2
    subject to:
        - if vel_limits is not None: vmin <= (x[i+1] - x[i])/dt <= vmax
        - if acc_limits is not None: amin <= (x[i+2] - 2x[i+1] + x[i])/dt^2 <= amax
        - if fix_ends: x[0] = a[0], x[-1] = a[-1]
    """
    try:
        import osqp
        from scipy import sparse
        import numpy as np
    except ImportError as e:
        raise ImportError("osqp and scipy are required") from e

    device = actions.device
    dtype = actions.dtype
    B, T, A_dim = actions.shape

    if verbose:
        print(f"QP with constraints: w_data={w_data}, w_acc={w_acc}, w_jerk={w_jerk}, fix_ends={fix_ends}")
        if vel_limits:
            print(f"  velocity limits: {vel_limits[0]:.3f} to {vel_limits[1]:.3f} (per sec)")
        if acc_limits:
            print(f"  acceleration limits: {acc_limits[0]:.3f} to {acc_limits[1]:.3f} (per sec^2)")

    # Build difference matrices
    I = sparse.eye(T, format='csc')
    if T >= 3:
        D2 = sparse.diags([1, -2, 1], [0, 1, 2], shape=(T-2, T), format='csc')
    else:
        D2 = sparse.csc_matrix((0, T), dtype=float)
    if T >= 4:
        D3 = sparse.diags([1, -3, 3, -1], [0, 1, 2, 3], shape=(T-3, T), format='csc')
    else:
        D3 = sparse.csc_matrix((0, T), dtype=float)

    # Hessian
    P = w_data * I
    if w_acc != 0 and T >= 3:
        P = P + w_acc * (D2.T @ D2)
    if w_jerk != 0 and T >= 4:
        P = P + w_jerk * (D3.T @ D3)
    P = P + eps * I  # ensure positive definite
    P = P.astype(np.float64)

    out = torch.empty_like(actions, dtype=dtype)
    N = B * A_dim
    a_flat = actions.permute(0, 2, 1).contiguous().view(N, T).cpu().numpy()

    # Precompute raw difference matrices (without dt scaling) for constraints
    A_vel_raw = None
    if vel_limits is not None and T >= 2:
        A_vel_raw = sparse.diags([-1, 1], [0, 1], shape=(T-1, T), format='csc')
    A_acc_raw = None
    if acc_limits is not None and T >= 3:
        A_acc_raw = sparse.diags([1, -2, 1], [0, 1, 2], shape=(T-2, T), format='csc')

    for n in range(N):
        an = a_flat[n].astype(np.float64)
        if verbose:
            print(f"\n=== Trajectory {n}/{N} ===")
            print(f"  Original range: [{an.min():.4f}, {an.max():.4f}]")
            print(f"  First/last: {an[0]:.4f}, {an[-1]:.4f}")

        q = (-w_data * an).astype(np.float64)

        # Build constraint matrix A and bounds l,u
        A_list = []
        l_list = []
        u_list = []

        if fix_ends:
            A_eq = sparse.vstack([
                sparse.eye(1, T, format='csc'),
                sparse.eye(1, T, format='csc', k=T-1)
            ])
            A_list.append(A_eq)
            l_list.extend([an[0], an[-1]])
            u_list.extend([an[0], an[-1]])

        if A_vel_raw is not None:
            vmin, vmax = vel_limits
            # (x[i+1]-x[i])/dt ∈ [vmin, vmax] → (x[i+1]-x[i]) ∈ [vmin*dt, vmax*dt]
            A_vel_scaled = A_vel_raw / dt
            A_list.append(A_vel_scaled)
            l_list.extend([vmin * dt] * (T-1))
            u_list.extend([vmax * dt] * (T-1))

        if A_acc_raw is not None:
            amin, amax = acc_limits
            # (x[i+2]-2x[i+1]+x[i])/dt^2 ∈ [amin, amax] → raw second diff ∈ [amin*dt^2, amax*dt^2]
            A_acc_scaled = A_acc_raw / (dt**2)
            A_list.append(A_acc_scaled)
            l_list.extend([amin * (dt**2)] * (T-2))
            u_list.extend([amax * (dt**2)] * (T-2))

        if len(A_list) > 1:
            A_constraint = sparse.vstack(A_list).tocsc()
        else:
            A_constraint = A_list[0] if A_list else sparse.csc_matrix((0, T), dtype=np.float64)
        l = np.array(l_list, dtype=np.float64) if l_list else np.array([], dtype=np.float64)
        u = np.array(u_list, dtype=np.float64) if u_list else np.array([], dtype=np.float64)

        if verbose and A_constraint.shape[0] > 0:
            print(f"  Constraint matrix shape: {A_constraint.shape}")
            print(f"  First few l: {l[:4]}, first few u: {u[:4]}")

        # Solveoptimize_actions_qp
        prob = osqp.OSQP()
        prob.setup(P=P, q=q, A=A_constraint, l=l, u=u, verbose=False,
                   polish=False, eps_abs=1e-4, eps_rel=1e-4, max_iter=100)
        res = prob.solve()

        status = res.info.status
        if verbose:
            print(f"  OSQP status: {status}")

        if status in ('solved', 'solved_inaccurate', 'solved_relaxed') and res.x is not None:
            xn = res.x
            if verbose:
                print(f"  Solution range: [{xn.min():.4f}, {xn.max():.4f}]")
                print(f"  First few values: {xn[:8]}")

                # Compute smoothness metrics
                dx = np.diff(xn)
                d2x = np.diff(dx)
                dx_orig = np.diff(an)
                d2x_orig = np.diff(dx_orig)

                print(f"  Mean |dx| (smoothed): {np.mean(np.abs(dx)):.4f}, original: {np.mean(np.abs(dx_orig)):.4f}")
                print(f"  Mean |d2x| (smoothed): {np.mean(np.abs(d2x)):.4f}, original: {np.mean(np.abs(d2x_orig)):.4f}")
                print(f"  Max |dx| (smoothed): {np.max(np.abs(dx)):.4f}, original: {np.max(np.abs(dx_orig)):.4f}")
        else:
            if verbose:
                print("  OSQP failed, falling back to original.")
            xn = an

        # Write to output
        b = n // A_dim
        a = n % A_dim
        out[b, :, a] = torch.from_numpy(xn).to(dtype)

    return out.to(device)


@ProcessorStepRegistry.register(name="mpc_action_smoothing_processor")
class MPCActionSmoothingProcessor(PolicyActionProcessorStep):
    """Processor step that applies QP-based smoothing to policy action tensors.

    This wraps `optimize_actions_qp` 
    The processor expects the action to be a tensor of shape (B, T, A).
    """

    def __init__(
        self,
        dt: float = 1.0,
        vel_limits: Tuple[float, float] | None = None,
        acc_limits: Tuple[float, float] | None = None,
        w_data: float = 1.0,
        w_acc: float = 1.0,
        w_jerk: float = 0.0,
        fix_ends: bool = True,
        eps: float = 1e-6,
        verbose: bool = False,
        fps: float | None = None,
    ):
        # If fps is provided, override dt
        self.dt = 1.0 / float(fps) if fps is not None else float(dt)
        self.vel_limits = tuple(vel_limits) if vel_limits is not None else None
        self.acc_limits = tuple(acc_limits) if acc_limits is not None else None
        self.w_data = float(w_data)
        self.w_acc = float(w_acc)
        self.w_jerk = float(w_jerk)
        self.fix_ends = bool(fix_ends)
        self.eps = float(eps)
        self.verbose = bool(verbose)

    def get_config(self) -> Dict[str, Any]:
        return {
            "dt": self.dt,
            "vel_limits": list(self.vel_limits) if self.vel_limits is not None else None,
            "acc_limits": list(self.acc_limits) if self.acc_limits is not None else None,
            "w_data": self.w_data,
            "w_acc": self.w_acc,
            "w_jerk": self.w_jerk,
            "fix_ends": self.fix_ends,
            "eps": self.eps,
            "verbose": self.verbose,
        }

    def state_dict(self) -> dict[str, Tensor]:
        # Stateless processor (no tensors to save)
        return {}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:  # pragma: no cover - trivial
        return None

    def transform_features(self, features: dict) -> dict:
        # This processor doesn't change the action feature shape or dtype.
        return features

    def action(self, action: Tensor) -> Tensor:
        # Expect a tensor of shape (B, T, A)
        if not isinstance(action, torch.Tensor):
            raise ValueError("QPActionSmoothingProcessor expects a torch.Tensor as action")

        orig_dtype = action.dtype
        device = action.device

        # Convert to float32 for solver stability
        action_fp = action.to(dtype=torch.float32)

        try:
            smoothed = optimize_actions_qp(
                action_fp,
                dt=self.dt,
                vel_limits=self.vel_limits,
                acc_limits=self.acc_limits,
                w_data=self.w_data,
                w_acc=self.w_acc,
                w_jerk=self.w_jerk,
                fix_ends=self.fix_ends,
                eps=self.eps,
                verbose=self.verbose,
            )
        except Exception as e:
            # On any solver error, fallback to original actions but keep device/dtype
            if self.verbose:
                print(f"QPActionSmoothingProcessor: solver failed ({e}), returning original action")
            return action

        # Ensure result has same dtype/device as input
        return smoothed.to(device=device).to(dtype=orig_dtype)
