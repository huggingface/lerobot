"""Correctness checkpoints for the exp0908 performance campaign (contract.md,
"Correctness Instrumentation Plan"). Every call returns on its first line unless
PROBE=1, so the timed run pays nothing.

Tags fold the step in. Steps count from 1, matching the training log.
"""

from __future__ import annotations

try:
    import probe
except ImportError:  # the recorder is not installed: every hook is a no-op
    probe = None

EARLY_STEPS = 5     # steps 1..5 are held tightly
DETAIL_STEPS = 3    # steps 1..3 also record batch and model outputs

_state = {"step": 0}


def enabled() -> bool:
    return probe is not None and probe.enabled()


def begin_step(step: int) -> None:
    """Called once per loop iteration with the 1-based step number."""
    _state["step"] = step


def step() -> int:
    return _state["step"]


def batch(batch: dict) -> None:
    """Shapes and dtypes of every tensor at steps 1..3, exact; action values, exact."""
    if not enabled() or step() > DETAIL_STEPS:
        return
    import torch

    s = step()
    for k in sorted(batch):
        v = batch[k]
        if isinstance(v, torch.Tensor):
            probe.record(f"batch_{k}_meta_step{s}", f"{tuple(v.shape)}/{v.dtype}/{v.device.type}")
    act = batch["action"].float()
    probe.record(f"batch_action_absmean_step{s}", act.abs().mean(), rtol=1e-6)
    probe.record(f"batch_action_sum_step{s}", act.sum(), rtol=1e-6)
    probe.record(f"batch_action_is_pad_sum_step{s}", int(batch["action_is_pad"].sum()))


def model_outputs(actions_hat, mu_hat, log_sigma_x2_hat) -> None:
    """ACT outputs at steps 1..3, at model precision."""
    if not enabled() or step() > DETAIL_STEPS:
        return
    s = step()
    probe.record_lazy(f"actions_hat_absmean_step{s}", lambda: actions_hat.float().abs().mean(), rtol=1e-3)
    probe.record_lazy(f"actions_hat_meta_step{s}", lambda: f"{tuple(actions_hat.shape)}/{actions_hat.dtype}")
    if mu_hat is not None:
        probe.record_lazy(f"mu_hat_absmean_step{s}", lambda: mu_hat.float().abs().mean(), rtol=1e-3)
        probe.record_lazy(f"log_sigma_x2_absmean_step{s}", lambda: log_sigma_x2_hat.float().abs().mean(), rtol=1e-3)


def step_metrics(loss: float, output_dict: dict | None, grad_norm: float | None) -> None:
    """Per-step scalars for every step. Tight for steps 1..5, looser after, where
    same-seed runs drift through cuDNN autotune; `probe derive` sets the truth."""
    if not enabled():
        return
    s = step()
    tol = {"atol": 1e-2} if s <= EARLY_STEPS else {"rtol": 3e-2, "atol": 1e-2}
    probe.record(f"loss_step{s}", float(loss), **tol)
    if grad_norm is not None:
        gtol = {"rtol": 1e-2} if s <= EARLY_STEPS else {"rtol": 1e-1}
        probe.record(f"grad_norm_step{s}", float(grad_norm), **gtol)
    for k in ("l1_loss", "kld_loss"):
        if output_dict and k in output_dict:
            probe.record(f"{k}_step{s}", float(output_dict[k]), **tol)


def flush() -> None:
    if enabled():
        probe.flush()
