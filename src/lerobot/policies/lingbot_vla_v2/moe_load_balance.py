"""
Auxiliary-loss-free load balancing via optimizer pre-hook.

The hook is registered on the optimizer and executed automatically before
each ``optimizer.step()``, ensuring:
  - Counts are accumulated across all micro-steps (gradient accumulation safe).
  - All-reduce happens outside the compiled forward graph (torch.compile safe).
  - Easy to swap the reduce group for expert parallelism (EP ready).

Usage::

    from lingbotvla.models.vla.lingbot_vla.moe_load_balance import build_moe_load_balance_hook

    hook = build_moe_load_balance_hook(model, coeff=0.001)
    optimizer.register_step_pre_hook(hook)

Reference: lumos/moe/load_balance.py (VideoPretrain repo)
Paper: Auxiliary-Loss-Free Load Balancing Strategy for MoE (arXiv:2408.15664)
"""

import torch
import torch.distributed as dist
import logging

from .qwen2_action_expert import Qwen2TokenMoeBlock

logger = logging.getLogger(__name__)


def build_moe_load_balance_hook(
    model, coeff: float, group=None, bias_centering: bool = False, update_interval: int = 1
):
    """Build an optimizer pre-hook for aux-loss-free expert load balancing.

    Every ``update_interval`` optimizer steps the hook:

    1. Reads ``tokens_per_expert`` from every MoE block (accumulated over the
       last ``update_interval`` steps' tokens, since it is only zeroed here).
    2. All-reduces the counts across ranks in a single fused operation.
    3. Updates ``e_score_correction_bias`` using the sign of the
       deviation from the mean: bias -= coeff * sign(load - mean).
    4. (Optional) Centers the bias by subtracting its per-layer mean.
    5. Zeros out ``tokens_per_expert`` for the next accumulation window.

    On the in-between steps it does nothing, letting ``tokens_per_expert`` keep
    accumulating — this averages the load over a larger token sample so the
    sign(load-mean) direction is reliable. Critical when the global batch is
    small (per-step load is noisy → sign flips → bias random-walks near 0 and
    never learns a stable correction).

    Args:
        model: The model containing MoE blocks.
        coeff: Update step size (bias_update_speed), e.g. 0.001.
        group: Optional process group for all-reduce.
            Defaults to the world group. Pass the DP group when using EP.
        bias_centering: If True, subtract the per-layer bias mean after each
            update to pin sum(bias)=0 and prevent cumulative drift.
        update_interval: Apply the bias update once every N optimizer steps,
            accumulating load in between. 1 = update every step (original).

    Returns:
        A callable suitable for ``optimizer.register_step_pre_hook``.
    """
    update_interval = max(1, int(update_interval))

    # Collect all MoE block instances once at registration time
    moe_blocks = [m for m in model.modules() if isinstance(m, Qwen2TokenMoeBlock)]
    if not moe_blocks:
        logger.warning("build_moe_load_balance_hook: no MoE blocks found in model. The hook will be a no-op.")

    _state = {"step": 0}

    def _hook(optimizer, *args, **kwargs):
        if not moe_blocks:
            return

        _state["step"] += 1
        if _state["step"] % update_interval != 0:
            # Skip: keep accumulating tokens_per_expert across steps (the forward
            # adds to it; we only zero it on update steps below). Larger token
            # sample -> reliable sign(load-mean) -> bias learns a stable correction.
            return

        # Stack all tokens_per_expert into [num_layers, num_experts] and
        # do a single all-reduce instead of 36 sequential ones.
        all_tpe = torch.stack([b.tokens_per_expert for b in moe_blocks])

        if dist.is_initialized() and dist.get_world_size(group) > 1:
            dist.all_reduce(all_tpe, op=dist.ReduceOp.SUM, group=group)

        for i, block in enumerate(moe_blocks):
            tpe = all_tpe[i]

            # Snapshot global load for monitoring BEFORE zeroing.
            # The monitoring path (compute_loss) reads last_tokens_per_expert
            # to report the true biased, top_k, all-reduced expert load.
            if hasattr(block, "last_tokens_per_expert"):
                block.last_tokens_per_expert.copy_(tpe)

            # Sign-based update (DeepSeek-V3 style):
            # overloaded experts get bias decreased, underloaded get increased
            mean_load = tpe.float().mean()
            deviation = (tpe.float() - mean_load).sign()
            block.e_score_correction_bias.add_(-coeff * deviation)

            # Optional centering: pin sum(bias)=0 to prevent cumulative drift.
            # Routing-invariant (top-k of sigmoid+bias unchanged by a constant shift).
            if bias_centering:
                block.e_score_correction_bias.sub_(block.e_score_correction_bias.mean())

            # Reset accumulator for next optimizer step
            block.tokens_per_expert.zero_()

    return _hook
