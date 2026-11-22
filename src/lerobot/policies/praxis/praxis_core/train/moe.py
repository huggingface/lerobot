# Vendored from runwayml/praxis-1-oss at commit 8dad1a4, original path praxis/train/moe.py.
# Import paths are rewritten to the LeRobot package; the class PraxisPolicy of praxis.model
# is renamed PraxisModel here. Keep this file in sync with the upstream file.
"""Load balance of the mixture-of-experts layers during fine-tuning.

The router of each MoE layer picks ``top_k`` experts per token. Two terms keep
the load of the experts balanced. Both follow the original training code.

* The routing bias (``expert_bias``) changes after each optimizer step. Each
  layer counts how often the router picked each expert during the step, over
  all micro-steps and all ranks. The target count of an expert is
  ``tokens * top_k / num_experts``. The bias of an expert below the target
  goes up by ``speed``. The bias of an expert above the target goes down by
  ``speed``. The bias changes which experts the router picks, but not the gate
  weights. It has no gradient.
* The sequence balance loss adds a gradient to the router. For each layer and
  window: ``num_experts * sum_i f_i * p_i``. ``f_i`` is the fraction of the
  picks that went to expert ``i``. ``p_i`` is the mean of the normalized
  router scores of expert ``i`` over the tokens. The picks of this loss ignore
  the routing bias when the bias update is on. The loss of a step is the mean
  over the layers.

``track_routing`` installs the hook that records these values during the
forward pass. Activation checkpointing runs each block forward a second time
during the backward pass. The hook computes the same values in that pass, but
it records nothing.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor, nn

from lerobot.policies.praxis.praxis_core.layers import MoEFeedForward

__all__ = [
    "RoutingStats",
    "expert_counts",
    "sequence_balance_loss",
    "take_sequence_balance_losses",
    "track_routing",
    "update_expert_biases",
]


def expert_counts(top_idx: Tensor, num_experts: int) -> Tensor:
    """Return how often the router picked each expert, as float32 ``(num_experts,)``."""
    counts = torch.zeros(num_experts, device=top_idx.device, dtype=torch.float32)
    flat = top_idx.reshape(-1).to(torch.long)
    counts.scatter_add_(0, flat, torch.ones(flat.numel(), device=flat.device))
    return counts


def sequence_balance_loss(logits: Tensor, picks: Tensor) -> Tensor:
    """Return the sequence balance loss of one layer for the tokens of one window.

    ``logits`` holds the float32 router logits ``(N, E)``. ``picks`` holds the
    picked experts ``(N, k)``. The scores are the sigmoid of the logits,
    normalized to sum to one over the experts. Only ``logits`` gets a gradient.
    The divisions use a token count on the device of ``logits``. A CUDA
    division by a Python number multiplies by the reciprocal, which can
    change the last bit.
    """
    num_tokens, num_experts = logits.shape
    scores = torch.sigmoid(logits.float())
    probs = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    # A segmented sum with one segment, as in the original training code.
    rows = torch.zeros(num_tokens, dtype=torch.long, device=logits.device)
    p_sum = probs.new_zeros(1, num_experts).index_add_(0, rows, probs)
    counts = expert_counts(picks, num_experts)[None]
    tokens = torch.full((1, 1), float(num_tokens), device=logits.device)
    f = counts / (tokens * picks.shape[-1])
    p_mean = p_sum / tokens
    return num_experts * (f * p_mean).sum()


class RoutingStats:
    """The routing hook of one MoE layer during training (see ``track_routing``).

    In training mode each call adds the picks of the forward pass to
    ``counts`` and ``tokens``. With ``seq_balance`` the call also stores the
    sequence balance loss of the forward pass in ``loss``. With
    ``unbiased_picks`` the picks of that loss come from the logits without the
    routing bias. Activation checkpointing runs a block forward a second time
    (the recompute). In that run the call computes the same loss, so that the
    recompute saves the same tensors. But it records nothing.
    """

    def __init__(self, seq_balance: bool, unbiased_picks: bool) -> None:
        self.seq_balance = seq_balance
        self.unbiased_picks = unbiased_picks
        self.loss: Tensor | None = None
        self.counts: Tensor | None = None
        self.tokens = 0

    def __call__(self, layer: MoEFeedForward, logits: Tensor, top_idx: Tensor) -> None:
        if not layer.training:
            return
        first_pass = torch._C._current_graph_task_id() == -1
        if self.seq_balance:
            picks = top_idx
            if self.unbiased_picks:
                picks = torch.topk(logits.detach(), layer.top_k, dim=-1)[1]
            loss = sequence_balance_loss(logits, picks)
            if first_pass:
                self.loss = loss
        if first_pass:
            counts = expert_counts(top_idx, layer.num_experts)
            self.counts = counts if self.counts is None else self.counts + counts
            self.tokens += int(top_idx.shape[0])


def track_routing(model: nn.Module, seq_balance: bool, unbiased_picks: bool) -> list[MoEFeedForward]:
    """Install a ``RoutingStats`` hook on each MoE layer of ``model``.

    Return the MoE layers in module order (encoder before decoder, block
    order).
    """
    layers = [m for m in model.modules() if isinstance(m, MoEFeedForward)]
    for layer in layers:
        layer.routing_hook = RoutingStats(seq_balance, unbiased_picks)
    return layers


def _stats(layer: MoEFeedForward) -> RoutingStats | None:
    hook = layer.routing_hook
    return hook if isinstance(hook, RoutingStats) else None


def take_sequence_balance_losses(layers: list[MoEFeedForward]) -> list[Tensor]:
    """Return the stored sequence balance losses in layer order and clear them."""
    losses = []
    for layer in layers:
        stats = _stats(layer)
        if stats is not None and stats.loss is not None:
            losses.append(stats.loss)
            stats.loss = None
    return losses


@torch.no_grad()
def update_expert_biases(model: nn.Module, speed: float, group: dist.ProcessGroup | None = None) -> None:
    """Update the routing biases from the counts of the step, then clear the counts.

    Call this once per optimizer step on every rank. The counts of all
    micro-steps of the step add up. One all-reduce over ``group`` adds the
    counts of all ranks, so each rank applies the same update. A layer
    without tokens in the step keeps its bias. ``speed`` 0 only clears the
    counts.
    """
    pairs = [
        (m, st) for m in model.modules() if isinstance(m, MoEFeedForward) and (st := _stats(m)) is not None
    ]
    if not pairs:
        return
    device = pairs[0][0].expert_bias.device
    rows = []
    for layer, st in pairs:
        counts = st.counts
        if counts is None:
            counts = torch.zeros(layer.num_experts, device=device)
        tokens = torch.full((1,), float(st.tokens), device=device)
        rows.append(torch.cat([counts.to(device), tokens]))
        st.counts, st.tokens = None, 0
    if speed == 0.0:
        return
    stats = torch.stack(rows)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, group=group)
    for (layer, _), row in zip(pairs, stats, strict=True):
        counts, total = row[: layer.num_experts], row[layer.num_experts]
        if float(total) == 0.0:
            continue
        target = total * layer.top_k / layer.num_experts
        under = (counts < target).float()
        over = (counts > target).float()
        layer.expert_bias.add_((under - over) * speed)
