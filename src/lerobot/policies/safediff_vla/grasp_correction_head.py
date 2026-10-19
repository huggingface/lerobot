"""`architecture="temporal_decoder_grounded_grasp_v2_5"` only, gated on
`SafeDiffVLAConfig.local_grasp_use_correction_head` (default `False` -- opt-in, so existing v2_5
checkpoints/configs are byte-for-byte unaffected unless this flag is explicitly turned on).

Motivated by the manual 18-episode visual audit (`examples/safediff_vla/local_grasp_correction_manual_review.py`
and the design note it feeds): `TargetPointHead`'s own `target_xyz` prediction shows a real,
non-noise, per-task-consistent spatial bias against the true grasp point in a subset of episodes
where the GROUNDED OBJECT IDENTITY was itself correct (the "A_FINE_GRASP_ERROR" class in that
review) -- i.e. `target_xyz` is in roughly the right place but not precisely on the graspable
point. This module adds ONE bounded correction on top of that estimate, LOCAL_GRASP-phase only,
consumed as extra context by `LocalGraspRefiner` (which itself stays completely unmodified --
see that module's own docstring) instead of `target_xyz`/`relative_xyz` directly:

    grounded_target_feature, image_feature, current_state, coarse_target_xyz, relative_xyz
        -> LocalGraspCorrectionHead -> bounded delta_grasp_xyz (meters)
        -> corrected_target_xyz = coarse_target_xyz(meters) + delta_grasp_xyz
        -> fed to `LocalGraspRefiner` in place of `target_xyz`/`relative_xyz`

Deliberately does NOT attempt to fix the "B_WRONG_OBJECT" class from that same review (wrong
object/instance entirely, sometimes just a few cm away in a cluttered scene -- see the review's
per-episode notes) -- that is a grounding/identity problem in `TargetQueryExtractor`/
`VisualGroundingCrossAttention`/`TargetPointHead`, all of which stay frozen and untouched here, and
a BOUNDED local correction cannot, and should not be trained to, jump to a different object.
`SafeDiffVLAConfig.local_grasp_max_grasp_correction_m` bounds this module's output at the
A_FINE_GRASP_ERROR class's own observed p90 residual (see the design note for the exact number and
its derivation) precisely so it cannot learn to attempt a B-class jump; the training loss
(`modeling_safediff_vla.py`'s `has_target_head_v2_5` branch) additionally masks out any individual
training frame whose own `gt_target_xyz - target_xyz` already exceeds
`local_grasp_correction_outlier_clip_m`, so a B-scale frame never contributes a gradient here
either.
"""

import torch
from torch import Tensor, nn


class LocalGraspCorrectionHead(nn.Module):
    """Predicts a single bounded spatial correction `delta_grasp_xyz` `[B, 3]` (meters, tanh-bounded
    by `max_grasp_correction_m`) on top of `TargetPointHead`'s own `coarse_target_xyz`. Unlike
    `LocalGraspRefiner`, this is NOT a temporal action-chunk decoder -- one correction per forward
    call, no `horizon` dimension -- since it corrects a single spatial anchor, not a trajectory.
    """

    def __init__(self, latent_dim: int, state_dim: int, hidden_dim: int, max_grasp_correction_m: float) -> None:
        super().__init__()
        if max_grasp_correction_m <= 0:
            raise ValueError("max_grasp_correction_m must be positive")
        self.max_grasp_correction_m = max_grasp_correction_m
        input_dim = 2 * latent_dim + state_dim + 3 + 3  # grounded + image feature, state, coarse target, relative
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        grounded_target_feature: Tensor,
        image_feature: Tensor,
        current_state: Tensor,
        coarse_target_xyz: Tensor,
        relative_xyz: Tensor,
    ) -> Tensor:
        """All args `[B, *]` batch-first (same convention as `LocalGraspRefiner.forward`).
        `coarse_target_xyz` is `TargetPointHead`'s own output (normalized action-position space);
        `relative_xyz` is `coarse_target_xyz(meters) - current_ee_xyz(meters)`, PHYSICAL METERS.
        Returns `delta_grasp_xyz` `[B, 3]`, meters, `|delta_grasp_xyz| < max_grasp_correction_m`
        element-wise (strict: `tanh` never saturates to exactly +/-1)."""
        x = torch.cat(
            (grounded_target_feature, image_feature, current_state, coarse_target_xyz, relative_xyz), dim=-1
        )
        raw = self.net(x)
        return self.max_grasp_correction_m * torch.tanh(raw)
