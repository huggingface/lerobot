# Adapted from NVlabs/LongLive at 1b9e348897cafa206fc6dd6feb80b59e2a8b1aa9.
# Changes: LeRobot package imports and reference-only runtime integration.
# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM research integration, adapted and renamed from the author branch.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: robocasa/FastWAM/src/fastwam/models/wan22/ar_wam.py
# Changes: Public Long-WAM interface; the FastWAM-derived base is attributed in longwam_base.py.
# Licensed under the Apache License, Version 2.0. See LICENSE at the repository root.
# License texts and provenance: see the Long-WAM _core/NOTICE.md.
# End Long-WAM attribution.

"""Long-WAM streaming observation memory with IDM or CodeDenoise prediction.

History is clean and causal. Future latents are generated, never supplied as
ground-truth observations at inference. Parameter names remain weight-compatible.
"""

import logging

from .longwam_base import LongWAMBase

logger = logging.getLogger(__name__)


class LongWAM(LongWAMBase):
    """LongLive 2.0 video expert with configurable history and future-latent prediction."""

    # Pixel `past_obs_size` -> latent clean frames. Each latent frame spans VAE_TEMPORAL_FACTOR (=4)
    # SAMPLED video frames AND `action_video_freq_ratio` raw frames per sampled frame, but the spec
    # fixes the bookkeeping directly: with the first v0b config (num_frames=33, video_sample_indices=
    # range(0,33,4) -> 9 sampled -> 3 latent, action_video_freq_ratio=4) one extra clean LATENT frame
    # corresponds to 16 raw frames of past obs. Hence M = past_obs_size // 16 + 1.
    LongWAM_RAW_FRAMES_PER_CLEAN_LATENT = 16

    def __init__(
        self,
        *args,
        longwam_num_frame_per_block: int = 1,
        longwam_local_attn_size: int = -1,
        longwam_sink_size: int = 0,
        longwam_past_obs_size: int = 0,
        longwam_num_imagine_frames: int = 0,
        longwam_imagine_sigma: float = 0.9,
        longwam_imagine_infer_steps: int = 4,
        longwam_joint_denoise: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Retained for the streaming-memory follow-up (v0b); unused in v0a.
        self.longwam_num_frame_per_block = int(longwam_num_frame_per_block)
        self.longwam_local_attn_size = int(longwam_local_attn_size)
        self.longwam_sink_size = int(longwam_sink_size)
        self._apply_longwam_past_obs_size(int(longwam_past_obs_size))
        self._apply_longwam_imagination(
            num_imagine_frames=int(longwam_num_imagine_frames),
            imagine_sigma=float(longwam_imagine_sigma),
            imagine_infer_steps=int(longwam_imagine_infer_steps),
            joint_denoise=bool(longwam_joint_denoise),
        )

    def _apply_longwam_imagination(
        self,
        num_imagine_frames: int,
        imagine_sigma: float,
        imagine_infer_steps: int,
        joint_denoise: bool,
    ) -> None:
        """Configure P4 joint imagination (k future latent frames the action attends).

        - num_imagine_frames (k): leading FUTURE latent frames the action additionally attends as
          imagination. k=0 -> action attends clean frames ONLY (byte-identical to v0a/v0b).
        - imagine_sigma in [0,1]: the HIGH/fixed noise level (fraction of full noise) the action reads
          the imagined future at (decoupled from the full-range video-loss timestep). Stored on LongWAMBase
          as `imagine_timestep` in scheduler timestep units (sigma * video_num_train_timesteps).
        - imagine_infer_steps: # partial-denoise video steps used at inference to roll the future to
          `imagine_sigma` (train==infer high-sigma target).
        - joint_denoise: jointly denoise future video and action. This mode uses a shared full
          schedule, so `imagine_sigma` must be zero.
        """
        self.longwam_num_imagine_frames = int(num_imagine_frames)
        self.longwam_joint_denoise = bool(joint_denoise)
        self.joint_denoise = self.longwam_joint_denoise
        if not (0.0 <= float(imagine_sigma) <= 1.0):
            raise ValueError(f"`longwam_imagine_sigma` must be in [0,1], got {imagine_sigma}")
        self.num_imagine_frames = int(num_imagine_frames)
        ntt = float(self.train_video_scheduler.num_train_timesteps)
        self.imagine_timestep = float(imagine_sigma) * ntt
        self.imagine_infer_steps = int(imagine_infer_steps)
        if self.imagine_infer_steps <= 0:
            raise ValueError("`longwam_imagine_infer_steps` must be positive.")
        if self.joint_denoise:
            if self.num_imagine_frames <= 0:
                raise ValueError("`longwam_joint_denoise=true` requires k>0 future frames.")
            if float(imagine_sigma) != 0.0:
                raise ValueError("Joint denoising must use `longwam_imagine_sigma=0.0`.")

    def _apply_longwam_past_obs_size(self, longwam_past_obs_size: int) -> None:
        """Configure v0b streaming clean-observation memory from the pixel `past_obs_size`.

        - num_clean_frames (M): leading clean latent frames = past_obs_size // 16 + 1.
          past_obs_size=0 -> M=1 (v0a, byte-identical). past_obs_size=16 -> M=2.
        - current_obs_idx: index of the CURRENT obs in the proprio time dim. Proprio is at RAW frame
          resolution (length num_frames-1, NOT subsampled by video_sample_indices), and the obs window
          is [t=-past_obs_size .. ], so the current obs (t=0) sits at index `past_obs_size`.
          past_obs_size=0 -> current_obs_idx=0 (v0a).
        """
        self.longwam_past_obs_size = int(longwam_past_obs_size)
        self.num_clean_frames = self.longwam_past_obs_size // self.LongWAM_RAW_FRAMES_PER_CLEAN_LATENT + 1
        self.current_obs_idx = self.longwam_past_obs_size

    @classmethod
    def from_wan22_pretrained(
        cls,
        *args,
        longwam_num_frame_per_block: int = 1,
        longwam_local_attn_size: int = -1,
        longwam_sink_size: int = 0,
        longwam_past_obs_size: int = 0,
        longwam_num_imagine_frames: int = 0,
        longwam_imagine_sigma: float = 0.9,
        longwam_imagine_infer_steps: int = 4,
        longwam_joint_denoise: bool = False,
        **kwargs,
    ):
        vc = kwargs.get("video_dit_config")
        if isinstance(vc, dict) and bool(vc.get("action_conditioned", False)):
            raise ValueError("LongWAM requires `video_dit_config['action_conditioned']=false`.")
        model = super().from_wan22_pretrained(*args, **kwargs)  # cls-bound -> returns LongWAM
        model.longwam_num_frame_per_block = int(longwam_num_frame_per_block)
        model.longwam_local_attn_size = int(longwam_local_attn_size)
        model.longwam_sink_size = int(longwam_sink_size)
        model._apply_longwam_past_obs_size(int(longwam_past_obs_size))
        model._apply_longwam_imagination(
            num_imagine_frames=int(longwam_num_imagine_frames),
            imagine_sigma=float(longwam_imagine_sigma),
            imagine_infer_steps=int(longwam_imagine_infer_steps),
            joint_denoise=bool(longwam_joint_denoise),
        )
        return model
