# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Release conversion preserves mode, cadence, and published denoising steps."""

import pytest
import yaml

from lerobot.policies.longwam.conversion_longwam import config_from_release


@pytest.mark.parametrize("mode,steps", [("idm", 10), ("cod", 4)])
def test_release_inference_settings_are_not_replaced_by_converter_defaults(tmp_path, mode, steps):
    common = {"num_layers": 30, "num_heads": 24, "attn_head_dim": 128, "text_dim": 4096, "freq_dim": 256}
    model = {
        "proprio_dim": 8,
        "arwam_past_obs_size": 48,
        "arwam_num_imagine_frames": 2,
        "arwam_joint_denoise": mode == "cod",
        "arwam_imagine_sigma": 0 if mode == "cod" else 0.9,
        "arwam_imagine_infer_steps": 4,
        "video_dit_config": {
            **common,
            "hidden_dim": 3072,
            "ffn_dim": 14336,
            "in_dim": 48,
            "out_dim": 48,
            "patch_size": [1, 2, 2],
            "video_attention_mask_mode": "per_frame_causal",
            "action_conditioned": False,
        },
        "action_dit_config": {**common, "hidden_dim": 1024, "ffn_dim": 4096, "action_dim": 7},
        "video_scheduler": {"train_shift": 5.0, "infer_shift": 5.0, "num_train_timesteps": 1000},
        "action_scheduler": {"train_shift": 5.0, "infer_shift": 5.0, "num_train_timesteps": 1000},
    }
    source = {
        "eval_num_inference_steps": steps,
        "model": model,
        "data": {
            "train": {
                "processor": {"norm_default_mode": "min/max"},
                "concat_multi_camera": "horizontal",
                "video_size": [224, 448],
                "shape_meta": {"images": [{"key": "image"}, {"key": "wrist_image"}]},
                "past_obs_size": 48,
                "num_frames": 81,
                "action_video_freq_ratio": 4,
                "action_chunk": 32,
                "context_len": 128,
            }
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(source))
    cfg = config_from_release(tmp_path)
    assert cfg.denoising == mode and cfg.num_inference_steps == steps
    assert cfg.imagine_sigma == model["arwam_imagine_sigma"]
    assert cfg.past_obs_steps == 48 and cfg.action_horizon == 32
    source["eval_num_inference_steps"] = 0
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(source))
    with pytest.raises(ValueError, match="positive"):
        config_from_release(tmp_path)
