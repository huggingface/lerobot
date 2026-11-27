# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Adversarial and regression tests for env/policy visual resolution mismatch warning in lerobot-eval."""

import logging
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import draccus
import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.configs.default import EvalConfig
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.scripts import lerobot_eval
from lerobot.scripts.lerobot_eval import _warn_on_visual_resolution_mismatch


def _dummy_eval_pipeline_config(env, policy_input_features, output_dir=None, rename_map=None):
    policy_cfg = SimpleNamespace(
        type="smolvla",
        device="cpu",
        use_amp=False,
        pretrained_path=None,
        input_features=policy_input_features,
    )
    cfg = EvalPipelineConfig(
        env=env,
        eval=EvalConfig(batch_size=1, n_episodes=1, use_async_envs=False),
        policy=policy_cfg,
        output_dir=output_dir,
        rename_map=rename_map or {},
    )
    return cfg


def _mock_eval_pipeline(monkeypatch):
    monkeypatch.setattr(lerobot_eval, "make_env", MagicMock(return_value={"libero": {0: MagicMock()}}))
    mock_policy = MagicMock()
    mock_policy.training = False
    mock_policy.config.device = "cpu"
    monkeypatch.setattr(lerobot_eval, "make_policy", MagicMock(return_value=mock_policy))
    monkeypatch.setattr(lerobot_eval, "make_pre_post_processors", MagicMock(return_value=(None, None)))
    monkeypatch.setattr(lerobot_eval, "make_env_pre_post_processors", MagicMock(return_value=(None, None)))
    monkeypatch.setattr(
        lerobot_eval,
        "eval_policy_all",
        MagicMock(
            return_value={
                "overall": {
                    "pc_success": 0.0,
                    "n_success": 0,
                    "n_episodes": 1,
                    "pc_success_ci95": (0.0, 0.0),
                    "video_paths": [],
                    "predicted_video_paths": [],
                },
                "per_task": [],
            }
        ),
    )


def test_eval_main_warns_on_libero_resolution_mismatch(caplog, monkeypatch, tmp_path):
    """Behavioral regression test: eval_main emits exactly one actionable warning when default LiberoEnv meets 256."""
    _mock_eval_pipeline(monkeypatch)

    env = LiberoEnv()  # defaults to 360x360
    policy_input_features = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    cfg = _dummy_eval_pipeline_config(env, policy_input_features, output_dir=tmp_path)

    with caplog.at_level(logging.WARNING):
        lerobot_eval.eval_main(cfg)

    mismatch_warnings = [
        r.message
        for r in caplog.records
        if "resolution" in r.message.lower()
        and (
            "differ" in r.message.lower()
            or "do not match" in r.message.lower()
            or "mismatch" in r.message.lower()
        )
    ]
    assert len(mismatch_warnings) == 1, (
        f"Expected exactly one resolution mismatch warning, got {len(mismatch_warnings)}"
    )
    w = mismatch_warnings[0]
    assert "360x360" in w
    assert "256x256" in w
    assert "--env.observation_height=256" in w
    assert "--env.observation_width=256" in w


def test_eval_main_no_warning_when_env_explicitly_set_to_256(caplog, monkeypatch, tmp_path):
    """Integration: no resolution warning when LiberoEnv is explicitly configured to 256x256."""
    _mock_eval_pipeline(monkeypatch)

    env = LiberoEnv(observation_height=256, observation_width=256)
    policy_input_features = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
    }
    cfg = _dummy_eval_pipeline_config(env, policy_input_features, output_dir=tmp_path)

    with caplog.at_level(logging.WARNING):
        lerobot_eval.eval_main(cfg)

    mismatch_warnings = [
        r.message
        for r in caplog.records
        if "resolution" in r.message.lower()
        and (
            "differ" in r.message.lower()
            or "do not match" in r.message.lower()
            or "mismatch" in r.message.lower()
        )
    ]
    assert len(mismatch_warnings) == 0


@pytest.mark.parametrize(
    "env_shape, policy_shape, should_warn, expected_env_str, expected_pol_str",
    [
        # Non-square matching: env (128, 192, 3) vs policy (3, 128, 192) => match
        ((128, 192, 3), (3, 128, 192), False, None, None),
        # Non-square swapped axes: env (192, 128, 3) vs policy (3, 128, 192) => mismatch
        ((192, 128, 3), (3, 128, 192), True, "192x128", "128x192"),
        # Swapped dimensions: env (360, 256, 3) vs policy (3, 256, 360) => mismatch
        ((360, 256, 3), (3, 256, 360), True, "360x256", "256x360"),
    ],
)
def test_non_square_and_swapped_axes(
    caplog, env_shape, policy_shape, should_warn, expected_env_str, expected_pol_str
):
    env_cfg = SimpleNamespace(
        features={"pixels": PolicyFeature(type=FeatureType.VISUAL, shape=env_shape)},
        features_map={"pixels": "observation.images.image"},
    )
    pol_cfg = SimpleNamespace(
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=policy_shape)
        }
    )

    with caplog.at_level(logging.WARNING):
        _warn_on_visual_resolution_mismatch(pol_cfg, env_cfg)

    warnings = [r.message for r in caplog.records if "resolution" in r.message.lower()]
    if should_warn:
        assert len(warnings) == 1
        assert expected_env_str in warnings[0]
        assert expected_pol_str in warnings[0]
    else:
        assert len(warnings) == 0


def test_two_cameras_swapped_shapes_detected(caplog):
    """Adversarial check: camera A and B have swapped shapes (same set of resolutions).

    Set-based difference (env_resolutions - policy_resolutions) yields empty set (false negative).
    Key-based camera identity check MUST catch and warn on both camera mismatches.
    """
    env_cfg = SimpleNamespace(
        features={
            "pixels/cam1": PolicyFeature(type=FeatureType.VISUAL, shape=(256, 128, 3)),
            "pixels/cam2": PolicyFeature(type=FeatureType.VISUAL, shape=(128, 256, 3)),
        },
        features_map={
            "pixels/cam1": "observation.images.cam1",
            "pixels/cam2": "observation.images.cam2",
        },
    )
    pol_cfg = SimpleNamespace(
        input_features={
            "observation.images.cam1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 256)),
            "observation.images.cam2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 128)),
        }
    )

    with caplog.at_level(logging.WARNING):
        _warn_on_visual_resolution_mismatch(pol_cfg, env_cfg)

    warnings = [r.message for r in caplog.records if "resolution" in r.message.lower()]
    assert len(warnings) == 1
    assert "observation.images.cam1" in warnings[0]
    assert "observation.images.cam2" in warnings[0]


def test_two_cameras_duplicate_env_resolutions_detected(caplog):
    """Adversarial check: env has two 256x256 cameras, but policy expects one 256x256 and one 128x128.

    Set subtraction {(256,256)} - {(256,256), (128,128)} == set() produces false negative in set-based checker.
    Key-based camera identity check MUST catch the 128x128 camera mismatch.
    """
    env_cfg = SimpleNamespace(
        features={
            "pixels/cam1": PolicyFeature(type=FeatureType.VISUAL, shape=(256, 256, 3)),
            "pixels/cam2": PolicyFeature(type=FeatureType.VISUAL, shape=(256, 256, 3)),
        },
        features_map={
            "pixels/cam1": "observation.images.cam1",
            "pixels/cam2": "observation.images.cam2",
        },
    )
    pol_cfg = SimpleNamespace(
        input_features={
            "observation.images.cam1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.images.cam2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
        }
    )

    with caplog.at_level(logging.WARNING):
        _warn_on_visual_resolution_mismatch(pol_cfg, env_cfg)

    warnings = [r.message for r in caplog.records if "resolution" in r.message.lower()]
    assert len(warnings) == 1
    assert "observation.images.cam2" in warnings[0]
    assert "128x128" in warnings[0]


def test_two_cameras_one_mismatched_identifies_camera(caplog):
    """Only the mismatched camera is reported when one matches and one does not."""
    env_cfg = SimpleNamespace(
        features={
            "pixels/cam1": PolicyFeature(type=FeatureType.VISUAL, shape=(256, 256, 3)),
            "pixels/cam2": PolicyFeature(type=FeatureType.VISUAL, shape=(360, 360, 3)),
        },
        features_map={
            "pixels/cam1": "observation.images.cam1",
            "pixels/cam2": "observation.images.cam2",
        },
    )
    pol_cfg = SimpleNamespace(
        input_features={
            "observation.images.cam1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.images.cam2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        }
    )

    with caplog.at_level(logging.WARNING):
        _warn_on_visual_resolution_mismatch(pol_cfg, env_cfg)

    warnings = [r.message for r in caplog.records if "resolution" in r.message.lower()]
    assert len(warnings) == 1
    assert "cam2" in warnings[0]
    assert "camera 'observation.images.cam1'" not in warnings[0]


def test_features_map_and_rename_map_integration(caplog):
    """Verifies that features_map and rename_map properly align camera keys across LIBERO and SmolVLA."""
    # LIBERO env produces observation.images.image and observation.images.image2
    env = LiberoEnv()
    # lerobot/smolvla_libero expects observation.images.camera1 and observation.images.camera2
    pol_cfg = SimpleNamespace(
        input_features={
            "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.images.camera2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        }
    )
    rename_map = {
        "observation.images.image": "observation.images.camera1",
        "observation.images.image2": "observation.images.camera2",
    }

    # 1. With rename_map, keys align; resolution 360 vs 256 warns
    with caplog.at_level(logging.WARNING):
        _warn_on_visual_resolution_mismatch(pol_cfg, env, rename_map=rename_map)
    assert any("observation.images.camera1" in r.message for r in caplog.records)

    # 2. When env is set to 256x256, rename_map aligns keys and matching resolution prevents warning
    caplog.clear()
    env_256 = LiberoEnv(observation_height=256, observation_width=256)
    with caplog.at_level(logging.WARNING):
        _warn_on_visual_resolution_mismatch(pol_cfg, env_256, rename_map=rename_map)
    assert not any("resolution" in r.message.lower() for r in caplog.records)


def test_custom_camera_name_mapping(caplog):
    """Verifies that LiberoEnv custom camera_name_mapping is respected."""
    env = LiberoEnv(camera_name_mapping={"agentview_image": "front", "robot0_eye_in_hand_image": "wrist"})
    pol_cfg = SimpleNamespace(
        input_features={
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        }
    )
    with caplog.at_level(logging.WARNING):
        _warn_on_visual_resolution_mismatch(pol_cfg, env)
    assert any("observation.images.front" in r.message for r in caplog.records)


def test_unmapped_camera_keys_fallback(caplog):
    """When camera keys cannot be aligned and resolutions differ, report unresolved identity and available resolution sets."""
    env_cfg = SimpleNamespace(
        features={"pixels/unknown": PolicyFeature(type=FeatureType.VISUAL, shape=(360, 360, 3))},
        features_map={},
    )
    pol_cfg = SimpleNamespace(
        input_features={
            "observation.images.other": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
        }
    )
    with caplog.at_level(logging.WARNING):
        _warn_on_visual_resolution_mismatch(pol_cfg, env_cfg)
    assert any(
        "360x360" in r.message and "256x256" in r.message and "unresolved" in r.message.lower()
        for r in caplog.records
    )


def test_unmapped_camera_does_not_manufacture_false_self_mismatch(caplog):
    """Regression test for Defect 1:
    When camera keys cannot be aligned, do NOT manufacture synthetic per-camera mismatch records
    (e.g., claiming a mismatch between 128x128 and 128x128) and do NOT choose an arbitrary CLI recommendation.
    """
    env_cfg = SimpleNamespace(
        features={"pixels/unknown": PolicyFeature(type=FeatureType.VISUAL, shape=(128, 128, 3))},
        features_map={},
        observation_height=128,
        observation_width=128,
    )
    pol_cfg = SimpleNamespace(
        input_features={
            "observation.images.cam1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
            "observation.images.cam2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        }
    )
    with caplog.at_level(logging.WARNING):
        _warn_on_visual_resolution_mismatch(pol_cfg, env_cfg)

    warnings = [r.message for r in caplog.records if "unresolved" in r.message.lower()]
    assert len(warnings) == 1
    w = warnings[0]
    # 1. Must report unresolved identity
    assert "camera identity is unresolved" in w.lower()
    # 2. Must report aggregate resolution sets
    assert "128x128" in w
    assert "256x256" in w
    # 3. Must NOT manufacture a contradictory per-camera comparison like "128x128 vs ... 128x128"
    assert "128x128 vs declared policy input-feature resolution 128x128" not in w
    assert "rendered 128x128 vs" not in w
    # 4. Must NOT suggest a specific resolution when correspondence is unknown
    assert "--env.observation_height" not in w


def test_features_map_transformed_key_not_renamed_by_raw_key_in_rename_map(caplog):
    """Regression test for Defect 2:
    Match policies/factory.py transformation order: env_to_policy_features (features_map)
    is applied first, then rename_map is applied to those mapped keys.
    If features_map transforms 'raw_cam' -> 'observation.images.intermediate', a rename_map
    entry targeting only the original 'raw_cam' is NOT applied by factory.py and must NOT
    be applied by the diagnostic checker.
    """
    env_cfg = SimpleNamespace(
        features={"raw_cam": PolicyFeature(type=FeatureType.VISUAL, shape=(360, 360, 3))},
        features_map={"raw_cam": "observation.images.intermediate"},
    )
    pol_cfg = SimpleNamespace(
        input_features={
            "observation.images.policy_cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
        }
    )
    # rename_map targeting the raw key that was already transformed by features_map
    rename_map = {"raw_cam": "observation.images.policy_cam"}
    with caplog.at_level(logging.WARNING):
        _warn_on_visual_resolution_mismatch(pol_cfg, env_cfg, rename_map=rename_map)

    # Because factory.py applies rename_map only to mapped keys ('observation.images.intermediate'),
    # 'raw_cam' is not renamed to 'observation.images.policy_cam'. Camera identity remains unresolved.
    warnings = [r.message for r in caplog.records if "unresolved" in r.message.lower()]
    assert len(warnings) == 1
    assert "camera identity is unresolved" in warnings[0].lower()
    # It must NOT claim that camera 'observation.images.policy_cam' was aligned
    assert "camera 'observation.images.policy_cam'" not in warnings[0]

    # Conversely, when rename_map targets the mapped key ('observation.images.intermediate'),
    # the keys align properly and a per-camera resolution mismatch is diagnosed.
    caplog.clear()
    rename_map_correct = {"observation.images.intermediate": "observation.images.policy_cam"}
    with caplog.at_level(logging.WARNING):
        _warn_on_visual_resolution_mismatch(pol_cfg, env_cfg, rename_map=rename_map_correct)
    warnings = [r.message for r in caplog.records if "resolution" in r.message.lower()]
    assert len(warnings) == 1
    assert "camera 'observation.images.policy_cam'" in warnings[0]


def test_unmapped_camera_keys_matching_resolutions_diagnosed_as_unresolved(caplog):
    """Conservative behavior: do not assume cameras correspond merely because resolution sets match."""
    env_cfg = SimpleNamespace(
        features={"pixels/unknown": PolicyFeature(type=FeatureType.VISUAL, shape=(256, 256, 3))},
        features_map={},
    )
    pol_cfg = SimpleNamespace(
        input_features={
            "observation.images.other": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
        }
    )
    with caplog.at_level(logging.WARNING):
        _warn_on_visual_resolution_mismatch(pol_cfg, env_cfg)
    warnings = [r.message for r in caplog.records if "unresolved" in r.message.lower()]
    assert len(warnings) == 1
    assert "camera identity is unresolved" in warnings[0].lower()


def test_conflicting_camera_mapping_detected(caplog):
    """Multiple environment cameras mapping to the same effective policy key must trigger a warning."""
    env_cfg = SimpleNamespace(
        features={
            "pixels/cam1": PolicyFeature(type=FeatureType.VISUAL, shape=(256, 256, 3)),
            "pixels/cam2": PolicyFeature(type=FeatureType.VISUAL, shape=(256, 256, 3)),
        },
        features_map={
            "pixels/cam1": "observation.images.cam",
            "pixels/cam2": "observation.images.cam",
        },
    )
    pol_cfg = SimpleNamespace(
        input_features={"observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))}
    )
    with caplog.at_level(logging.WARNING):
        _warn_on_visual_resolution_mismatch(pol_cfg, env_cfg)
    assert any("conflicting camera mapping" in r.message.lower() for r in caplog.records)


def test_partial_camera_mapping_detected(caplog):
    """When some cameras are mapped and others remain unmapped, report mapped mismatch and partial mapping."""
    env_cfg = SimpleNamespace(
        features={
            "pixels/cam1": PolicyFeature(type=FeatureType.VISUAL, shape=(360, 360, 3)),
            "pixels/cam_extra": PolicyFeature(type=FeatureType.VISUAL, shape=(256, 256, 3)),
        },
        features_map={
            "pixels/cam1": "observation.images.cam1",
        },
    )
    pol_cfg = SimpleNamespace(
        input_features={
            "observation.images.cam1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.images.cam2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        }
    )
    with caplog.at_level(logging.WARNING):
        _warn_on_visual_resolution_mismatch(pol_cfg, env_cfg)
    assert any("partial camera mapping" in r.message.lower() for r in caplog.records)
    assert any("360x360" in r.message and "256x256" in r.message for r in caplog.records)


def test_rename_map_direct_on_raw_env_key(caplog):
    """rename_map can directly map raw environment keys even if features_map did not map them."""
    env_cfg = SimpleNamespace(
        features={"raw_cam": PolicyFeature(type=FeatureType.VISUAL, shape=(360, 360, 3))},
        features_map={},
    )
    pol_cfg = SimpleNamespace(
        input_features={
            "observation.images.target": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
        }
    )
    # Direct raw key rename
    rename_map = {"raw_cam": "observation.images.target"}
    with caplog.at_level(logging.WARNING):
        _warn_on_visual_resolution_mismatch(pol_cfg, env_cfg, rename_map=rename_map)
    assert any("360x360" in r.message and "256x256" in r.message for r in caplog.records)


def test_resizing_policy_diagnostic_wording(caplog):
    """Verify diagnostic wording explains domain shift and clarifies intentional resizing policies."""
    env = LiberoEnv()
    pol_cfg = SimpleNamespace(
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        }
    )
    with caplog.at_level(logging.WARNING):
        _warn_on_visual_resolution_mismatch(pol_cfg, env)
    warnings = [r.message for r in caplog.records if "resolution" in r.message.lower()]
    assert len(warnings) == 1
    w = warnings[0]
    assert "visual domain shift" in w
    assert "scale or aspect-ratio" in w
    assert "intentionally resizes or crops" in w
    assert "declared policy input-feature resolution" in w


def test_robustness_missing_and_none_features(caplog):
    """Missing or None features must be gracefully handled with no warnings and no crashes."""
    with caplog.at_level(logging.WARNING):
        _warn_on_visual_resolution_mismatch(None, None)
        _warn_on_visual_resolution_mismatch(
            SimpleNamespace(input_features=None), SimpleNamespace(features=None)
        )
        _warn_on_visual_resolution_mismatch(SimpleNamespace(input_features={}), SimpleNamespace(features={}))
        _warn_on_visual_resolution_mismatch(
            SimpleNamespace(
                input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,))}
            ),
            SimpleNamespace(features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}),
        )
    assert len(caplog.records) == 0


def test_robustness_malformed_shapes_and_types(caplog):
    """Malformed shapes (e.g. 1D, negative dims, 0 dims) must not crash."""
    env_cfg = SimpleNamespace(
        features={
            "pixels/bad1": PolicyFeature(type=FeatureType.VISUAL, shape=(0, 256, 3)),
            "pixels/bad2": PolicyFeature(type=FeatureType.VISUAL, shape=(-1, 256, 3)),
            "pixels/bad3": PolicyFeature(type=FeatureType.VISUAL, shape=(256,)),
        },
        features_map={},
    )
    pol_cfg = SimpleNamespace(
        input_features={
            "observation.images.bad1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 0, 256)),
            "observation.images.bad2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256)),
        }
    )
    with caplog.at_level(logging.WARNING):
        _warn_on_visual_resolution_mismatch(pol_cfg, env_cfg)
    # Graceful exit without exception
    assert True


def test_no_suggestion_if_env_lacks_observation_height(caplog):
    """If environment does not support observation_height/width, do not invent flags in suggestion."""
    env_cfg = SimpleNamespace(
        features={"pixels": PolicyFeature(type=FeatureType.VISUAL, shape=(100, 100, 3))},
        features_map={"pixels": "observation.images.image"},
    )
    pol_cfg = SimpleNamespace(
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 200, 200))
        }
    )
    with caplog.at_level(logging.WARNING):
        _warn_on_visual_resolution_mismatch(pol_cfg, env_cfg)

    warnings = [r.message for r in caplog.records if "resolution" in r.message.lower()]
    assert len(warnings) == 1
    assert "--env.observation_height" not in warnings[0]


def test_draccus_cli_flags_syntax():
    """Verify that the suggested flags --observation_height=256 --observation_width=256 are valid in draccus."""
    cfg = draccus.parse(
        LiberoEnv,
        args=["--task=libero_10", "--observation_height=256", "--observation_width=256"],
    )
    assert cfg.observation_height == 256
    assert cfg.observation_width == 256
    assert cfg.gym_kwargs["observation_height"] == 256
    assert cfg.gym_kwargs["observation_width"] == 256


def test_namespaced_eval_pipeline_config_parsing():
    """Finding 4: Verify full namespaced lerobot-eval CLI parsing via EvalPipelineConfig."""
    from lerobot.utils.constants import LIBERO_KEY_PIXELS_AGENTVIEW, LIBERO_KEY_PIXELS_EYE_IN_HAND

    cfg = draccus.parse(
        EvalPipelineConfig,
        args=[
            "--env.type=libero",
            "--env.task=libero_10",
            "--env.observation_height=256",
            "--env.observation_width=256",
        ],
    )
    assert isinstance(cfg.env, LiberoEnv)
    assert cfg.env.observation_height == 256
    assert cfg.env.observation_width == 256
    assert cfg.env.gym_kwargs["observation_height"] == 256
    assert cfg.env.gym_kwargs["observation_width"] == 256
    assert cfg.env.features[LIBERO_KEY_PIXELS_AGENTVIEW].shape == (256, 256, 3)
    assert cfg.env.features[LIBERO_KEY_PIXELS_EYE_IN_HAND].shape == (256, 256, 3)


def test_wrapper_inference_warns_on_resolution_mismatch(caplog):
    """Behavioral red-to-green test for reporter's wrapper path:
    Direct policy inference (e.g. wrapper calling select_action/prepare_images without lerobot-eval)
    must emit an advisory warning when input tensor is 360x360 but declared metadata is 256x256.
    """
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SimpleNamespace(
        training=False,
        _warned_visual_resolution=False,
        config=SimpleNamespace(
            image_features=["observation.images.image"],
            input_features={
                "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
            },
            resize_imgs_with_padding=[512, 512],
            empty_cameras=0,
        ),
    )
    # Reporter's wrapper passes rendered 360x360 frames (B, C, H, W)
    batch = {
        "observation.images.image": torch.zeros((1, 3, 360, 360), dtype=torch.float32),
    }

    with caplog.at_level(logging.WARNING):
        SmolVLAPolicy.prepare_images(policy, batch)

    warnings = [r.message for r in caplog.records if "differing from declared" in r.message]
    assert len(warnings) == 1, f"Expected 1 runtime warning in wrapper path, got {len(warnings)}"
    assert "360x360" in warnings[0]
    assert "256x256" in warnings[0]


def test_wrapper_inference_warns_at_most_once(caplog):
    """Runtime warning in wrapper inference must fire at most once across repeated steps."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SimpleNamespace(
        training=False,
        _warned_visual_resolution=False,
        config=SimpleNamespace(
            image_features=["observation.images.image"],
            input_features={
                "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
            },
            resize_imgs_with_padding=[512, 512],
            empty_cameras=0,
        ),
    )
    batch = {
        "observation.images.image": torch.zeros((1, 3, 360, 360), dtype=torch.float32),
    }

    with caplog.at_level(logging.WARNING):
        # Step 1
        SmolVLAPolicy.prepare_images(policy, batch)
        # Step 2
        SmolVLAPolicy.prepare_images(policy, batch)
        # Step 3
        SmolVLAPolicy.prepare_images(policy, batch)

    warnings = [r.message for r in caplog.records if "differing from declared" in r.message]
    assert len(warnings) == 1, f"Expected exactly 1 warning across repeated steps, got {len(warnings)}"


def test_wrapper_inference_no_warning_when_matching_or_training(caplog):
    """Runtime check in prepare_images must be silent when inputs match, during training, or when resizing is None."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    # 1. Matching resolution 256x256
    policy_matching = SimpleNamespace(
        training=False,
        _warned_visual_resolution=False,
        config=SimpleNamespace(
            image_features=["observation.images.image"],
            input_features={
                "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
            },
            resize_imgs_with_padding=[512, 512],
            empty_cameras=0,
        ),
    )
    batch_256 = {"observation.images.image": torch.zeros((1, 3, 256, 256), dtype=torch.float32)}
    with caplog.at_level(logging.WARNING):
        SmolVLAPolicy.prepare_images(policy_matching, batch_256)
    assert not any("differing from declared" in r.message for r in caplog.records)

    # 2. Training mode (policy.training is True)
    caplog.clear()
    policy_training = SimpleNamespace(
        training=True,
        _warned_visual_resolution=False,
        config=SimpleNamespace(
            image_features=["observation.images.image"],
            input_features={
                "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
            },
            resize_imgs_with_padding=[512, 512],
            empty_cameras=0,
        ),
    )
    batch_360 = {"observation.images.image": torch.zeros((1, 3, 360, 360), dtype=torch.float32)}
    with caplog.at_level(logging.WARNING):
        SmolVLAPolicy.prepare_images(policy_training, batch_360)
    assert not any("differing from declared" in r.message for r in caplog.records)

    # 3. resize_imgs_with_padding is None
    caplog.clear()
    policy_no_resize = SimpleNamespace(
        training=False,
        _warned_visual_resolution=False,
        config=SimpleNamespace(
            image_features=["observation.images.image"],
            input_features={
                "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
            },
            resize_imgs_with_padding=None,
            empty_cameras=0,
        ),
    )
    with caplog.at_level(logging.WARNING):
        SmolVLAPolicy.prepare_images(policy_no_resize, batch_360)
    assert not any("differing from declared" in r.message for r in caplog.records)


def test_wrapper_inference_select_and_predict_action_reach_prepare_images(caplog, monkeypatch):
    """Verify that both select_action and predict_action_chunk reach prepare_images in SmolVLA."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    # Mock VLAFlowMatching model and RTC check to isolate prepare_images call path
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    torch.nn.Module.__init__(policy)
    policy.training = False
    policy._warned_visual_resolution = False
    policy.config = SimpleNamespace(
        image_features=["observation.images.image"],
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
        },
        resize_imgs_with_padding=[512, 512],
        empty_cameras=0,
        adapt_to_pi_aloha=False,
        n_action_steps=1,
        max_action_dim=7,
        max_state_dim=8,
        action_feature=SimpleNamespace(shape=(7,)),
        rtc_config=None,
    )
    policy._queues = {
        "action": deque([torch.zeros((1, 7), dtype=torch.float32)]),
    }
    policy.model = MagicMock()
    policy.model.sample_actions.return_value = torch.zeros((1, 1, 7), dtype=torch.float32)

    batch = {
        "observation.images.image": torch.zeros((1, 3, 360, 360), dtype=torch.float32),
        "observation.state": torch.zeros((1, 8), dtype=torch.float32),
        "observation.language.tokens": torch.zeros((1, 10), dtype=torch.long),
        "observation.language.attention_mask": torch.ones((1, 10), dtype=torch.bool),
    }

    # 1. predict_action_chunk reaches prepare_images
    with caplog.at_level(logging.WARNING):
        policy.predict_action_chunk(batch)
    assert any("SmolVLA received input image(s) differing" in r.message for r in caplog.records)

    # 2. select_action with a fresh unwarned policy reaches prepare_images
    caplog.clear()
    policy._warned_visual_resolution = False
    policy._queues["action"] = deque()  # empty to trigger get_action_chunk
    with caplog.at_level(logging.WARNING):
        policy.select_action(batch)
    assert any("SmolVLA received input image(s) differing" in r.message for r in caplog.records)


def test_deduplication_between_eval_main_and_runtime_policy(caplog, monkeypatch, tmp_path):
    """In stock lerobot-eval, eval_main's early warning marks the policy so prepare_images does not duplicate."""
    _mock_eval_pipeline(monkeypatch)

    env = LiberoEnv()  # defaults to 360x360
    policy_input_features = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    cfg = _dummy_eval_pipeline_config(env, policy_input_features, output_dir=tmp_path)

    # Mock make_policy to return a policy object tracking _warned_visual_resolution
    mock_policy = MagicMock()
    mock_policy.training = False
    mock_policy._warned_visual_resolution = False
    mock_policy.config.device = "cpu"
    monkeypatch.setattr(lerobot_eval, "make_policy", MagicMock(return_value=mock_policy))

    with caplog.at_level(logging.WARNING):
        lerobot_eval.eval_main(cfg)

    # 1. Exactly one resolution warning from eval_main
    mismatch_warnings = [
        r.message
        for r in caplog.records
        if "resolution" in r.message.lower() and "differ" in r.message.lower()
    ]
    assert len(mismatch_warnings) == 1
    # 2. Policy was marked as warned by eval_main to prevent runtime duplicate
    assert mock_policy._warned_visual_resolution is True


def test_runtime_latch_not_consumed_by_initial_matching_frame(caplog):
    """Section D Defect 1: Latch must not be consumed by a matching frame.
    Sequence: 256 -> 360 -> 360 must yield exactly 0 -> 1 -> 1 warnings total.
    """
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SimpleNamespace(
        training=False,
        _warned_visual_resolution=False,
        config=SimpleNamespace(
            image_features=["observation.images.image"],
            input_features={
                "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
            },
            resize_imgs_with_padding=[512, 512],
            empty_cameras=0,
        ),
    )

    batch_256 = {"observation.images.image": torch.zeros((1, 3, 256, 256), dtype=torch.float32)}
    batch_360 = {"observation.images.image": torch.zeros((1, 3, 360, 360), dtype=torch.float32)}

    with caplog.at_level(logging.WARNING):
        # Step 1: Matching frame (256x256)
        SmolVLAPolicy.prepare_images(policy, batch_256)
        assert len([r for r in caplog.records if "differing from declared" in r.message]) == 0
        assert policy._warned_visual_resolution is False, "Latch must not be consumed on matching frame!"

        # Step 2: Mismatched frame (360x360) - must warn
        SmolVLAPolicy.prepare_images(policy, batch_360)
        assert len([r for r in caplog.records if "differing from declared" in r.message]) == 1
        assert policy._warned_visual_resolution is True, "Latch must be set after mismatch warning"

        # Step 3: Subsequent mismatched frame (360x360) - must not warn again
        SmolVLAPolicy.prepare_images(policy, batch_360)
        assert len([r for r in caplog.records if "differing from declared" in r.message]) == 1


def test_runtime_latch_handles_late_appearing_mismatched_camera(caplog):
    """Section D Defect 1: A camera appearing late with mismatched resolution must trigger the advisory."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SimpleNamespace(
        training=False,
        _warned_visual_resolution=False,
        config=SimpleNamespace(
            image_features=["observation.images.cam1", "observation.images.cam2"],
            input_features={
                "observation.images.cam1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
                "observation.images.cam2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            },
            resize_imgs_with_padding=[512, 512],
            empty_cameras=0,
        ),
    )

    with caplog.at_level(logging.WARNING):
        # Step 1: Only cam1 is present and matches
        batch_step1 = {"observation.images.cam1": torch.zeros((1, 3, 256, 256), dtype=torch.float32)}
        SmolVLAPolicy.prepare_images(policy, batch_step1)
        assert len([r for r in caplog.records if "differing from declared" in r.message]) == 0
        assert policy._warned_visual_resolution is False

        # Step 2: cam2 appears with 360x360
        batch_step2 = {
            "observation.images.cam1": torch.zeros((1, 3, 256, 256), dtype=torch.float32),
            "observation.images.cam2": torch.zeros((1, 3, 360, 360), dtype=torch.float32),
        }
        SmolVLAPolicy.prepare_images(policy, batch_step2)
        mismatch_records = [r for r in caplog.records if "differing from declared" in r.message]
        assert len(mismatch_records) == 1
        assert "cam2" in mismatch_records[0].message
        assert policy._warned_visual_resolution is True


def test_runtime_warning_independent_across_policy_instances(caplog):
    """Section D Defect 1: Separate policy instances must have independent warning state."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    def make_dummy_policy():
        return SimpleNamespace(
            training=False,
            _warned_visual_resolution=False,
            config=SimpleNamespace(
                image_features=["observation.images.image"],
                input_features={
                    "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
                },
                resize_imgs_with_padding=[512, 512],
                empty_cameras=0,
            ),
        )

    pol1 = make_dummy_policy()
    pol2 = make_dummy_policy()
    batch_360 = {"observation.images.image": torch.zeros((1, 3, 360, 360), dtype=torch.float32)}

    with caplog.at_level(logging.WARNING):
        SmolVLAPolicy.prepare_images(pol1, batch_360)
        assert len([r for r in caplog.records if "differing from declared" in r.message]) == 1
        assert pol1._warned_visual_resolution is True
        assert pol2._warned_visual_resolution is False

        caplog.clear()
        SmolVLAPolicy.prepare_images(pol2, batch_360)
        assert len([r for r in caplog.records if "differing from declared" in r.message]) == 1
        assert pol2._warned_visual_resolution is True


def test_cross_layer_suppression_only_on_actual_mismatch(caplog, monkeypatch, tmp_path):
    """Section D Defect 2: eval_main must only suppress runtime warning if a static mismatch was actually warned."""
    _mock_eval_pipeline(monkeypatch)

    # 1. Matching static config: eval_main should NOT set policy._warned_visual_resolution = True
    env_matching = LiberoEnv(observation_height=256, observation_width=256)
    policy_input_features = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    cfg_matching = _dummy_eval_pipeline_config(env_matching, policy_input_features, output_dir=tmp_path)

    mock_policy = MagicMock()
    mock_policy.training = False
    mock_policy._warned_visual_resolution = False
    mock_policy.config.device = "cpu"
    monkeypatch.setattr(lerobot_eval, "make_policy", MagicMock(return_value=mock_policy))

    lerobot_eval.eval_main(cfg_matching)
    # Because static config matched, runtime warning must NOT be suppressed
    assert mock_policy._warned_visual_resolution is False


def test_many_to_one_camera_collision_diagnostic(caplog):
    """Section D Defect 3: Detect multiple env cameras mapping to one policy key and log conflict."""
    env_cfg = SimpleNamespace(
        features={
            "pixels/cam_a": PolicyFeature(type=FeatureType.VISUAL, shape=(360, 360, 3)),
            "pixels/cam_b": PolicyFeature(type=FeatureType.VISUAL, shape=(360, 360, 3)),
        },
        features_map={
            "pixels/cam_a": "observation.images.image",
            "pixels/cam_b": "observation.images.image",
        },
    )
    pol_cfg = SimpleNamespace(
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
        }
    )

    with caplog.at_level(logging.WARNING):
        result = _warn_on_visual_resolution_mismatch(pol_cfg, env_cfg)

    assert result is False
    # Conflict diagnostic must be logged
    conflict_warnings = [
        r.message for r in caplog.records if "Conflicting camera mapping detected" in r.message
    ]
    assert len(conflict_warnings) == 1
    assert "'pixels/cam_a'" in conflict_warnings[0]
    assert "'pixels/cam_b'" in conflict_warnings[0]
    assert "'observation.images.image'" in conflict_warnings[0]


def test_typed_nonvisual_3d_feature_not_treated_as_camera(caplog):
    """Section D Defect 4: Typed STATE/ACTION features shaped (10, 10, 3) must NOT be classified as cameras."""
    env_cfg = SimpleNamespace(
        features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(10, 10, 3)),
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(10, 10, 3)),
            "pixels/cam": PolicyFeature(type=FeatureType.VISUAL, shape=(256, 256, 3)),
        },
        features_map={
            "pixels/cam": "observation.images.image",
        },
    )
    pol_cfg = SimpleNamespace(
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(10, 10, 3)),
        }
    )

    with caplog.at_level(logging.WARNING):
        result = _warn_on_visual_resolution_mismatch(pol_cfg, env_cfg)

    # No resolution mismatch should be logged because the only visual camera matches (256x256 == 256x256)
    mismatch_warnings = [r.message for r in caplog.records if "differ from declared policy" in r.message]
    assert len(mismatch_warnings) == 0
    assert result is False


def test_incompatible_multi_target_resolutions_explains_no_global_size(caplog):
    """Section D Defect 5: Incompatible target resolutions must not suggest literal <h> <w> placeholders."""
    env_cfg = SimpleNamespace(
        features={
            "pixels/cam1": PolicyFeature(type=FeatureType.VISUAL, shape=(360, 360, 3)),
            "pixels/cam2": PolicyFeature(type=FeatureType.VISUAL, shape=(360, 360, 3)),
        },
        features_map={
            "pixels/cam1": "observation.images.cam1",
            "pixels/cam2": "observation.images.cam2",
        },
        observation_height=360,
        observation_width=360,
    )
    pol_cfg = SimpleNamespace(
        input_features={
            "observation.images.cam1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.images.cam2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
        }
    )

    with caplog.at_level(logging.WARNING):
        result = _warn_on_visual_resolution_mismatch(pol_cfg, env_cfg)

    assert result is True
    mismatch_warnings = [r.message for r in caplog.records if "differ from declared policy" in r.message]
    assert len(mismatch_warnings) == 1
    # Must NOT contain literal placeholders <h> or <w>
    assert "<h>" not in mismatch_warnings[0]
    assert "<w>" not in mismatch_warnings[0]
    # Must explain that multiple distinct target resolutions are required
    assert "Multiple distinct policy target resolutions are required" in mismatch_warnings[0]
    assert "128x128" in mismatch_warnings[0]
    assert "256x256" in mismatch_warnings[0]
    assert (
        "no single global environment resolution can align all cameras simultaneously" in mismatch_warnings[0]
    )
