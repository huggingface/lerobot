from pathlib import Path

import torch

from lerobot.policies.hy_vla.configuration_hy_vla import HyVLAConfig
from lerobot.policies.hy_vla.processor_hy_vla import (
    make_hy_vla_pre_post_processors,
    reconnect_hy_vla_processors,
)
from lerobot.processor import (
    PolicyProcessorPipeline,
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)


def _native_state() -> torch.Tensor:
    state = torch.zeros(16)
    state[6] = 1
    state[14] = 1
    state[7] = 0.2
    state[15] = 0.8
    return state


def _identity_relative(left_gripper: float = 0.2, right_gripper: float = 0.8) -> torch.Tensor:
    return torch.tensor(
        [0, 0, 0, 1, 0, 0, 0, 1, 0, left_gripper, 0, 0, 0, 1, 0, 0, 0, 1, 0, right_gripper],
        dtype=torch.float32,
    )


def _stats(horizon: int, absolute: bool = False) -> dict[str, torch.Tensor]:
    stats = {
        "qpos_mean": torch.zeros(20),
        "qpos_std": torch.ones(20),
        "action_mean": torch.zeros(horizon, 20),
        "action_std": torch.ones(horizon, 20),
    }
    if absolute:
        stats["action_mean_abs"] = torch.zeros(horizon, 20)
        stats["action_std_abs"] = torch.ones(horizon, 20)
    return stats


def _batch(state: torch.Tensor, action: torch.Tensor | None = None) -> dict:
    batch = {
        "observation.state": state,
        "observation.images.top_head": torch.zeros(3, 8, 8),
        "observation.images.hand_left": torch.zeros(3, 8, 8),
        "observation.images.hand_right": torch.zeros(3, 8, 8),
        "task": "raw_task_with_underscore\nkeep-newline",
    }
    if action is not None:
        batch["action"] = action
    return batch


def test_processor_preserves_task_and_serializes_stats(tmp_path: Path):
    config = HyVLAConfig(device="cpu")
    stats = _stats(50)
    stats["action_std"] = stats["action_std"].double()
    preprocessor, postprocessor = make_hy_vla_pre_post_processors(config, norm_stats=stats)
    state = _native_state()
    processed = preprocessor(_batch(state, state.repeat(50, 1)))
    assert processed["task"] == ["raw_task_with_underscore\nkeep-newline"]
    assert processed["observation.state"].shape == (1, 32)
    assert processed["observation.state.mask"].shape == (1, 32)
    assert processed["action"].shape == (1, 50, 32)

    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)
    assert (tmp_path / "policy_preprocessor_step_2_hy_vla_encode_v1.safetensors").is_file()
    assert (tmp_path / "policy_postprocessor_step_0_hy_vla_decode_v1.safetensors").is_file()

    loaded_pre = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    loaded_post = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    reconnect_hy_vla_processors(loaded_pre, loaded_post)
    loaded_encoder = loaded_pre.steps[2]
    loaded_decoder = loaded_post.steps[0]
    assert loaded_encoder.action_std.dtype == torch.float64
    assert loaded_decoder.action_std.dtype == torch.float64
    loaded_pre(_batch(state))
    decoded = loaded_post(_identity_relative())
    torch.testing.assert_close(decoded, state.double())


def test_relative_absolute_tokens_are_paired_and_blended():
    config = HyVLAConfig(
        device="cpu",
        chunk_size=40,
        n_action_steps=40,
        action_representation="relative_absolute",
        action_decode_mode="blend",
        embodiment="robotwin_dual_arm",
        native_quaternion_order="xyzw",
        use_video_encoder=True,
        img_history_size=6,
        img_history_interval=5,
        execution_horizon=7,
    )
    preprocessor, postprocessor = make_hy_vla_pre_post_processors(
        config, norm_stats=_stats(20, absolute=True)
    )
    state = _native_state()
    preprocessor(_batch(state))
    absolute = torch.tensor(
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0.4, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0.6],
        dtype=torch.float32,
    )
    decoded = postprocessor(torch.cat((_identity_relative(), absolute)))
    torch.testing.assert_close(decoded[[0, 8]], torch.tensor([0.5, 1.0]))
    torch.testing.assert_close(decoded[[7, 15]], torch.tensor([0.3, 0.7]))
