# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0
"""Offline checks of SANA-Robot, inference and training, with a reduced policy, video VAE and text encoder."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Gemma2Config, Gemma2ForCausalLM, PreTrainedTokenizerFast

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.policies import make_pre_post_processors
from lerobot.policies.sana_robot import modeling_sana_robot
from lerobot.policies.sana_robot.configuration_sana_robot import SanaRobotConfig
from lerobot.policies.sana_robot.modeling_sana_robot import (
    SanaRobotPolicy,
    episode_window,
    resolve_training_backend,
)
from lerobot.policies.sana_robot.processor_sana_robot import (
    LIBERO_BASE_POS,
    LIBERO_BASE_QUAT,
    LIBERO_JOINT_POS,
    SanaRobotLiberoAction,
    SanaRobotLiberoInstruction,
    SanaRobotLiberoObservation,
    SanaRobotObservations,
    panda_grip_site_pose,
    panda_hand_pose,
    pose_matrix,
    robot_statistics,
)
from lerobot.policies.sana_robot.sana_robot_core.causal_vae import LTX2VideoEncoder3d
from lerobot.policies.sana_robot.sana_robot_core.robot80 import rotation_to_rot6d
from lerobot.policies.sana_robot.sana_robot_core.teacher_forcing import padded_window_frames
from lerobot.policies.sana_robot.sana_robot_core.windows import window_chunk_count
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.rotation import Rotation
from tests.utils import require_cuda

REPO_ROOT = Path(__file__).resolve().parents[3]
LATENT_CHANNELS = 8
TEXT_WIDTH = 32
VAE_CONFIG = {
    "in_channels": 3,
    "latent_channels": LATENT_CHANNELS,
    "block_out_channels": [8, 8, 16, 16],
    "down_block_types": ["LTX2VideoDownBlock3D"] * 4,
    "spatio_temporal_scaling": [True, True, True, True],
    "layers_per_block": [1, 1, 1, 1, 1],
    "downsample_type": ["spatial", "temporal", "spatiotemporal", "spatiotemporal"],
    "patch_size": 4,
    "patch_size_t": 1,
    "resnet_norm_eps": 1e-6,
    "encoder_causal": True,
    "encoder_spatial_padding_mode": "zeros",
    "scaling_factor": 1.0,
    "temporal_compression_ratio": 8,
}


@pytest.fixture
def encoders(tmp_path, monkeypatch):
    """Folders of a reduced LTX-2 video VAE and of a reduced Gemma 2 with a word tokenizer."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    torch.manual_seed(0)
    vae = tmp_path / "ltx" / "vae"
    vae.mkdir(parents=True)
    (vae / "config.json").write_text(json.dumps(VAE_CONFIG))
    encoder = LTX2VideoEncoder3d(
        in_channels=3,
        out_channels=LATENT_CHANNELS,
        block_out_channels=tuple(VAE_CONFIG["block_out_channels"]),
        down_block_types=tuple(VAE_CONFIG["down_block_types"]),
        spatio_temporal_scaling=tuple(VAE_CONFIG["spatio_temporal_scaling"]),
        layers_per_block=tuple(VAE_CONFIG["layers_per_block"]),
        downsample_type=tuple(VAE_CONFIG["downsample_type"]),
    )
    weights = {f"encoder.{name}": value for name, value in encoder.state_dict().items()}
    weights["latents_mean"] = torch.zeros(LATENT_CHANNELS)
    weights["latents_std"] = torch.ones(LATENT_CHANNELS)
    save_file(weights, vae / "diffusion_pytorch_model.safetensors")

    text = tmp_path / "gemma"
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "pick": 2, "place": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]").save_pretrained(
        text
    )
    Gemma2ForCausalLM(
        Gemma2Config(
            vocab_size=8,
            hidden_size=TEXT_WIDTH,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=1,
        )
    ).save_pretrained(text)
    return {"vae_path": str(tmp_path / "ltx"), "text_encoder_path": str(text)}


def config_fields(encoders, **overrides):
    """The fields of a reduced policy on the reduced encoders."""
    return {
        "device": "cpu",
        "dtype": torch.float32,
        "push_to_hub": False,
        "view_sizes": [[64, 64], [64, 64]],
        "hidden_size": 48,
        "depth": 4,
        "cross_attention_heads": 4,
        "in_channels": LATENT_CHANNELS,
        "caption_channels": TEXT_WIDTH,
        "linear_head_dim": 12,
        "softmax_head_dim": 24,
        "attn_res_block_size": 2,
        **encoders,
        **overrides,
    }


def make_config(encoders, **overrides):
    return SanaRobotConfig(**config_fields(encoders, **overrides))


def make_policy(config, noise=0.1):
    """A policy whose weights are all random: a new policy predicts no action flow, so its actions are its noise."""
    policy = SanaRobotPolicy(config)
    generator = torch.Generator().manual_seed(0)
    for parameter in policy.parameters():
        parameter.data.add_(torch.randn(parameter.shape, generator=generator) * noise)
    return policy


def stats(config):
    (states, actions) = (len(config.state_slots), len(config.action_slots))
    return {
        OBS_STATE: {"mean": torch.linspace(-0.2, 0.2, states), "std": torch.linspace(0.5, 1.5, states)},
        ACTION: {"mean": torch.linspace(-0.1, 0.1, actions), "std": torch.linspace(0.5, 2.0, actions)},
    }


def observation(config, step, height=64, width=64, task="pick"):
    generator = torch.Generator().manual_seed(step)
    batch = {key: torch.rand(1, 3, height, width, generator=generator) for key in config.camera_keys}
    batch[OBS_STATE] = torch.rand(1, len(config.state_slots), generator=generator)
    batch["task"] = [task]
    return batch


def rollout(policy, preprocess, postprocess, steps, seed=0, **arguments):
    policy.reset()
    torch.manual_seed(seed)
    actions = [
        postprocess(policy.select_action(preprocess(observation(policy.config, step, **arguments))))
        for step in range(steps)
    ]
    return torch.cat(actions)


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_action_steps": 6},
        {"n_action_steps": 64},
        {"chunk_size": 24},
        {"view_sizes": [[64, 64], [60, 64]]},
        {"view_slots": [0]},
        {"camera_keys": ["observation.images.image", "observation.images.image"]},
        {"resize": "pad"},
        {"action_mode": "delta"},
        {"spatial_layout": "3x3"},
        {"state_slots": [79, 80]},
        {"num_inference_steps": 0},
        {"training_backend": "flash"},
        {"augmentation": {"mixup": {}}},
    ],
)
def test_config_rejects_what_the_policy_cannot_run(encoders, overrides):
    with pytest.raises(ValueError):
        make_config(encoders, **overrides)


def test_select_action_plans_on_the_episode_clock(encoders):
    config = make_config(encoders)
    policy = make_policy(config)
    (preprocess, postprocess) = make_pre_post_processors(config, dataset_stats=stats(config))
    planned = []
    plan = policy._plan
    policy._plan = lambda state, instruction: (planned.append(policy._step), plan(state, instruction))
    actions = rollout(policy, preprocess, postprocess, 40)
    assert planned == [0, 16, 32]
    assert actions.shape == (40, len(config.action_slots))
    assert torch.isfinite(actions).all()
    assert policy.count_queued_actions() == 8
    assert all(name.startswith("model.") for name in policy.state_dict())


def test_actions_repeat_after_reset_and_depend_on_the_memory(encoders):
    config = make_config(encoders)
    policy = make_policy(config)
    (preprocess, postprocess) = make_pre_post_processors(config, dataset_stats=stats(config))
    first = rollout(policy, preprocess, postprocess, 36)
    assert torch.equal(first, rollout(policy, preprocess, postprocess, 36))
    assert not torch.equal(first, rollout(policy, preprocess, postprocess, 36, seed=1))

    # The chunk generated at step 32 reads the frames before it: other frames in between give other actions.
    policy.reset()
    torch.manual_seed(0)
    for step in range(36):
        frames = observation(config, step if step < 4 or step >= 32 else step + 100)
        batch = {**observation(config, step), **{key: frames[key] for key in config.camera_keys}}
        action = postprocess(policy.select_action(preprocess(batch)))
    assert not torch.equal(action, first[-1:])


@pytest.mark.parametrize("video_cfg_scale", [1.0, 3.0])
def test_save_and_load_keep_the_actions(encoders, tmp_path, video_cfg_scale):
    config = make_config(encoders, video_cfg_scale=video_cfg_scale)
    policy = make_policy(config)
    (preprocess, postprocess) = make_pre_post_processors(config, dataset_stats=stats(config))
    expected = rollout(policy, preprocess, postprocess, 20)
    policy.save_pretrained(tmp_path / "policy")
    preprocess.save_pretrained(tmp_path / "policy")
    postprocess.save_pretrained(tmp_path / "policy")

    loaded = SanaRobotPolicy.from_pretrained(tmp_path / "policy")
    assert loaded.config.video_cfg_scale == video_cfg_scale
    assert loaded.config.view_sizes == config.view_sizes
    (preprocess, postprocess) = make_pre_post_processors(
        loaded.config, pretrained_path=str(tmp_path / "policy")
    )
    assert torch.equal(expected, rollout(loaded, preprocess, postprocess, 20))


def test_three_views_of_two_sizes_in_the_t_shape_layout(encoders):
    config = make_config(
        encoders,
        camera_keys=["observation.images.head", "observation.images.left", "observation.images.right"],
        view_sizes=[[64, 96], [32, 64], [32, 64]],
        view_slots=[0, 1, 2],
        view_descriptions=["head camera", "left wrist camera", "right wrist camera"],
        spatial_layout="t_shape",
        resize="stretch",
        state_slots=list(range(0, 17)) + list(range(29, 46)),
        action_slots=list(range(7, 17)) + list(range(36, 46)),
        n_action_steps=32,
    )
    policy = make_policy(config)
    (preprocess, postprocess) = make_pre_post_processors(config, dataset_stats=stats(config))
    actions = rollout(policy, preprocess, postprocess, 34, height=48, width=80)
    assert actions.shape == (34, 20)
    assert torch.isfinite(actions).all()


def test_a_new_instruction_takes_over_at_the_next_chunk(encoders):
    config = make_config(encoders)
    policy = make_policy(config)
    (preprocess, postprocess) = make_pre_post_processors(config, dataset_stats=stats(config))
    same = rollout(policy, preprocess, postprocess, 20)
    policy.reset()
    torch.manual_seed(0)
    changed = torch.cat(
        [
            postprocess(
                policy.select_action(
                    preprocess(observation(config, step, task="pick" if step < 8 else "place"))
                )
            )
            for step in range(20)
        ]
    )
    assert torch.equal(same[:16], changed[:16])
    assert not torch.equal(same[16:], changed[16:])


def test_batches_of_episodes_are_rejected_at_inference(encoders):
    config = make_config(encoders)
    policy = make_policy(config)
    with pytest.raises(NotImplementedError):
        policy.predict_action_chunk({})
    batch = observation(config, 0)
    batch["task"] = ["pick", "place"]
    preprocess = make_pre_post_processors(config, dataset_stats=stats(config))[0]
    with pytest.raises(ValueError):
        policy.select_action(preprocess(batch))
    with pytest.raises(ValueError):
        SanaRobotPolicy(make_config(encoders, vae_path=None)).select_action(
            preprocess(observation(config, 0))
        )


@pytest.mark.parametrize(
    ("resize", "input_size", "expected"),
    [("stretch", (48, 80), (64, 64)), ("crop", (48, 80), (64, 64)), ("crop", (64, 64), (64, 64))],
)
def test_frames_are_resized_as_in_training(resize, input_size, expected):
    step = SanaRobotObservations(["observation.images.image"], [list(expected)], resize)
    image = torch.rand(1, 3, *input_size)
    before = image.clone()
    frames = step.observation({"observation.images.image": image})["observation.images.image"]
    assert torch.equal(image, before)
    assert frames.shape == (1, 3, *expected)
    assert frames.min() >= -1 and frames.max() <= 1
    levels = (frames + 1) * 127.5
    assert torch.allclose(levels, levels.round(), atol=1e-4)
    if input_size == expected:
        assert torch.allclose(frames, (image * 255).round() / 127.5 - 1, atol=1e-6)
    elif resize == "crop":
        scaled = torch.nn.functional.interpolate(image, size=(64, 107), mode="bilinear", align_corners=False)
        assert torch.allclose(frames, (scaled[..., 21:85] * 255).round() / 127.5 - 1, atol=1e-6)


def test_state_and_actions_use_the_checkpoint_statistics(encoders):
    config = make_config(encoders)
    statistics = stats(config)
    (preprocess, postprocess) = make_pre_post_processors(config, dataset_stats=statistics)
    batch = observation(config, 3)
    state = batch[OBS_STATE].clone()
    normalized = preprocess(batch)[OBS_STATE]
    assert torch.allclose(
        normalized, (state - statistics[OBS_STATE]["mean"]) / statistics[OBS_STATE]["std"], atol=1e-6
    )
    action = torch.rand(1, len(config.action_slots))
    restored = postprocess(action.clone())
    assert torch.allclose(
        restored, action * statistics[ACTION]["std"] + statistics[ACTION]["mean"], atol=1e-6
    )


def libero_observation(joints, flange, fingers, world_from_base):
    """The observation LIBERO gives for joint positions [7], a flange pose [4, 4] in the frame of the robot base,
    finger positions [2] and a base at ``world_from_base``."""
    site = world_from_base @ flange @ panda_grip_site_pose()
    hand = world_from_base @ flange @ panda_hand_pose()
    state = np.concatenate((site[:3, 3], Rotation.from_matrix(hand[:3, :3]).as_rotvec(), fingers))
    batch = {
        OBS_STATE: torch.from_numpy(state).float()[None],
        LIBERO_JOINT_POS: torch.from_numpy(joints).float()[None],
        LIBERO_BASE_POS: torch.from_numpy(world_from_base[:3, 3]).float()[None],
        LIBERO_BASE_QUAT: torch.from_numpy(Rotation.from_matrix(world_from_base[:3, :3]).as_quat()).float()[
            None
        ],
    }
    return batch, site


def test_libero_processors_move_between_the_world_and_the_robot_base():
    (pre, post) = make_env_pre_post_processors(
        LiberoEnvConfig(control_mode="absolute"), SanaRobotConfig(device="cpu")
    )
    (observe, act) = (pre.steps[0], post.steps[0])
    assert isinstance(observe, SanaRobotLiberoObservation) and isinstance(act, SanaRobotLiberoAction)
    world_from_base = pose_matrix(
        np.array([-0.66, 0.1, 0.912]), Rotation.from_rotvec(np.array([0.0, 0.0, 0.3])).as_matrix()
    )
    joints = np.array([0.1, -0.3, 0.05, -2.2, 0.02, 2.0, 0.7])
    flange = pose_matrix(
        np.array([0.45, -0.05, 0.3]), Rotation.from_rotvec(np.array([2.9, 0.4, -0.2])).as_matrix()
    )
    (batch, site) = libero_observation(joints, flange, np.array([0.03, -0.03]), world_from_base)
    result = observe.observation(batch)
    state = result[OBS_STATE]
    assert state.shape == (1, 17)
    assert not {LIBERO_JOINT_POS, LIBERO_BASE_POS, LIBERO_BASE_QUAT} & set(result)
    assert np.allclose(state[0, :7].numpy(), joints, atol=1e-6)
    assert np.allclose(state[0, 7:10].numpy(), flange[:3, 3], atol=1e-6)
    assert np.allclose(state[0, 10:16].numpy(), rotation_to_rot6d(flange[:3, :3]), atol=1e-6)
    assert np.isclose(state[0, 16].item(), 0.25, atol=1e-6)
    assert np.allclose(observe.frames.world_from_base, world_from_base, atol=1e-6)

    # The flange pose of the observation, sent back as an action, is the grip site's pose in the world.
    action = torch.cat((state[0, 7:16], torch.tensor([0.2])))[None]
    command = act.action(action)[0].numpy()
    assert np.allclose(command[:3], site[:3, 3], atol=1e-5)
    assert np.allclose(Rotation.from_rotvec(command[3:6]).as_matrix(), site[:3, :3], atol=1e-5)
    assert command[6] == -1.0
    assert act.action(torch.cat((state[0, 7:16], torch.tensor([0.8])))[None])[0, 6] == 1.0

    # Without the joint positions and the base pose of `--env.obs_type=pixels_agent_pos_joints`.
    with pytest.raises(ValueError, match="pixels_agent_pos_joints"):
        observe.observation({OBS_STATE: torch.zeros(1, 8)})
    with pytest.raises(ValueError):
        SanaRobotLiberoAction().action(action)


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        ("put the bowl on the plate", "put the bowl on the plate"),
        ("put the bowl on the plate table 1", "put the bowl on the plate"),
        ("put the bowl on the plate tb 9", "put the bowl on the plate"),
        ("put the bowl on the plate view 0 0 100 8 350 initstate 0", "put the bowl on the plate"),
        ("put the bowl on the plate view 0 0 100 0 0 initstate 0 noise 45", "put the bowl on the plate"),
        ("put the bowl on the plate add 10", "put the bowl on the plate"),
        ("put the bowl on the plate light 1", "put the bowl on the plate"),
        ("turn on the light and set the table", "turn on the light and set the table"),
    ],
)
def test_libero_plus_tags_leave_the_instruction(instruction, expected):
    (pre, _post) = make_env_pre_post_processors(
        LiberoEnvConfig(control_mode="absolute"), SanaRobotConfig(device="cpu")
    )
    assert isinstance(pre.steps[1], SanaRobotLiberoInstruction)
    assert pre.steps[1].complementary_data({"task": [instruction]})["task"] == [expected]
    assert pre.steps[1].complementary_data({"task": instruction})["task"] == expected


def load_converter():
    """`examples/sana_robot/convert_checkpoint.py` as a module."""
    path = REPO_ROOT / "examples" / "sana_robot" / "convert_checkpoint.py"
    spec = importlib.util.spec_from_file_location("sana_robot_convert_checkpoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def released(tmp_path, encoders):
    """A released checkpoint folder (weights, dataset description, normalization) and the policy it holds."""
    policy = make_policy(make_config(encoders))
    folder = tmp_path / "released"
    folder.mkdir()
    torch.save({"state_dict_ema": policy.model.state_dict()}, folder / "sana_robot_ar_test.pth")
    meta = {
        "fps": 10.0,
        "embodiment": "test robot",
        "action_mode": "joint_only",
        "views": [
            {"name": "a", "slot": 0, "description": "first camera"},
            {"name": "b", "slot": 1, "description": "second camera"},
        ],
        "state_slots": list(range(8)),
        "action_slots": list(range(7)),
        "resize": "stretch",
    }
    (folder / "meta.json").write_text(json.dumps(meta))
    normalization = {
        "center": [0.1 * slot for slot in range(80)],
        "scale": [1.0 + 0.05 * slot for slot in range(80)],
        "mask": [slot < 6 for slot in range(80)],
    }
    (folder / "normalization.json").write_text(json.dumps(normalization))
    return folder, policy, meta, normalization


def test_a_converted_checkpoint_has_the_weights_the_robot_fields_and_the_statistics(
    encoders, released, tmp_path
):
    (folder, source, meta, normalization) = released
    out = tmp_path / "converted"
    load_converter().convert(folder, out, **config_fields(encoders))

    # The LeRobot directory stands alone: `config.json`, the weights and the two processors.
    policy = SanaRobotPolicy.from_pretrained(out)
    config = policy.config
    assert (config.fps, config.embodiment, config.action_mode, config.resize) == (
        10.0,
        "test robot",
        "joint_only",
        "stretch",
    )
    assert (config.view_slots, config.view_descriptions) == ([0, 1], ["first camera", "second camera"])
    assert (config.state_slots, config.action_slots) == (meta["state_slots"], meta["action_slots"])
    assert config.robot_state_feature.shape == (8,) and config.action_feature.shape == (7,)
    for name, value in source.model.state_dict().items():
        assert torch.equal(policy.model.state_dict()[name], value), name

    (preprocess, postprocess) = make_pre_post_processors(config, pretrained_path=str(out))
    batch = observation(config, 0)
    state = batch[OBS_STATE].clone()
    (center, scale) = (torch.tensor(normalization["center"][:8]), torch.tensor(normalization["scale"][:8]))
    expected = torch.cat((((state - center) / scale)[:, :6], state[:, 6:]), dim=1)
    assert torch.allclose(preprocess(batch)[OBS_STATE], expected, atol=1e-6)
    action = torch.rand(1, 7)
    restored = torch.cat(((action * scale[:7] + center[:7])[:, :6], action[:, 6:]), dim=1)
    assert torch.allclose(postprocess(action.clone()), restored, atol=1e-6)
    actions = rollout(policy, preprocess, postprocess, 18)
    assert actions.shape == (18, 7) and torch.isfinite(actions).all()


def test_a_converted_pretrained_checkpoint_holds_weights_only(encoders, released, tmp_path):
    (folder, source, _meta, _normalization) = released
    (folder / "meta.json").unlink()
    (folder / "normalization.json").unlink()
    converter = load_converter()
    out = tmp_path / "converted"
    converter.convert(folder, out, **config_fields(encoders))

    policy = SanaRobotPolicy.from_pretrained(out)
    defaults = make_config(encoders)
    assert policy.config.state_slots == defaults.state_slots and policy.config.fps == defaults.fps
    assert torch.equal(
        policy.model.state_dict()["blocks.0.attn.proj.weight"],
        source.model.state_dict()["blocks.0.attn.proj.weight"],
    )
    (preprocess, _postprocess) = make_pre_post_processors(policy.config, pretrained_path=str(out))
    assert not preprocess.steps[3].stats

    (folder / "second.pth").write_bytes(b"")
    with pytest.raises(ValueError, match="exactly one"):
        converter.convert(folder, tmp_path / "again", **config_fields(encoders))


def loader_window(values, anchor, offsets):
    """What a LeRobot dataset returns for the rows of an episode at ``anchor`` plus ``offsets``: the rows clamped
    to the episode, and which of them are padding."""
    rows = anchor + torch.as_tensor(list(offsets))
    return (values[rows.clamp(0, len(values) - 1)], (rows < 0) | (rows >= len(values)))


def training_batch(config, episode_rows, anchors, seed=0):
    """A batch of training samples after the preprocessor: one episode per sample, sampled at its anchor."""
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for rows, anchor in zip(episode_rows, anchors, strict=True):
        sample = {}
        for key, size in zip(config.camera_keys, config.view_sizes, strict=True):
            frames = torch.rand(rows, 3, *size, generator=generator) * 2 - 1
            sample[key] = loader_window(frames, anchor, config.image_observation_delta_indices)[0]
        state = torch.rand(rows, len(config.state_slots), generator=generator)
        sample[OBS_STATE] = loader_window(state, anchor, config.state_observation_delta_indices)[0]
        action = torch.rand(rows, len(config.action_slots), generator=generator)
        (sample[ACTION], sample[f"{ACTION}_is_pad"]) = loader_window(
            action, anchor, config.action_delta_indices
        )
        samples.append(sample)
    batch = {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}
    batch["task"] = ["pick"] * len(samples)
    return batch


@pytest.mark.parametrize("rows", [5, 36, 40, 100, 150])
def test_a_sample_is_trained_on_the_window_of_its_episode(rows):
    (reach, size, stride) = (256, 32, 4)
    for anchor in range(rows):
        offsets = torch.arange(-reach, reach) + anchor
        (first_row, chunks, head_rows) = episode_window((offsets >= 0) & (offsets < rows), size, stride)
        # The window starts at a chunk boundary at or before the first frame, and the sampled frame starts a chunk.
        assert head_rows == -anchor % size
        assert first_row == reach - anchor - head_rows
        assert (reach - first_row) % size == 0
        assert chunks == window_chunk_count(rows, head_rows, size, stride)


def test_an_episode_beyond_the_reach_is_trained_on_whole_chunks_inside_it():
    # In the middle the reach is cut on both sides; near the start only the end is cut and the window keeps its
    # rows before the episode; near the end only the start is cut and the last chunk runs past the episode.
    (reach, size, stride, rows) = (64, 32, 4, 400)
    for anchor, expected in ((200, (0, 4, 0)), (10, (32, 3, 22)), (390, (0, 3, 0))):
        offsets = torch.arange(-reach, reach) + anchor
        assert episode_window((offsets >= 0) & (offsets < rows), size, stride) == expected


def test_training_loss_of_whole_episodes(encoders):
    config = make_config(encoders, max_episode_steps=128, instruction_dropout=0.5)
    policy = make_policy(config)
    policy.train()
    assert config.image_observation_delta_indices == list(range(-128, 129, 4))
    assert config.state_observation_delta_indices == list(range(-128, 128, 32))
    assert config.action_delta_indices == list(range(-128, 128))
    batch = training_batch(config, [70, 120], [3, 77])
    torch.manual_seed(0)
    (loss, info) = policy.forward(batch)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss > 0
    # 70 rows from 29 rows before the first frame are 3 chunks, 120 rows from 19 rows before it are 5.
    assert info["chunks"] == 4.0
    assert info["video_loss"] > 0 and info["action_loss"] > 0
    loss.backward()
    assert all(parameter.grad is not None for parameter in policy.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in policy.parameters())

    torch.manual_seed(0)
    (losses, _info) = policy.forward(batch, reduction="none")
    assert losses.shape == (2,) and torch.allclose(losses.mean(), loss.detach(), atol=1e-6)

    with pytest.raises(ValueError, match="actions per sample"):
        policy.forward({**batch, ACTION: batch[ACTION][:, :-1]})


def test_a_packed_row_computes_every_window_as_alone(encoders, monkeypatch):
    config = make_config(encoders, max_episode_steps=128, instruction_dropout=0.0)
    policy = make_policy(config)
    policy.train()
    batch = training_batch(config, [70, 120, 4, 40], [3, 77, 2, 0])
    batch["task"] = ["pick", "place", "pick", "pick"]
    model_forward = policy.model.forward
    calls = []

    def recorded_forward(**arguments):
        calls.append((arguments, model_forward(**arguments)))
        return calls[-1][1]

    monkeypatch.setattr(policy.model, "forward", recorded_forward)
    torch.manual_seed(0)
    (losses, info) = policy.forward(batch, reduction="none")
    # One pass of the model for the three episodes that have a chunk to train.
    assert len(calls) == 1 and info["chunks"] == 2.5
    assert losses[2] == 0 and (losses[[0, 1, 3]] > 0).all()
    (arguments, packed) = calls[0]
    data_info = arguments["data_info"]
    assert data_info["chunk_counts"].tolist() == [3, 5, 2]
    (chunk_actions, chunk_frames) = policy.model.chunk_size(data_info)
    for index, count in enumerate([3, 5, 2]):
        row = slice(index, index + 1)
        frames = padded_window_frames(count, 5, chunk_frames)
        robot_rows = count * (chunk_actions + 1) - 1
        alone_info = {
            name: value[row] if torch.is_tensor(value) else value
            for (name, value) in data_info.items()
            if name != "chunk_counts"
        }
        alone_info["teacher_forcing_num_chunks"] = count
        for name in ("action", "action_mask", "action_timestep"):
            alone_info[name] = alone_info[name][:, :robot_rows]
        for name in ("chunk_states", "chunk_states_mask"):
            alone_info[name] = alone_info[name][:, : count - 1]
        with torch.no_grad():
            alone = model_forward(
                video_latents=arguments["video_latents"][row].index_select(2, frames),
                timestep=arguments["timestep"][row].index_select(1, frames),
                text_embeddings=arguments["text_embeddings"][row],
                text_mask=arguments["text_mask"][row],
                data_info=alone_info,
            )
        assert torch.allclose(
            packed["video_pred"][row].index_select(2, frames), alone["video_pred"], atol=1e-5
        )
        assert torch.allclose(packed["action_pred"][row, :robot_rows], alone["action_pred"], atol=1e-5)
        assert not packed["action_pred"][row, robot_rows:].any()


def test_the_training_backend_follows_the_device_and_triton(monkeypatch):
    (cpu, cuda) = (torch.device("cpu"), torch.device("cuda"))
    monkeypatch.setattr(modeling_sana_robot, "is_package_available", lambda name: True)
    assert resolve_training_backend("auto", cuda) == "triton"
    assert resolve_training_backend("triton", cuda) == "triton"
    assert resolve_training_backend("eager", cuda) == "eager"
    assert resolve_training_backend("auto", cpu) == "eager"
    assert resolve_training_backend("triton", cpu) == "eager"
    monkeypatch.setattr(modeling_sana_robot, "is_package_available", lambda name: False)
    assert resolve_training_backend("auto", cuda) == "eager"
    with pytest.raises(ValueError, match="triton"):
        resolve_training_backend("triton", cuda)


@require_cuda
def test_the_triton_kernels_give_the_loss_and_the_gradients_of_pytorch(encoders):
    pytest.importorskip("triton")
    # The fused kernel takes heads of 128 channels under BF16 autocast.
    fields = {
        "max_episode_steps": 128,
        "instruction_dropout": 0.0,
        "hidden_size": 256,
        "linear_head_dim": 128,
        "softmax_head_dim": 64,
        "device": "cuda",
    }
    batch = None
    results = []
    for backend in ("eager", "triton"):
        config = make_config(encoders, training_backend=backend, **fields)
        torch.manual_seed(1)
        policy = make_policy(config).to("cuda")
        policy.train()
        if batch is None:
            batch = training_batch(config, [70, 120], [3, 77])
            batch = {
                key: value.to("cuda") if torch.is_tensor(value) else value for (key, value) in batch.items()
            }
        torch.manual_seed(0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            (loss, _info) = policy.forward(batch)
        loss.backward()
        assert policy.training_backend == backend
        results.append(
            (loss.detach().float(), torch.cat([p.grad.float().flatten() for p in policy.parameters()]))
        )
    ((eager_loss, eager_gradient), (triton_loss, triton_gradient)) = results
    assert torch.isfinite(triton_gradient).all()
    assert torch.allclose(eager_loss, triton_loss, rtol=1e-2)
    assert torch.nn.functional.cosine_similarity(eager_gradient, triton_gradient, dim=0) > 0.99


LIBERO_VIEW_CHANGES = {
    "clean_probability": 0.0,
    "crop_area": [0.75, 1.0],
    "rotation": 10.0,
    "translation": 0.1,
    "brightness": 0.3,
    "contrast": 0.3,
    "saturation": 0.3,
    "hue": 0.03,
    "gamma": 0.2,
}


def test_augmentation_changes_a_window_once_for_all_its_frames(encoders):
    keys = make_config(encoders).camera_keys
    camera_move = {
        "views": [keys[0]],
        "probability": 1.0,
        "yaw": 3.0,
        "pitch": 2.5,
        "roll": 0.75,
        "focal": [0.96, 1.06],
        "shift_x": 4.0,
        "shift_y": 3.0,
        "radial": [-0.015, 0.03],
    }
    augmentation = {
        "camera": {"views": [keys[1]], "probability": 1.0, "scale": [0.8, 0.9]},
        "lighting": {
            "gain_probability": 1.0,
            "gain": [0.8, 1.2],
            "shadow_probability": 1.0,
            "shadow_width": [0.2, 0.4],
            "shadow_depth": [0.2, 0.5],
        },
        "color": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "gamma": 0.1},
        "view_changes": {**LIBERO_VIEW_CHANGES, "camera_move": camera_move},
    }
    config = make_config(encoders, max_episode_steps=128, augmentation=augmentation)
    policy = make_policy(config)
    policy.train()
    generator = torch.Generator().manual_seed(0)
    # A still video per camera: every frame of a window gets the same change.
    clips = [
        (torch.rand(1, 1, 3, *size, generator=generator) * 2 - 1).repeat(1, 9, 1, 1, 1)
        for size in config.view_sizes
    ]
    torch.manual_seed(0)
    changed = policy._augment(clips, config.augmentation)
    for clip, result in zip(clips, changed, strict=True):
        assert result.shape == clip.shape and result.dtype == clip.dtype
        assert result.min() >= -1 and result.max() <= 1
        assert torch.equal(result, result[:, :1].expand_as(result))
        assert (result - clip).abs().mean() > 0.05
        levels = (result + 1) * 127.5
        assert torch.allclose(levels, levels.round(), atol=1e-3)

    clean = make_policy(
        make_config(
            encoders, augmentation={"view_changes": {**LIBERO_VIEW_CHANGES, "clean_probability": 1.0}}
        )
    )
    clean.train()
    for clip, result in zip(clips, clean._augment(clips, clean.config.augmentation), strict=True):
        assert torch.allclose(result, clip, atol=1 / 255)

    torch.manual_seed(0)
    (loss, _info) = policy.forward(training_batch(config, [70], [3]))
    assert torch.isfinite(loss) and loss > 0


def test_gradient_checkpointing_keeps_the_loss_and_the_gradients(encoders):
    batch = training_batch(make_config(encoders, max_episode_steps=128), [90], [40])
    results = []
    for checkpointing in (True, False):
        torch.manual_seed(1)
        policy = make_policy(
            make_config(encoders, max_episode_steps=128, gradient_checkpointing=checkpointing)
        )
        policy.train()
        torch.manual_seed(0)
        (loss, _info) = policy.forward(batch)
        loss.backward()
        results.append((loss.detach(), [parameter.grad.clone() for parameter in policy.parameters()]))
    assert torch.allclose(results[0][0], results[1][0], atol=1e-6)
    for with_checkpointing, without in zip(results[0][1], results[1][1], strict=True):
        assert torch.allclose(with_checkpointing, without, atol=1e-5)


def test_an_episode_too_short_for_a_chunk_trains_nothing(encoders):
    config = make_config(encoders, max_episode_steps=128)
    policy = make_policy(config)
    policy.train()
    (loss, info) = policy.forward(training_batch(config, [4], [2]))
    assert float(loss.detach()) == 0.0 and info["chunks"] == 0.0
    loss.backward()


def test_training_reads_the_windows_of_a_lerobot_dataset(encoders, tmp_path):
    config = make_config(encoders, max_episode_steps=64, instruction_dropout=0.0)
    features = {
        key: {"dtype": "image", "shape": (64, 64, 3), "names": ["height", "width", "channels"]}
        for key in config.camera_keys
    }
    features[OBS_STATE] = {"dtype": "float32", "shape": (len(config.state_slots),), "names": None}
    features[ACTION] = {"dtype": "float32", "shape": (len(config.action_slots),), "names": None}
    dataset = LeRobotDataset.create(
        "test/sana_robot", fps=20, features=features, root=tmp_path / "data", use_videos=False
    )
    generator = np.random.default_rng(0)
    for rows in (45, 70):
        for _row in range(rows):
            frame = {
                key: generator.integers(0, 255, (64, 64, 3), dtype=np.uint8) for key in config.camera_keys
            }
            frame[OBS_STATE] = generator.random(len(config.state_slots), dtype=np.float32)
            frame[ACTION] = generator.random(len(config.action_slots), dtype=np.float32)
            dataset.add_frame({**frame, "task": "pick"})
        dataset.save_episode()
    dataset.finalize()

    dataset = LeRobotDataset(
        "test/sana_robot",
        root=tmp_path / "data",
        delta_timestamps=resolve_delta_timestamps(config, dataset.meta),
    )
    (preprocess, _postprocess) = make_pre_post_processors(config, dataset_stats=dataset.meta.stats)
    samples = [dataset[10], dataset[45 + 60]]
    batch = {
        key: torch.stack([sample[key] for sample in samples])
        for key in samples[0]
        if torch.is_tensor(samples[0][key])
    }
    batch["task"] = [sample["task"] for sample in samples]
    assert batch[config.camera_keys[0]].shape == (2, 33, 3, 64, 64)
    assert batch[OBS_STATE].shape == (2, 4, len(config.state_slots))
    assert batch[ACTION].shape == (2, 128, len(config.action_slots))
    # The first sample is row 10 of 45: 54 rows before the episode and 29 after it are padding.
    assert int(batch[f"{ACTION}_is_pad"][0].sum()) == 54 + 29

    policy = make_policy(config)
    policy.train()
    (loss, info) = policy.forward(preprocess(batch))
    # 45 rows from 22 rows before the first frame are 2 chunks; 70 rows from 4 rows before it are 3.
    assert info["chunks"] == 2.5 and torch.isfinite(loss)


def test_training_frames_keep_their_time_axis():
    step = SanaRobotObservations(["observation.images.image"], [[64, 64]], "crop")
    frames = torch.rand(2, 5, 3, 48, 80)
    window = step.observation({"observation.images.image": frames.clone()})["observation.images.image"]
    assert window.shape == (2, 5, 3, 64, 64)
    single = step.observation({"observation.images.image": frames[:, 3].clone()})["observation.images.image"]
    assert torch.equal(window[:, 3], single)


def test_the_statistics_of_a_dataset_become_the_normalization(encoders, tmp_path):
    config = SanaRobotConfig(device="cpu")
    (states, actions) = (len(config.state_slots), len(config.action_slots))
    dataset_stats = {
        OBS_STATE: {
            "mean": torch.full((states,), 9.0),
            "std": torch.full((states,), 9.0),
            "q01": torch.linspace(-1.0, 0.0, states),
            "q99": torch.linspace(1.0, 4.0, states),
        },
        ACTION: {
            "mean": torch.full((actions,), 9.0),
            "std": torch.full((actions,), 9.0),
            "q01": torch.full((actions,), -5.0),
            "q99": torch.full((actions,), 5.0),
        },
    }
    statistics = robot_statistics(config, dataset_stats)
    (low, high) = (dataset_stats[OBS_STATE]["q01"], dataset_stats[OBS_STATE]["q99"])
    # The joints and the position map their 1% and 99% quantiles to -1 and 1; the rotation and the gripper stay.
    assert torch.allclose(statistics[OBS_STATE]["mean"][:10], ((low + high) / 2)[:10])
    assert torch.allclose(statistics[OBS_STATE]["std"][:10], ((high - low) / 2)[:10])
    assert torch.equal(statistics[OBS_STATE]["mean"][10:], torch.zeros(7))
    assert torch.equal(statistics[OBS_STATE]["std"][10:], torch.ones(7))
    # An action is a target of the state's quantities: it takes the state's center and scale.
    assert torch.equal(statistics[ACTION]["mean"], statistics[OBS_STATE]["mean"][7:])
    assert torch.equal(statistics[ACTION]["std"], statistics[OBS_STATE]["std"][7:])
    # Statistics that already hold a center and a scale, as a checkpoint's, are kept.
    kept = robot_statistics(config, stats(config))
    assert all(
        torch.equal(kept[key][name], value)
        for key, entry in stats(config).items()
        for name, value in entry.items()
    )

    (preprocess, postprocess) = make_pre_post_processors(config, dataset_stats=stats(config))
    preprocess.save_pretrained(tmp_path)
    postprocess.save_pretrained(tmp_path)
    (preprocess, postprocess) = make_pre_post_processors(
        config,
        pretrained_path=str(tmp_path),
        preprocessor_overrides={"normalizer_processor": {"stats": dataset_stats}},
        postprocessor_overrides={"unnormalizer_processor": {"stats": dataset_stats}},
    )
    state = torch.rand(1, states)
    batch = {key: torch.rand(1, 3, 256, 256) for key in config.camera_keys} | {
        OBS_STATE: state.clone(),
        "task": ["pick"],
    }
    expected = (state - statistics[OBS_STATE]["mean"]) / statistics[OBS_STATE]["std"]
    assert torch.allclose(preprocess(batch)[OBS_STATE], expected, atol=1e-6)
    action = torch.rand(1, actions)
    expected = action * statistics[ACTION]["std"] + statistics[ACTION]["mean"]
    assert torch.allclose(postprocess(action.clone()), expected, atol=1e-6)

    # A fine-tune that keeps the checkpoint's normalization ignores the statistics of the dataset.
    (preprocess, _postprocess) = make_pre_post_processors(
        SanaRobotConfig(device="cpu", keep_checkpoint_statistics=True),
        pretrained_path=str(tmp_path),
        preprocessor_overrides={"normalizer_processor": {"stats": dataset_stats}},
    )
    expected = (state - stats(config)[OBS_STATE]["mean"]) / stats(config)[OBS_STATE]["std"]
    batch[OBS_STATE] = state.clone()
    assert torch.allclose(preprocess(batch)[OBS_STATE], expected, atol=1e-6)


def test_optimizer_and_scheduler_presets(encoders):
    config = make_config(encoders)
    optimizer = config.get_optimizer_preset()
    assert (optimizer.lr, optimizer.betas, optimizer.eps) == (5e-5, (0.9, 0.95), 1e-10)
    assert (optimizer.weight_decay, optimizer.grad_clip_norm) == (0.01, 1.0)
    scheduler = config.get_scheduler_preset()
    assert (scheduler.peak_lr, scheduler.decay_lr) == (5e-5, 5e-7)
    assert (scheduler.num_warmup_steps, scheduler.num_decay_steps) == (1000, 20000)

    policy = make_policy(config)
    assert SanaRobotPolicy._fsdp_wrap_modules == ["PolicyBlock"]
    groups = policy.get_optim_params()
    assert len(groups) == 1 and len(groups[0]["params"]) == len(list(policy.parameters()))
    policy = make_policy(make_config(encoders, optimizer_decay_biases_and_norms=False))
    (decayed, exempt) = policy.get_optim_params()
    biases = [parameter for name, parameter in policy.named_parameters() if name.endswith(".bias")]
    assert exempt["weight_decay"] == 0.0 and len(exempt["params"]) >= len(biases) > 0
    assert len(decayed["params"]) + len(exempt["params"]) == len(list(policy.parameters()))
    with pytest.raises(ValueError, match="max_episode_steps"):
        make_config(encoders, max_episode_steps=100)
