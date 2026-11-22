# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0
"""Offline checks of SANA-Robot with a reduced policy, video VAE and text encoder."""

import json
import shutil

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

from lerobot.envs.configs import LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.policies import make_pre_post_processors
from lerobot.policies.sana_robot.configuration_sana_robot import SanaRobotConfig
from lerobot.policies.sana_robot.modeling_sana_robot import SanaRobotPolicy
from lerobot.policies.sana_robot.processor_sana_robot import (
    LIBERO_JOINT_POS,
    SanaRobotLiberoAction,
    SanaRobotLiberoInstruction,
    SanaRobotLiberoObservation,
    SanaRobotObservations,
    pose_matrix,
)
from lerobot.policies.sana_robot.sana_robot_core.causal_vae import LTX2VideoEncoder3d
from lerobot.policies.sana_robot.sana_robot_core.kinematics import flange_poses
from lerobot.policies.sana_robot.sana_robot_core.libero import (
    FRANKA_FLANGE,
    FRANKA_FLANGE_AXES,
    FRANKA_JOINTS,
    grip_site_pose,
    hand_pose,
)
from lerobot.policies.sana_robot.sana_robot_core.robot80 import rotation_to_rot6d
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.rotation import Rotation

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


def make_config(encoders, **overrides):
    arguments = {
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
    return SanaRobotConfig(**arguments)


def make_policy(config):
    """A policy whose weights are all random: a new policy predicts no action flow, so its actions are its noise."""
    policy = SanaRobotPolicy(config)
    generator = torch.Generator().manual_seed(0)
    for parameter in policy.parameters():
        parameter.data.add_(torch.randn(parameter.shape, generator=generator) * 0.1)
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


def test_training_and_batches_are_rejected(encoders):
    config = make_config(encoders)
    policy = make_policy(config)
    with pytest.raises(NotImplementedError):
        policy.forward({})
    with pytest.raises(NotImplementedError):
        policy.predict_action_chunk({})
    with pytest.raises(NotImplementedError):
        config.get_optimizer_preset()
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


def libero_observation(joints, fingers, world_from_base):
    """The observation LIBERO gives for joint angles [7], finger positions [2] and an arm base at ``world_from_base``."""
    flange = flange_poses(FRANKA_JOINTS, joints[None], FRANKA_FLANGE, FRANKA_FLANGE_AXES)[0]
    site = world_from_base @ flange @ grip_site_pose()
    hand = world_from_base @ flange @ hand_pose()
    state = np.concatenate((site[:3, 3], Rotation.from_matrix(hand[:3, :3]).as_rotvec(), fingers))
    return (
        {
            OBS_STATE: torch.from_numpy(state).float()[None],
            LIBERO_JOINT_POS: torch.from_numpy(joints).float()[None],
        },
        flange,
        site,
    )


def test_libero_processors_move_between_the_world_and_the_arm_base():
    (pre, post) = make_env_pre_post_processors(
        LiberoEnvConfig(control_mode="absolute"), SanaRobotConfig(device="cpu")
    )
    (observe, act) = (pre.steps[0], post.steps[0])
    assert isinstance(observe, SanaRobotLiberoObservation) and isinstance(act, SanaRobotLiberoAction)
    world_from_base = pose_matrix(
        np.array([-0.66, 0.1, 0.912]), Rotation.from_rotvec(np.array([0.0, 0.0, 0.3])).as_matrix()
    )
    joints = np.array([0.1, -0.3, 0.05, -2.2, 0.02, 2.0, 0.7])
    (batch, flange, site) = libero_observation(joints, np.array([0.03, -0.03]), world_from_base)
    result = observe.observation(batch)
    state = result[OBS_STATE]
    assert state.shape == (1, 17)
    assert LIBERO_JOINT_POS not in result
    assert np.allclose(state[0, :7].numpy(), joints, atol=1e-6)
    assert np.allclose(state[0, 7:10].numpy(), flange[:3, 3], atol=1e-6)
    assert np.allclose(state[0, 10:16].numpy(), rotation_to_rot6d(flange[:3, :3]), atol=1e-6)
    assert np.isclose(state[0, 16].item(), 0.25, atol=1e-6)
    assert np.allclose(observe.frames.world_from_base, world_from_base, atol=1e-5)

    # The flange pose of the observation, sent back as an action, is the grip site's pose in the world.
    action = torch.cat((state[0, 7:16], torch.tensor([0.2])))[None]
    command = act.action(action)[0].numpy()
    assert np.allclose(command[:3], site[:3, 3], atol=1e-5)
    assert np.allclose(Rotation.from_rotvec(command[3:6]).as_matrix(), site[:3, :3], atol=1e-5)
    assert command[6] == -1.0
    assert act.action(torch.cat((state[0, 7:16], torch.tensor([0.8])))[None])[0, 6] == 1.0

    with pytest.raises(ValueError):
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


def test_a_released_checkpoint_gives_the_weights_the_robot_fields_and_the_statistics(encoders, released):
    (folder, source, meta, normalization) = released
    config = make_config(encoders, checkpoint=str(folder))
    assert (config.fps, config.embodiment, config.action_mode, config.resize) == (
        10.0,
        "test robot",
        "joint_only",
        "stretch",
    )
    assert (config.view_slots, config.view_descriptions) == ([0, 1], ["first camera", "second camera"])
    assert (config.state_slots, config.action_slots) == (meta["state_slots"], meta["action_slots"])
    assert config.robot_state_feature.shape == (8,) and config.action_feature.shape == (7,)

    for policy in (
        SanaRobotPolicy(config),
        SanaRobotPolicy.from_pretrained(folder, config=make_config(encoders)),
    ):
        assert policy.config.checkpoint == str(folder)
        for name, value in source.model.state_dict().items():
            assert torch.equal(policy.model.state_dict()[name], value), name

    (preprocess, postprocess) = make_pre_post_processors(config)
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


def test_a_saved_policy_no_longer_reads_the_released_checkpoint(encoders, released, tmp_path):
    (folder, _source, _meta, _normalization) = released
    config = make_config(encoders, checkpoint=str(folder))
    policy = SanaRobotPolicy(config)
    (preprocess, postprocess) = make_pre_post_processors(config)
    expected = rollout(policy, preprocess, postprocess, 18)
    policy.save_pretrained(tmp_path / "saved")
    preprocess.save_pretrained(tmp_path / "saved")
    postprocess.save_pretrained(tmp_path / "saved")
    shutil.rmtree(folder)

    loaded = SanaRobotPolicy.from_pretrained(tmp_path / "saved")
    assert loaded.config.checkpoint is None
    assert loaded.config.state_slots == config.state_slots and loaded.config.fps == 10.0
    (preprocess, postprocess) = make_pre_post_processors(
        loaded.config, pretrained_path=str(tmp_path / "saved")
    )
    assert torch.equal(expected, rollout(loaded, preprocess, postprocess, 18))


def test_a_pretrained_checkpoint_holds_weights_only(encoders, released):
    (folder, source, _meta, _normalization) = released
    (folder / "meta.json").unlink()
    (folder / "normalization.json").unlink()
    defaults = make_config(encoders)
    policy = SanaRobotPolicy.from_pretrained(folder, config=defaults)
    assert policy.config.state_slots == defaults.state_slots and policy.config.fps == defaults.fps
    assert torch.equal(
        policy.model.state_dict()["blocks.0.attn.proj.weight"],
        source.model.state_dict()["blocks.0.attn.proj.weight"],
    )
    (preprocess, _postprocess) = make_pre_post_processors(policy.config)
    assert not preprocess.steps[3].stats
    (folder / "second.pth").write_bytes(b"")
    with pytest.raises(ValueError):
        SanaRobotPolicy.from_pretrained(folder, config=defaults)
