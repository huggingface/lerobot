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

from lerobot.configs.sampling import EpisodeTable, SampleKey
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.inference.sync import SyncInferenceEngine
from lerobot.policies import make_pre_post_processors
from lerobot.policies.sana_robot import modeling_sana_robot
from lerobot.policies.sana_robot.configuration_sana_robot import EpisodeWindowSampler, SanaRobotConfig
from lerobot.policies.sana_robot.modeling_sana_robot import (
    SanaRobotPolicy,
    resolve_training_backend,
    window_shape,
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
from lerobot.policies.sana_robot.sana_robot_core.windows import build_windows
from lerobot.utils.collate import lerobot_collate_fn
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


def engine(policy, preprocess, postprocess):
    """The runner of scripts and simulator evaluation, on the policy's cameras and state."""
    keys = [*policy.config.camera_keys, OBS_STATE]
    return SyncInferenceEngine(policy, preprocess, postprocess, observation_keys=keys)


def rollout(policy, preprocess, postprocess, steps, seed=0, observations=None, **arguments):
    """The actions of ``steps`` control steps, one call of the runner per step."""
    runner = engine(policy, preprocess, postprocess)
    runner.reset()
    torch.manual_seed(seed)
    observations = observations or (lambda step: observation(policy.config, step, **arguments))
    return torch.cat([runner.step(observations(step), float(step)) for step in range(steps)])


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


def test_chunks_are_planned_on_the_episode_clock_from_the_executed_frames(encoders):
    config = make_config(encoders)
    policy = make_policy(config)
    spec = policy.chunk_inference_spec()
    assert spec.complete_execution
    assert spec.prediction_steps == spec.execution_steps == config.n_action_steps == 16
    # The memory reads a frame every video_frame_stride steps: after actions 3, 7, 11 and 15 of a chunk.
    for key in config.camera_keys:
        assert list(config.inference_deltas.of(key, FeatureType.VISUAL)) == [-12, -8, -4, 0]
    assert config.inference_deltas.of(OBS_STATE, FeatureType.STATE) is None

    (preprocess, postprocess) = make_pre_post_processors(config, dataset_stats=stats(config))
    planned = []
    plan = policy._plan
    policy._plan = lambda episode, state, instruction: (
        planned.append((episode.step, len(episode.window))),
        plan(episode, state, instruction),
    )[1]
    runner = engine(policy, preprocess, postprocess)
    actions = rollout(policy, preprocess, postprocess, 40)
    assert planned == [(0, 8), (16, 8), (32, 8)]
    assert actions.shape == (40, len(config.action_slots))
    assert torch.isfinite(actions).all()
    runner.reset()
    for step in range(40):
        runner.step(observation(config, step), float(step))
    assert runner.queued_actions == 8
    assert all(name.startswith("model.") for name in policy.state_dict())


def test_actions_repeat_after_reset_and_depend_on_the_memory(encoders):
    config = make_config(encoders)
    policy = make_policy(config)
    (preprocess, postprocess) = make_pre_post_processors(config, dataset_stats=stats(config))
    first = rollout(policy, preprocess, postprocess, 36)
    assert torch.equal(first, rollout(policy, preprocess, postprocess, 36))
    assert not torch.equal(first, rollout(policy, preprocess, postprocess, 36, seed=1))

    # The chunk generated at step 32 reads the frames before it: other frames in between give other actions.
    def other_frames(step):
        frames = observation(config, step if step < 4 or step >= 32 else step + 100)
        return {**observation(config, step), **{key: frames[key] for key in config.camera_keys}}

    changed = rollout(policy, preprocess, postprocess, 36, observations=other_frames)
    assert torch.equal(changed[:4], first[:4])
    assert not torch.equal(changed[-1:], first[-1:])


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
    changed = rollout(
        policy,
        preprocess,
        postprocess,
        20,
        observations=lambda step: observation(config, step, task="pick" if step < 8 else "place"),
    )
    assert torch.equal(same[:16], changed[:16])
    assert not torch.equal(same[16:], changed[16:])


def test_every_environment_of_a_batch_has_its_own_episode(encoders):
    config = make_config(encoders)
    policy = make_policy(config)
    (preprocess, postprocess) = make_pre_post_processors(config, dataset_stats=stats(config))

    def two_environments(step):
        (first, second) = (observation(config, step), observation(config, step + 1000, task="place"))
        batch = {key: torch.cat([first[key], second[key]]) for key in [*config.camera_keys, OBS_STATE]}
        return {**batch, "task": ["pick", "place"]}

    torch.manual_seed(0)
    both = rollout(policy, preprocess, postprocess, 20, observations=two_environments)
    assert both.shape == (40, len(config.action_slots)) and torch.isfinite(both).all()
    # The first environment's first chunk draws its noise first, as when it runs alone.
    alone = rollout(policy, preprocess, postprocess, 16)
    assert torch.equal(both[0:32:2], alone)

    # Missing frames would leave the memory behind the robot: the policy refuses the call.
    policy.reset()
    batch = preprocess(observation(config, 0))
    policy.predict_action_chunk(batch)
    batch = {**batch, **{key: batch[key][:, None].repeat(1, 3, 1, 1, 1) for key in config.camera_keys}}
    with pytest.raises(ValueError, match="frames of the previous chunk"):
        policy.predict_action_chunk(batch)
    with pytest.raises(ValueError):
        SanaRobotPolicy(make_config(encoders, vae_path=None)).predict_action_chunk(
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
    restored = postprocess({ACTION: action.clone()})[ACTION]
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
    assert torch.allclose(postprocess({ACTION: action.clone()})[ACTION], restored, atol=1e-6)
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


def training_batch(config, episode_rows, head_rows, seed=0):
    """A batch of training samples after the preprocessor, collated as the training loader does: the window of
    `config.training_sampler` that opens ``head_rows`` rows before each episode."""
    generator = torch.Generator().manual_seed(seed)
    sampler = config.training_sampler
    episodes = EpisodeTable(np.arange(len(episode_rows)), np.asarray(episode_rows))
    samples = []
    for episode, (rows, head) in enumerate(zip(episode_rows, head_rows, strict=True)):
        index = [window_head for (window_head, _chunks) in sampler.windows(rows)].index(head)
        sample = sampler.sample(episodes, SampleKey(episode, index))
        windows = {}
        for key, size in zip(config.camera_keys, config.view_sizes, strict=True):
            frames = torch.rand(rows, 3, *size, generator=generator) * 2 - 1
            windows[key] = loader_window(frames, sample.anchor, sample.deltas.of(key, FeatureType.VISUAL))
        state = torch.rand(rows, len(config.state_slots), generator=generator)
        windows[OBS_STATE] = loader_window(
            state, sample.anchor, sample.deltas.of(OBS_STATE, FeatureType.STATE)
        )
        action = torch.rand(rows, len(config.action_slots), generator=generator)
        windows[ACTION] = loader_window(action, sample.anchor, sample.deltas.action)
        samples.append(
            {
                **{key: values for key, (values, _padding) in windows.items()},
                **{f"{key}_is_pad": padding for key, (_values, padding) in windows.items()},
                "task": "pick",
            }
        )
    return lerobot_collate_fn(samples)


def test_the_sampler_gives_the_windows_of_the_original_trainer():
    # Every episode once per chunk phase, as whole chunks while a chunk has a real first target frame.
    lengths = list(range(1, 400))
    (episodes, heads, chunks) = build_windows(lengths, 32, 4)
    sampler = EpisodeWindowSampler(32, 4)
    windows = [(e, h, n) for e, rows in enumerate(lengths) for (h, n) in sampler.windows(rows)]
    assert windows == list(zip(episodes.tolist(), heads.tolist(), chunks.tolist(), strict=True))
    table = EpisodeTable(np.arange(3), np.array([4, 70, 120]))
    # An episode too short for a trained chunk gives no window.
    assert [sampler.count(table, episode) for episode in range(3)] == [0, 32, 32]
    sample = sampler.sample(table, SampleKey(1, 29))
    # 70 rows from 29 rows before the first frame are 3 chunks.
    assert sample.anchor == -29
    assert sample.deltas.action == tuple(range(96))
    assert sample.deltas.of("observation.images.image", FeatureType.VISUAL) == tuple(range(0, 97, 4))
    assert sample.deltas.of(OBS_STATE, FeatureType.STATE) == (0, 32, 64)


@pytest.mark.parametrize("head_rows", [0, 1, 19, 29, 31])
def test_a_window_is_read_back_from_its_padding_after_the_collate(head_rows):
    sampler = EpisodeWindowSampler(32, 4)
    for rows in (5, 36, 40, 100, 150):
        chunks = dict(sampler.windows(rows)).get(head_rows)
        if chunks is None:
            continue
        positions = torch.arange(chunks * 32) - head_rows
        padding = (positions < 0) | (positions >= rows)
        # The collate pads a window to the batch's longest one.
        padded = torch.cat([padding, torch.ones(64, dtype=torch.bool)])
        assert window_shape(padding, 32, 4) == window_shape(padded, 32, 4) == (head_rows, chunks)


def test_training_loss_of_whole_episodes(encoders):
    config = make_config(encoders, instruction_dropout=0.5)
    policy = make_policy(config)
    policy.train()
    batch = training_batch(config, [70, 120], [29, 19])
    # Each sample holds its own window, padded to the longest of the batch.
    assert batch[ACTION].shape == (2, 160, len(config.action_slots))
    assert batch[config.camera_keys[0]].shape == (2, 41, 3, 64, 64)
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

    with pytest.raises(ValueError, match="window of actions"):
        policy.forward({**batch, ACTION: batch[ACTION][:, 0]})


def test_a_packed_row_computes_every_window_as_alone(encoders, monkeypatch):
    config = make_config(encoders, instruction_dropout=0.0)
    policy = make_policy(config)
    policy.train()
    batch = training_batch(config, [70, 120, 40], [29, 19, 0])
    batch["task"] = ["pick", "place", "pick"]
    model_forward = policy.model.forward
    calls = []

    def recorded_forward(**arguments):
        calls.append((arguments, model_forward(**arguments)))
        return calls[-1][1]

    monkeypatch.setattr(policy.model, "forward", recorded_forward)
    torch.manual_seed(0)
    (losses, info) = policy.forward(batch, reduction="none")
    # One pass of the model for the three windows.
    assert len(calls) == 1 and info["chunks"] == 10 / 3
    assert (losses > 0).all()
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
            batch = training_batch(config, [70, 120], [29, 19])
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
    config = make_config(encoders, augmentation=augmentation)
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
    (loss, _info) = policy.forward(training_batch(config, [70], [29]))
    assert torch.isfinite(loss) and loss > 0


def test_gradient_checkpointing_keeps_the_loss_and_the_gradients(encoders):
    batch = training_batch(make_config(encoders), [90], [24])
    results = []
    for checkpointing in (True, False):
        torch.manual_seed(1)
        policy = make_policy(make_config(encoders, gradient_checkpointing=checkpointing))
        policy.train()
        torch.manual_seed(0)
        (loss, _info) = policy.forward(batch)
        loss.backward()
        results.append((loss.detach(), [parameter.grad.clone() for parameter in policy.parameters()]))
    assert torch.allclose(results[0][0], results[1][0], atol=1e-6)
    for with_checkpointing, without in zip(results[0][1], results[1][1], strict=True):
        assert torch.allclose(with_checkpointing, without, atol=1e-5)


def test_training_reads_the_windows_of_a_lerobot_dataset(encoders, tmp_path):
    config = make_config(encoders, instruction_dropout=0.0)
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

    dataset = LeRobotDataset("test/sana_robot", root=tmp_path / "data", sampler=config.training_sampler)
    # Every episode gives one window per chunk phase.
    assert dataset.sample_counts == {0: 32, 1: 32} and len(dataset) == 64
    (preprocess, _postprocess) = make_pre_post_processors(config, dataset_stats=dataset.meta.stats)
    # Windows of 45 rows from 22 rows before the first frame (2 chunks) and of 70 rows from 4 (3 chunks).
    batch = lerobot_collate_fn(dataset.__getitems__([SampleKey(0, 22), SampleKey(1, 4)]))
    assert batch[config.camera_keys[0]].shape == (2, 25, 3, 64, 64)
    assert batch[OBS_STATE].shape == (2, 3, len(config.state_slots))
    assert batch[ACTION].shape == (2, 96, len(config.action_slots))
    # The first window: 22 rows before the episode, and the collate pads it from 64 to 96 rows.
    assert int(batch[f"{ACTION}_is_pad"][0].sum()) == 22 + 32
    episode = dataset.__getitems__([SampleKey(0, 0)])[0]
    torch.testing.assert_close(batch[ACTION][0, 22:64], episode[ACTION][:42])

    policy = make_policy(config)
    policy.train()
    (loss, info) = policy.forward(preprocess(batch))
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
    assert torch.allclose(postprocess({ACTION: action.clone()})[ACTION], expected, atol=1e-6)

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
