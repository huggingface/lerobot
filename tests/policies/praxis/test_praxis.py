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

"""CPU tests of the Praxis-1 adapter on a tiny random model.

The parity tests drive the LeRobot policy and the praxis repository's ``Session``
(the ``praxis`` package, when it is installed) with the same weights, the same fake
frozen parts and the same observations, and require bitwise equal commands.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.praxis.configuration_praxis import (
    PRAXIS_ACTION_DIM,
    PRAXIS_COMMAND_DIM,
    PRAXIS_POSE_DIM,
    PraxisConfig,
)
from lerobot.policies.praxis.modeling_praxis import OBS_SCENE_SEED, PraxisPolicy
from lerobot.policies.praxis.praxis_core.checkpoint import save_checkpoint, verify_checksums
from lerobot.policies.praxis.processor_praxis import droid7_to_pose8, pose8_to_droid7, pose12_grip_to_droid7
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.rotation import Rotation

from .tiny_model import (
    FakeAutoencoder,
    FakeTextEncoder,
    fake_observation,
    make_tiny_policy,
    tiny_lerobot_config,
    tiny_praxis_config,
)

pytest.importorskip("transformers", reason="praxis requires the praxis extra (transformers)")

REPO_ROOT = Path(__file__).resolve().parents[3]
#: LeRobot clients report the flange pose itself; the praxis session needs the frame under a descriptor.
LEROBOT_FRAME = "lerobot.flange"


def _converter():
    """The checkpoint converter of ``examples/praxis`` (a script, imported by path)."""
    import importlib.util

    path = REPO_ROOT / "examples" / "praxis" / "convert_checkpoint.py"
    spec = importlib.util.spec_from_file_location("praxis_convert_checkpoint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _praxis_session_cls():
    """The praxis repository's ``Session`` (the reference of the parity tests), or skip."""
    pipeline = pytest.importorskip("praxis.pipeline", reason="the praxis package gives the reference Session")
    single_arm = pytest.importorskip("praxis.single_arm")
    single_arm.RIG_FRAMES.setdefault(
        LEROBOT_FRAME, single_arm.RigFrame(0.0, single_arm._rows(single_arm.NO_TURN))
    )
    return pipeline.Session


class _ReferencePipeline:
    """What the praxis repository's ``PraxisPipeline`` exposes to a ``Session``, with the same parts."""

    def __init__(self, policy: PraxisPolicy) -> None:
        self.policy = policy.model
        self.vae = FakeAutoencoder()
        self.text = FakeTextEncoder()
        self.device = torch.device("cpu")
        self.stats = policy.stats
        self.variant = policy.variant
        self.sampler = policy.sampler_settings
        self.embodiment = "single_arm"
        self._cache: dict[str, torch.Tensor] = {}

    def encode_text(self, prompt: str) -> torch.Tensor:
        if prompt not in self._cache:
            self._cache[prompt] = self.text.encode([prompt])[0]
        return self._cache[prompt]

    def encode_uncond_text(self) -> torch.Tensor:
        return self.encode_text(self.variant.blank_caption)


def _reference_obs(observations: list[dict]) -> dict:
    """The praxis observation stack of a list of LeRobot observations (the last 25)."""
    rows = observations[-25:]
    ext = np.stack(
        [
            (o["observation.images.exterior"][0] * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
            for o in rows
        ]
    )
    wrist = np.stack(
        [
            (o["observation.images.wrist"][0] * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
            for o in rows
        ]
    )
    pose8 = np.stack([droid7_to_pose8(o[OBS_STATE][0].numpy()) for o in rows])
    return {
        "image.exterior": ext,
        "image.wrist": wrist,
        "robot_state.ee_pose": pose8[:, :7],
        "grip": pose8[:, 7:8],
        "descriptor": LEROBOT_FRAME,
        "task": rows[-1]["task"][0],
    }


# ---------------------------------------------------------------------------
# Registration and config
# ---------------------------------------------------------------------------


def test_factory_registers_praxis() -> None:
    assert get_policy_class("praxis") is PraxisPolicy
    config = make_policy_config("praxis")
    assert isinstance(config, PraxisConfig)
    assert config.chunk_size == 24 and config.num_steps == 2 and config.max_context_chunks == 1
    # the recommended settings of `praxis-1-droid`
    assert (config.t_shift, config.cfg_scale, config.cfg_video_scale) == (12.5, 1.0, 4.0)


def test_config_round_trip_keeps_the_praxis_config(tmp_path: Path) -> None:
    config = tiny_lerobot_config(num_steps=3)
    config._save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path, cli_overrides=["--cfg_scale=1.0"])
    assert isinstance(loaded, PraxisConfig)
    assert loaded.model_config == tiny_praxis_config()
    assert loaded.num_steps == 3 and loaded.cfg_scale == 1.0


def test_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="attention_backend"):
        tiny_lerobot_config(attention_backend="sdpa")
    with pytest.raises(ValueError, match="n_action_steps"):
        tiny_lerobot_config(n_action_steps=25)
    with pytest.raises(ValueError, match="max_context_chunks"):
        tiny_lerobot_config(max_context_chunks=4)
    config = tiny_lerobot_config()
    config.input_features.pop("observation.images.wrist")
    with pytest.raises(ValueError, match="camera inputs"):
        config.validate_features()


def test_processors_are_identity(tmp_path: Path) -> None:
    from lerobot.policies.praxis.praxis_core.single_arm import _tile_view
    from lerobot.processor import PolicyProcessorPipeline

    config = tiny_lerobot_config()
    pre, post = make_pre_post_processors(config, None)
    obs = fake_observation(0)
    out = pre(obs)
    # cameras: each view is its tile (32 x 64 for the tiny checkpoint), the bytes of the praxis canvas
    tile = _tile_view(_to_hwc_uint8(obs["observation.images.exterior"][0]), 32, 64, 0.03)[0].float() / 255
    assert torch.equal(out["observation.images.exterior"][0], tile)
    assert out["observation.images.wrist"].shape == (1, 3, 32, 64)
    assert torch.equal(
        pre(out)["observation.images.exterior"], out["observation.images.exterior"]
    )  # idempotent
    # state: the DROID pose becomes the praxis pose8
    state7 = obs[OBS_STATE].numpy()
    assert torch.equal(out[OBS_STATE], torch.from_numpy(droid7_to_pose8(state7)))
    assert out[OBS_STATE].shape == (1, PRAXIS_POSE_DIM)
    # command: the wire pose and the gripper become the DROID 7 values
    rot = Rotation.euler_xyz_to_matrix(np.array([[3.0, 0.1, 0.2]]))[0]
    command = torch.from_numpy(np.concatenate([[0.5, 0.0, 0.4], rot.reshape(-1), [0.25]]).astype(np.float32))[
        None
    ]
    want = pose12_grip_to_droid7(command[0, :12].numpy(), 0.25)
    np.testing.assert_array_equal(post(command)[0].numpy(), want)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    assert (tmp_path / "policy_preprocessor.json").is_file()
    assert (tmp_path / "policy_postprocessor.json").is_file()
    loaded = PolicyProcessorPipeline.from_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    assert [type(step).__name__ for step in loaded.steps] == [type(step).__name__ for step in pre.steps]
    assert torch.equal(loaded(obs)[OBS_STATE], out[OBS_STATE])


def test_tile_step_follows_the_checkpoint_antialias() -> None:
    """With `video.antialias` the tile step applies the fit rule of `praxis-1-droid`, bit for bit."""
    from lerobot.policies.praxis.praxis_core.single_arm import _tile_view

    model_config = tiny_praxis_config()
    model_config["video"]["antialias"] = True
    config = tiny_lerobot_config(image_shape=(3, 72, 128), model_config=model_config)
    pre, _ = make_pre_post_processors(config, None)
    obs = fake_observation(0, image_shape=(3, 72, 128))
    out = pre(obs)
    frame = _to_hwc_uint8(obs["observation.images.wrist"][0])
    smooth = _tile_view(frame, 32, 64, 0.03, antialias=True)[0].float() / 255
    assert torch.equal(out["observation.images.wrist"][0], smooth)
    assert not torch.equal(smooth, _tile_view(frame, 32, 64, 0.03)[0].float() / 255)


def _to_hwc_uint8(image: torch.Tensor) -> np.ndarray:
    return (image * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
# Pose math
# ---------------------------------------------------------------------------


def test_euler_round_trip() -> None:
    rng = np.random.default_rng(0)
    rpy = np.stack(
        [
            rng.uniform(-np.pi, np.pi, 200),
            rng.uniform(-np.pi / 2 + 0.05, np.pi / 2 - 0.05, 200),
            rng.uniform(-np.pi, np.pi, 200),
        ],
        axis=1,
    )
    back = Rotation.matrix_to_euler_xyz(Rotation.euler_xyz_to_matrix(rpy))
    np.testing.assert_allclose(back, rpy, atol=1e-9)


def test_pose_round_trips() -> None:
    rows = np.array(
        [[0.5, -0.1, 0.3, 3.0, 0.2, -0.4, 0.7], [0.4, 0.2, 0.5, -2.9, -0.3, 1.2, 1.3]], dtype=np.float32
    )
    pose8 = droid7_to_pose8(rows)
    assert pose8.shape == (2, 8) and pose8.dtype == np.float32
    assert np.allclose(np.linalg.norm(pose8[:, 3:7], axis=1), 1.0, atol=1e-6)
    assert pose8[1, 7] == 1.0  # the gripper is clipped to [0, 1]
    back = pose8_to_droid7(pose8)
    np.testing.assert_allclose(back[:, :6], rows[:, :6], atol=1e-5)
    rot = Rotation.euler_xyz_to_matrix(rows[:1, 3:6])[0]
    pose12 = np.concatenate([rows[0, :3], rot.reshape(-1)]).astype(np.float32)
    row = pose12_grip_to_droid7(pose12, 0.25)
    np.testing.assert_allclose(row[:6], rows[0, :6], atol=1e-5)
    assert row[6] == np.float32(0.25)


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("antialias", [False, True])
def test_select_action_matches_the_praxis_session_bitwise(antialias: bool) -> None:
    """The LeRobot glue (images, poses, history, seeds) gives the praxis commands, bit for bit.

    With `video.antialias` (`praxis-1-droid`) the preprocessor makes the tiles with the fit rule and the policy
    feeds them to the session as they are; the praxis session gets the raw frames.
    """
    model_config = tiny_praxis_config()
    model_config["video"]["antialias"] = antialias
    image_shape = (3, 72, 128) if antialias else (3, 36, 64)
    policy = make_tiny_policy(
        seed=1, n_action_steps=24, max_context_chunks=1, model_config=model_config, image_shape=image_shape
    )
    reference = _praxis_session_cls()(
        _ReferencePipeline(policy),
        prompt="",
        seed=policy.config.seed,
        max_context_chunks=1,
        control_mode="none",
        exec_horizon=24,
    )  # type: ignore[arg-type]
    pre, post = make_pre_post_processors(policy.config, None)
    observations: list[dict] = []
    policy.reset()
    actions: list[torch.Tensor] = []
    for step in range(22):  # 21 rows of the first chunk, then the first row of the second
        obs = fake_observation(step, image_shape=image_shape)
        observations.append(obs)
        command = policy.select_action(pre(obs))
        assert command.shape == (1, PRAXIS_COMMAND_DIM)
        actions.append(post(command)[0])
    # Chunk 0 (cold start, 21 rows) and chunk 1 (depth 1, 24 rows) through the praxis session.
    res0 = reference.step(_reference_obs(observations[:1]))
    res1 = reference.step(_reference_obs(observations[:22]))
    assert len(res0.commands) == 21 and len(res1.commands) == 24
    assert res1.depth == 1
    expected = [pose12_grip_to_droid7(c["robot_command"]["pose"], c["target_grip"]) for c in res0.commands]
    expected.append(
        pose12_grip_to_droid7(res1.commands[0]["robot_command"]["pose"], res1.commands[0]["target_grip"])
    )
    for step, (got, want) in enumerate(zip(actions, expected, strict=True)):
        assert got.shape == (PRAXIS_ACTION_DIM,)
        np.testing.assert_array_equal(got.numpy(), want, err_msg=f"step {step}")
    session = policy.sessions()[0]
    assert session is not None and session.state.stream.depth == 1
    assert policy.count_queued_actions() == 23


def test_scene_seed_selects_the_sampler_seed() -> None:
    policy = make_tiny_policy(seed=2)
    policy.reset()
    obs = fake_observation(0)
    obs[OBS_SCENE_SEED] = torch.tensor([[3.0]])
    first_seeded = policy.select_action(obs)[0].clone()
    session = policy.sessions()[0]
    assert session is not None and session.seed == policy.config.seed + policy.config.seed_stride * 3
    policy.reset()
    obs.pop(OBS_SCENE_SEED)
    first_unseeded = policy.select_action(obs)[0]
    assert policy.sessions()[0].seed == policy.config.seed  # type: ignore[union-attr]
    assert not torch.equal(first_seeded, first_unseeded)


def test_batched_environments_match_single_runs() -> None:
    policy = make_tiny_policy(seed=3)
    single = []
    for env in range(2):
        policy.reset()
        obs = fake_observation(0, seed=env)
        obs[OBS_SCENE_SEED] = torch.tensor([[float(env)]])
        single.append(policy.select_action(obs)[0])
    policy.reset()
    a, b = fake_observation(0, seed=0), fake_observation(0, seed=1)
    batch = {k: torch.cat([a[k], b[k]]) if isinstance(a[k], torch.Tensor) else a[k] + b[k] for k in a}
    batch[OBS_SCENE_SEED] = torch.tensor([[0.0], [1.0]])
    out = policy.select_action(batch)
    assert out.shape == (2, PRAXIS_COMMAND_DIM)
    assert torch.equal(out[0], single[0]) and torch.equal(out[1], single[1])


def test_short_execution_horizon_is_stateless() -> None:
    policy = make_tiny_policy(seed=4, n_action_steps=8)
    policy.reset()
    for step in range(9):
        policy.select_action(fake_observation(step))
    session = policy.sessions()[0]
    assert session is not None and session.state.stateless
    assert session.state.stream.depth == 0


def test_predict_action_chunk_pads_the_first_chunk() -> None:
    policy = make_tiny_policy(seed=5)
    policy.reset()
    chunk = policy.predict_action_chunk(fake_observation(0))
    assert chunk.shape == (1, 24, PRAXIS_COMMAND_DIM)
    assert torch.equal(chunk[0, 20], chunk[0, 23])  # 21 real rows, padded with the last one


def test_reset_and_drop_queued_actions() -> None:
    policy = make_tiny_policy(seed=6)
    policy.reset()
    policy.select_action(fake_observation(0))
    assert policy.count_queued_actions() == 20
    session = policy.sessions()[0]
    assert session is not None and session.state.restart_count == 0
    policy.drop_queued_actions()
    assert policy.count_queued_actions() == 0
    assert session.state.restart_count == 1  # the chunk was cut short: the next call is a cold start
    policy.select_action(fake_observation(1))
    assert policy.count_queued_actions() == 20 and session.state.stream.depth == 0
    policy.drop_queued_actions()
    policy.drop_queued_actions()  # nothing queued: nothing to restart
    assert session.state.restart_count == 2
    policy.reset()
    assert policy.sessions() == []


def test_return_prediction_gives_frames_for_both_cameras() -> None:
    policy = make_tiny_policy(seed=7, return_prediction=True)
    policy.reset()
    out = policy.select_action(fake_observation(0))
    assert isinstance(out, dict) and ACTION in out and "prediction" in out
    frames = out["prediction"]["observation"]
    assert set(frames) == {"observation.images.exterior", "observation.images.wrist"}
    assert frames["observation.images.exterior"].shape[1:] == (3, 32, 64)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def test_convert_and_load_pretrained(tmp_path: Path) -> None:
    convert = _converter().convert

    policy = make_tiny_policy(seed=8)
    src = tmp_path / "praxis"
    save_checkpoint(src, policy.model.state_dict(), policy.config.model_config)
    assert (src / "config.json").is_file()
    out = convert(
        src,
        tmp_path / "lerobot",
        link=False,
        device="cpu",
        attention_backend="eager",
        image_shape=(3, 36, 64),
    )
    assert (out / "config.json").is_file() and (out / "policy_preprocessor.json").is_file()
    assert json.loads((out / "config.json").read_text())["type"] == "praxis"
    verify_checksums(out)  # the directory's own list: the shards, the index and the LeRobot config
    loaded = _load_without_frozen_parts(out)
    for (name, p), (_, q) in zip(
        policy.model.state_dict().items(), loaded.model.state_dict().items(), strict=True
    ):
        assert torch.equal(p, q), name
    loaded.vae, loaded.text_encoder = FakeAutoencoder(), FakeTextEncoder()
    policy.reset(), loaded.reset()
    obs = fake_observation(0)
    assert torch.equal(policy.select_action(obs), loaded.select_action(obs))


def _load_without_frozen_parts(path: Path) -> PraxisPolicy:
    """``PraxisPolicy.from_pretrained`` without the download of the Wan2.2 VAE and Qwen3.5."""
    config = PreTrainedConfig.from_pretrained(path)
    with torch.device("meta"):
        policy = PraxisPolicy(config)
    PraxisPolicy._load_weights(policy, PraxisPolicy._resolve_weight_files(str(path)), torch.device("cpu"))
    return policy.eval()


def test_save_pretrained_writes_shards_that_load_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``save_pretrained`` writes the transformers layout (policy-prefixed names); it loads bitwise."""
    policy = make_tiny_policy(seed=11)
    monkeypatch.setattr(PraxisPolicy, "_max_shard_size", 200_000)  # bytes: several shards for the tiny model
    out = tmp_path / "saved"
    policy.save_pretrained(out)
    shards = sorted(out.glob("model-*-of-*.safetensors"))
    assert (out / "model.safetensors.index.json").is_file() and len(shards) >= 2
    assert not (out / "model.safetensors").exists()
    loaded = _load_without_frozen_parts(out)
    for (name, p), (_, q) in zip(policy.state_dict().items(), loaded.state_dict().items(), strict=True):
        assert torch.equal(p, q), name
    loaded.vae, loaded.text_encoder = FakeAutoencoder(), FakeTextEncoder()
    policy.reset(), loaded.reset()
    obs = fake_observation(3)
    assert torch.equal(policy.select_action(obs), loaded.select_action(obs))


def test_from_pretrained_leaves_the_weights_trainable_and_honours_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loading does not freeze the weights; a tensor the model lacks is refused only under ``strict``."""
    from safetensors.torch import load_file, save_file

    monkeypatch.setattr(PraxisPolicy, "_load_frozen_parts", lambda self: None)
    policy = make_tiny_policy(seed=12)
    out = tmp_path / "saved"
    policy.save_pretrained(out)
    config = PreTrainedConfig.from_pretrained(out)
    loaded = PraxisPolicy.from_pretrained(out, config=config)
    assert all(p.requires_grad for p in loaded.model.parameters())
    weights = load_file(str(out / "model.safetensors"))
    weights["model.extra.weight"] = torch.zeros(2)
    save_file(weights, str(out / "model.safetensors"), metadata={"format": "pt"})
    PraxisPolicy.from_pretrained(out, config=config, strict=False)
    with pytest.raises(RuntimeError, match="1 unexpected"):
        PraxisPolicy.from_pretrained(out, config=config, strict=True)
    del weights["model.extra.weight"], weights[next(k for k in weights if k.endswith(".weight"))]
    save_file(weights, str(out / "model.safetensors"), metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="1 missing"):
        PraxisPolicy.from_pretrained(out, config=config, strict=False)


# ---------------------------------------------------------------------------
# Text encoder weight repair (transformers >= 5.4 reports Qwen3.5 text tensors as missing)
# ---------------------------------------------------------------------------


def test_repair_language_model_weights(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    from lerobot.policies.praxis.praxis_core.text_encoder import _repair_language_model_weights

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = torch.nn.Linear(4, 3)
            self.visual = torch.nn.Linear(2, 2)

    model = Tiny()
    stored_w, stored_b = torch.arange(12.0).view(3, 4), torch.tensor([1.0, 2.0, 3.0])
    save_file(
        {
            "model.language_model.weight": stored_w,
            "model.language_model.bias": stored_b,
            "model.visual.weight": torch.ones(2, 2),
        },
        str(tmp_path / "model-00001-of-00001.safetensors"),
    )
    _repair_language_model_weights(
        model, tmp_path, {"missing_keys": ["language_model.weight", "language_model.bias"]}
    )
    assert torch.equal(model.language_model.weight.data, stored_w)
    assert torch.equal(model.language_model.bias.data, stored_b)
    _repair_language_model_weights(model, tmp_path, {"missing_keys": []})  # nothing to do
    with pytest.raises(RuntimeError, match="missing from the checkpoint"):
        _repair_language_model_weights(model, tmp_path, {"missing_keys": ["language_model.nope"]})


@pytest.mark.parametrize("host", ["tf32_flags", "precision_high", "backend_tf32"])
def test_numerics_guard_pins_and_restores_the_host_settings(host: str) -> None:
    """The guard pins the praxis numerics and gives the host its settings back, however they were
    set: the legacy TF32 flags (lerobot-eval `--tf32`), `set_float32_matmul_precision`, or the
    per-backend `fp32_precision` of PyTorch 2.9+ (which the global switch cannot report)."""
    from lerobot.policies.praxis.modeling_praxis import _praxis_numerics

    cudnn, matmul = torch.backends.cudnn, torch.backends.cuda.matmul
    saved = (cudnn.benchmark, cudnn.allow_tf32, torch.get_float32_matmul_precision())
    try:
        cudnn.benchmark, cudnn.allow_tf32 = True, False
        if host == "tf32_flags":
            matmul.allow_tf32 = True
        elif host == "precision_high":
            torch.set_float32_matmul_precision("high")
        else:
            matmul.fp32_precision = "tf32"  # legacy `get_float32_matmul_precision` now raises
            with pytest.raises(RuntimeError):
                torch.get_float32_matmul_precision()
        with _praxis_numerics():
            assert (cudnn.benchmark, cudnn.allow_tf32, matmul.allow_tf32) == (False, True, False)
            assert torch.get_float32_matmul_precision() == "highest"
            assert matmul.fp32_precision == "ieee"
        assert (cudnn.benchmark, cudnn.allow_tf32) == (True, False)
        assert matmul.fp32_precision == "tf32"
        if host != "backend_tf32":
            assert torch.get_float32_matmul_precision() == "high"
        # Whatever the host did, the guard leaves torch able to report the global precision again
        # after the host's own reset, and a second guard still runs.
        torch.set_float32_matmul_precision("highest")
        with _praxis_numerics():
            pass
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        cudnn.benchmark, cudnn.allow_tf32 = saved[:2]
        torch.set_float32_matmul_precision(saved[2])


def test_from_pretrained_applies_the_requested_dtype_to_the_weights_only(tmp_path: Path) -> None:
    """``config.dtype`` casts the parameters; the float32 buffers (routing biases) keep their dtype."""
    convert = _converter().convert

    policy = make_tiny_policy(seed=12)
    src = tmp_path / "praxis"
    save_checkpoint(src, policy.model.state_dict(), policy.config.model_config)
    out = convert(
        src,
        tmp_path / "lerobot",
        link=False,
        device="cpu",
        attention_backend="eager",
        image_shape=(3, 36, 64),
    )
    for dtype in (torch.float32, torch.bfloat16):
        config = PreTrainedConfig.from_pretrained(out)
        config.dtype = dtype
        with torch.device("meta"):
            loaded = PraxisPolicy(config)
        PraxisPolicy._load_weights(loaded, PraxisPolicy._resolve_weight_files(str(out)), torch.device("cpu"))
        assert {p.dtype for p in loaded.model.parameters()} == {dtype}
        biases = [b for n, b in loaded.model.named_buffers() if n.endswith("expert_bias")]
        assert biases and all(b.dtype == torch.float32 for b in biases)
        for (name, p), (_, q) in zip(
            policy.model.state_dict().items(), loaded.model.state_dict().items(), strict=True
        ):
            assert torch.equal(p.to(q.dtype), q), name


def test_frozen_parts_use_the_loading_options(monkeypatch: pytest.MonkeyPatch) -> None:
    import lerobot.policies.praxis.modeling_praxis as modeling

    calls: list[tuple] = []
    monkeypatch.setattr(
        modeling,
        "hf_hub_download",
        lambda repo, filename, **kw: calls.append(("file", repo, filename, kw)) or "/tmp/x",
    )
    monkeypatch.setattr(
        modeling, "snapshot_download", lambda repo, **kw: calls.append(("dir", repo, kw)) or "/tmp/y"
    )
    options = {"cache_dir": "/cache", "local_files_only": True, "token": "t"}
    assert modeling._resolve_weight_file("org/vae:weights.pth", "default.pth", **options) == "/tmp/x"
    assert modeling._resolve_model_dir("org/text", **options) == "/tmp/y"
    assert calls == [("file", "org/vae", "weights.pth", options), ("dir", "org/text", options)]


def test_kernels_import_without_triton() -> None:
    """The kernel modules import on hosts without Triton (macOS, CPU CI); the fast paths stay off."""
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['triton'] = None; sys.modules['triton.language'] = None; "
        "from lerobot.policies.praxis.praxis_core.kernels import layers_triton as l, vae_encoder_triton as v; "
        "assert not l.HAS_TRITON and not v.HAS_TRITON; "
        "from lerobot.policies.praxis.praxis_core.text_encoder import pin_fla_kernels; "
        "assert pin_fla_kernels() == {}; print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr[-2000:]


def test_one_instruction_row_serves_every_environment() -> None:
    """A single task string tokenizes to one row; every environment of the batch takes it."""
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    policy = make_tiny_policy(seed=5)
    policy.reset()
    obs = fake_observation(0)
    batch = {k: (torch.cat([v, v]) if isinstance(v, torch.Tensor) else v) for k, v in obs.items()}
    batch["task"] = "Pick up the red cube."  # one string for two environments
    batch[OBS_LANGUAGE_TOKENS] = torch.tensor([[11, 12, 13, 0]])
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.tensor([[True, True, True, False]])
    out = policy.select_action(batch)
    assert out.shape == (2, PRAXIS_COMMAND_DIM)
    assert all(env.text is not None for env in policy.sessions())
    assert torch.equal(policy.sessions()[0].text, policy.sessions()[1].text)


def test_predict_action_chunk_takes_the_executed_frames_as_a_stack() -> None:
    """A caller that executes whole chunks hands over the executed frames as a stack; the next
    chunk then equals the one of a call-per-frame rollout (same history, same context)."""
    policy = make_tiny_policy(seed=7)
    policy.reset()
    rows = [policy.select_action(fake_observation(s))[0] for s in range(45)]  # chunk 0: 21 rows, chunk 1: 24
    chunked = make_tiny_policy(seed=7)
    chunked.reset()
    first = chunked.predict_action_chunk(fake_observation(0))[0]
    assert torch.equal(first[:21], torch.stack(rows[:21]))
    since = [fake_observation(s) for s in range(1, 22)]  # the observations since the first call
    stack = {
        key: torch.stack([obs[key][0] for obs in since])[None] if isinstance(value, torch.Tensor) else value
        for key, value in since[0].items()
    }
    second = chunked.predict_action_chunk(stack)[0]
    assert torch.equal(second, torch.stack(rows[21:45]))
    assert chunked.sessions()[0].state.stream.depth == 1  # the context carried over
    # the state of every frame is needed with a frame stack
    bad = dict(stack, **{OBS_STATE: stack[OBS_STATE][:, :3]})
    with pytest.raises(ValueError, match="state of every frame"):
        chunked.predict_action_chunk(bad)


def test_tokenizer_step_strips_the_instruction() -> None:
    """The step tokenizes the instruction as the praxis session and data module hand it to the text
    encoder: stripped of surrounding whitespace, a blank one as a single space."""
    from lerobot.lerobot_types import TransitionKey
    from lerobot.policies.praxis.processor_praxis import PraxisTokenizerProcessorStep
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    seen: list[str] = []

    class FakeTokenizer:
        def __call__(self, text: list[str], **kwargs) -> dict[str, torch.Tensor]:
            seen.extend(text)
            width = max(len(t) for t in text)
            return {
                "input_ids": torch.ones(len(text), width, dtype=torch.int64),
                "attention_mask": torch.ones(len(text), width, dtype=torch.int64),
            }

    step = PraxisTokenizerProcessorStep(tokenizer_id="unused")
    step.tokenizer = FakeTokenizer()
    out = step(
        {
            TransitionKey.OBSERVATION: {},
            TransitionKey.COMPLEMENTARY_DATA: {"task": [" Pick up the cube.\n", "  "]},
        }
    )
    assert seen == ["Pick up the cube.", " "]
    assert out[TransitionKey.OBSERVATION][OBS_LANGUAGE_TOKENS].dtype == torch.int64
    assert out[TransitionKey.OBSERVATION][OBS_LANGUAGE_ATTENTION_MASK].dtype == torch.bool
    out = step({TransitionKey.OBSERVATION: {}, TransitionKey.COMPLEMENTARY_DATA: {"task": "Open it.  "}})
    assert seen[-1] == "Open it." and out[TransitionKey.OBSERVATION][OBS_LANGUAGE_TOKENS].shape[0] == 1


def test_token_rows_must_be_one_or_one_per_environment() -> None:
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS

    policy = make_tiny_policy(seed=9)
    policy.reset()
    obs = fake_observation(0)
    batch = {k: (torch.cat([v, v, v]) if isinstance(v, torch.Tensor) else v) for k, v in obs.items()}
    batch["task"] = ["a", "b", "c"]
    batch[OBS_LANGUAGE_TOKENS] = torch.tensor([[11, 12], [13, 14]])  # two rows for three environments
    with pytest.raises(ValueError, match="one row per environment"):
        policy.select_action(batch)
    batch[OBS_LANGUAGE_TOKENS] = torch.tensor([[11, 12]]).expand(3, 2, 2)  # a stacked (B, T, L) tensor
    with pytest.raises(ValueError, match="must be \\(rows, length\\)"):
        policy.select_action(batch)


def test_config_rejects_bad_seeds_and_time_grid() -> None:
    with pytest.raises(ValueError, match="seed"):
        tiny_lerobot_config(seed=-1)
    with pytest.raises(ValueError, match="seed_stride"):
        tiny_lerobot_config(seed_stride=0)
    with pytest.raises(ValueError, match="t_shift"):
        tiny_lerobot_config(t_shift=0.0)


@pytest.mark.parametrize(
    ("backend", "capability", "block", "want"),
    [
        ("flash", (10, 3), 128, 256),  # GB300
        ("flash", (10, 0), 128, 256),  # B200
        ("flash", (10, 0), 512, 512),
        ("flash", (11, 0), 128, 256),  # Jetson Thor
        ("flash", (9, 0), 128, 128),  # H100
        ("flash", (12, 0), 128, 128),  # RTX Blackwell
        ("triton", (10, 3), 128, 128),
    ],
)
def test_flash_masks_take_256_token_blocks_on_datacenter_blackwell(
    monkeypatch: pytest.MonkeyPatch, backend: str, capability: tuple[int, int], block: int, want: int
) -> None:
    """FlashAttention-4 refuses block-sparse masks of 128-token blocks on SM100, SM103 and SM110."""
    from lerobot.policies.praxis.modeling_praxis import mask_block_size

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    assert mask_block_size(block, backend, torch.device("cuda")) == want
    assert mask_block_size(block, backend, torch.device("cpu")) == block


@pytest.mark.parametrize(
    ("capability", "flash_installed", "want"),
    [
        ((9, 0), True, "flash"),  # H100
        ((10, 0), True, "flash"),  # B200
        ((10, 3), True, "flash"),  # GB300
        ((11, 0), True, "flash"),  # Jetson Thor
        ((12, 0), True, "triton"),  # RTX PRO 6000 Blackwell
        ((12, 1), True, "triton"),  # GB10
        ((10, 0), False, "triton"),
    ],
)
def test_auto_backend_follows_the_block_masks_flash_attention_takes(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int], flash_installed: bool, want: str
) -> None:
    """FlashAttention-4 has no block-sparse forward on compute capability 12.x: `auto` takes `triton`."""
    from lerobot.policies.praxis import modeling_praxis

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(modeling_praxis, "flash_backend_available", lambda: flash_installed)
    cuda = torch.device("cuda")
    assert modeling_praxis.resolve_attention_backend("auto", cuda) == (want, True)
    assert modeling_praxis.resolve_attention_backend("triton", cuda) == ("triton", True)
    assert modeling_praxis.resolve_attention_backend("auto", torch.device("cpu")) == ("triton", False)
    if capability[0] == 12:
        with pytest.raises(ValueError, match="no block-sparse forward"):
            modeling_praxis.resolve_attention_backend("flash", cuda)
    else:
        assert modeling_praxis.resolve_attention_backend("flash", cuda) == ("flash", True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_to_another_device_selects_its_attention_kernels() -> None:
    """A policy built for the CPU (dense flex attention, SDPA expert) takes the GPU kernels of the
    checkpoint when moved to a GPU, and the CPU ones again when moved back."""
    from lerobot.policies.praxis.modeling_praxis import mask_block_size
    from lerobot.policies.praxis.praxis_core import layers

    policy = make_tiny_policy(seed=8)
    towers = (policy.model.backbone.encoder, policy.model.backbone.decoder)
    block = policy.config.model_config["backbone"]["attention"]["flex_block_size"]
    assert policy.attention_backend == "triton" and policy.model.action_expert.attention == "sdpa"
    policy.to("cuda")
    assert policy.attention_backend in ("flash", "triton") and policy.model.device.type == "cuda"
    if torch.cuda.get_device_capability()[0] == 12:
        assert policy.attention_backend == "triton"
    attention = [m for m in policy.model.modules() if isinstance(m, layers.Attention)]
    assert attention and all(m.backend == policy.attention_backend for m in attention)
    assert policy.model.action_expert.attention == policy.config.model_config["action_expert"]["attention"]
    gpu_block = mask_block_size(block, policy.attention_backend, policy.model.device)
    assert all(t._tf_masks.block_size == gpu_block for t in towers)
    policy.to("cpu")
    assert policy.attention_backend == "triton" and policy.model.action_expert.attention == "sdpa"
    assert all(m.backend == "triton" and m._kernel_options is None for m in attention)
    assert all(t._tf_masks.block_size == block for t in towers)
