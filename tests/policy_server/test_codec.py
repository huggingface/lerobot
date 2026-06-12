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

"""Unit tests for the MessagePack wire codecs (tensors, images, messages)."""

import numpy as np
import pytest

msgpack = pytest.importorskip("msgpack")

from lerobot.policy_server.codec import (  # noqa: E402
    decode_action_chunk,
    decode_image,
    decode_observation,
    decode_raw,
    decode_reset,
    decode_reset_ack,
    decode_session_ack,
    decode_session_close,
    decode_session_open,
    decode_status,
    decode_tensor,
    encode_action_chunk,
    encode_image,
    encode_observation,
    encode_reset,
    encode_reset_ack,
    encode_session_ack,
    encode_session_close,
    encode_session_open,
    encode_status,
    encode_tensor,
)
from lerobot.policy_server.schema import (  # noqa: E402
    IMAGE_CODEC_JPEG,
    IMAGE_CODEC_RAW,
    ActionChunkMsg,
    ObservationMsg,
    ResetAckMsg,
    ResetMsg,
    SessionAckMsg,
    SessionCloseMsg,
    SessionOpenMsg,
    StatusMsg,
)

# ---------------------------------------------------------------------------
# Tensor codec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arr",
    [
        np.array([1.5, -2.25, 3.0], dtype=np.float32),
        np.arange(12, dtype=np.float64).reshape(3, 4),
        np.array([[1, -2], [3, 4]], dtype=np.int64),
        np.array([[True, False], [False, True]], dtype=np.bool_),
        np.zeros((0,), dtype=np.float32),  # empty 1-d
        np.zeros((0, 6), dtype=np.float64),  # empty 2-d
        np.arange(24, dtype=np.int64).reshape(2, 3, 4),
    ],
    ids=["f32_1d", "f64_2d", "i64_2d", "bool_2d", "f32_empty", "f64_empty_2d", "i64_3d"],
)
def test_tensor_roundtrip(arr):
    out = decode_tensor(encode_tensor(arr))
    assert out.dtype == arr.dtype
    assert out.shape == arr.shape
    np.testing.assert_array_equal(out, arr)


def test_tensor_roundtrip_0d_preserves_value():
    # KNOWN QUIRK: np.ascontiguousarray inside encode_tensor promotes
    # 0-d arrays to shape (1,), so the round-trip is value-preserving
    # but not shape-preserving for scalars.
    arr = np.array(3.5, dtype=np.float32)
    out = decode_tensor(encode_tensor(arr))
    assert out.dtype == arr.dtype
    assert out.shape in ((), (1,))
    assert float(np.squeeze(out)) == 3.5


def test_tensor_none_passthrough():
    assert encode_tensor(None) is None
    assert decode_tensor(None) is None


def test_tensor_big_endian_input_values_identical():
    be = np.array([1.0, 2.5, -3.75], dtype=">f4")
    enc = encode_tensor(be)
    assert np.dtype(enc["dtype"]).byteorder != ">"
    out = decode_tensor(enc)
    np.testing.assert_array_equal(out, be.astype("<f4"))
    np.testing.assert_array_equal(out, np.array([1.0, 2.5, -3.75], dtype=np.float32))


def test_tensor_decoded_writable_and_contiguous():
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = decode_tensor(encode_tensor(arr))
    assert out.flags.writeable
    assert out.flags.c_contiguous
    out[0, 0] = 99.0  # must not raise
    assert out[0, 0] == 99.0


def test_tensor_decode_refuses_object_dtype():
    with pytest.raises(ValueError, match="object dtype"):
        decode_tensor({"dtype": "|O", "shape": [1], "data": b"\x00" * 8})


def test_tensor_roundtrip_through_msgpack():
    arr = np.arange(10, dtype=np.float32)
    packed = msgpack.packb(encode_tensor(arr), use_bin_type=True)
    out = decode_tensor(msgpack.unpackb(packed, raw=False))
    np.testing.assert_array_equal(out, arr)


# ---------------------------------------------------------------------------
# Image codec
# ---------------------------------------------------------------------------


def _gradient_image(h: int = 32, w: int = 48) -> np.ndarray:
    """Smooth RGB gradient: JPEG-friendly, deterministic."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    img[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    img[..., 2] = 128
    return img


def test_image_raw_roundtrip_byte_exact():
    img = _gradient_image()
    enc = encode_image(img, jpeg_quality=0)
    assert enc["codec"] == IMAGE_CODEC_RAW
    out = decode_image(enc)
    assert out.dtype == np.uint8
    assert out.shape == img.shape
    np.testing.assert_array_equal(out, img)


def test_image_jpeg_roundtrip_approximately_equal():
    img = _gradient_image()
    enc = encode_image(img, jpeg_quality=95)
    assert enc["codec"] == IMAGE_CODEC_JPEG
    out = decode_image(enc)
    assert out.dtype == np.uint8
    assert out.shape == img.shape
    err = np.abs(out.astype(np.int32) - img.astype(np.int32)).mean()
    assert err < 5.0, f"JPEG round-trip too lossy: mean abs error {err}"


def test_image_jpeg_rgb_order_regression_pure_red_stays_red():
    # A silent BGR swap would poison every VLA in a fleet: pure red must
    # come back red-dominant, not blue-dominant.
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[..., 0] = 255  # RGB: red channel
    out = decode_image(encode_image(img, jpeg_quality=90))
    red_mean = out[..., 0].astype(np.float64).mean()
    blue_mean = out[..., 2].astype(np.float64).mean()
    assert red_mean > 200, f"red channel lost: mean {red_mean}"
    assert blue_mean < 50, f"blue channel gained: mean {blue_mean}"
    assert red_mean > blue_mean


def test_encode_image_rejects_float_arrays():
    with pytest.raises(ValueError, match="uint8 HWC RGB"):
        encode_image(np.zeros((8, 8, 3), dtype=np.float32))


@pytest.mark.parametrize(
    "shape", [(3, 16, 24), (16, 24), (16, 24, 4), (16, 24, 1)], ids=["chw", "hw", "hwc4", "hwc1"]
)
def test_encode_image_rejects_non_hwc(shape):
    with pytest.raises(ValueError, match="uint8 HWC RGB"):
        encode_image(np.zeros(shape, dtype=np.uint8))


def test_decode_image_rejects_unknown_codec():
    with pytest.raises(ValueError, match="Unknown image codec"):
        decode_image({"codec": "webp", "data": b""})


# ---------------------------------------------------------------------------
# Data-plane messages
# ---------------------------------------------------------------------------


def test_observation_roundtrip_full():
    rng = np.random.default_rng(0)
    state = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
    front = rng.integers(0, 255, size=(16, 24, 3), dtype=np.uint8)
    wrist = rng.integers(0, 255, size=(8, 12, 3), dtype=np.uint8)
    prefix_model = rng.standard_normal((5, 4)).astype(np.float32)
    prefix_robot = (prefix_model * 2.0).astype(np.float32)
    msg = ObservationMsg(
        state=state,
        images={"front": front, "wrist": wrist},
        task="fold the towel",
        inference_delay_steps=3,
        prefix_model=prefix_model,
        prefix_robot=prefix_robot,
        episode_start=True,
        jpeg_quality=0,  # raw: byte-exact images
    )
    out = decode_observation(encode_observation(msg))
    np.testing.assert_array_equal(out.state, state)
    assert set(out.images) == {"front", "wrist"}
    np.testing.assert_array_equal(out.images["front"], front)
    np.testing.assert_array_equal(out.images["wrist"], wrist)
    assert out.task == "fold the towel"
    assert out.inference_delay_steps == 3
    np.testing.assert_array_equal(out.prefix_model, prefix_model)
    np.testing.assert_array_equal(out.prefix_robot, prefix_robot)
    assert out.episode_start is True


def test_observation_roundtrip_minimal_defaults():
    out = decode_observation(encode_observation(ObservationMsg()))
    assert out.state is None
    assert out.images == {}
    assert out.task == ""
    assert out.inference_delay_steps == 0
    assert out.prefix_model is None
    assert out.prefix_robot is None
    assert out.episode_start is False


def test_action_chunk_roundtrip():
    chunk_model = np.arange(12, dtype=np.float32).reshape(3, 4)
    chunk_robot = chunk_model * 2.0
    msg = ActionChunkMsg(
        seq_id_echo=17,
        client_mono_ns_echo=123456789,
        episode_id_echo=2,
        chunk_model=chunk_model,
        chunk_robot=chunk_robot,
        queue_wait_ms=1.5,
        inference_ms=12.25,
        superseded_seqs=4,
        server_load=0.75,
    )
    out = decode_action_chunk(encode_action_chunk(msg))
    assert out.seq_id_echo == 17
    assert out.client_mono_ns_echo == 123456789
    assert out.episode_id_echo == 2
    np.testing.assert_array_equal(out.chunk_model, chunk_model)
    np.testing.assert_array_equal(out.chunk_robot, chunk_robot)
    assert out.queue_wait_ms == 1.5
    assert out.inference_ms == 12.25
    assert out.superseded_seqs == 4
    assert out.server_load == 0.75


# ---------------------------------------------------------------------------
# Control-plane messages
# ---------------------------------------------------------------------------


def test_session_open_roundtrip():
    msg = SessionOpenMsg(
        client_uuid="uuid-1",
        robot_type="so101_follower",
        policy_type="pi0",
        fps=30.0,
        action_names=["j0.pos", "j1.pos"],
        camera_names=["front", "wrist"],
        state_dim=6,
        rtc_enabled=True,
        task="fold",
        tags={"site": "lab-3"},
    )
    out = decode_session_open(encode_session_open(msg))
    assert out == msg


def test_session_ack_roundtrip():
    msg = SessionAckMsg(
        accepted=True,
        warnings=["fps mismatch"],
        session_id="sess-1",
        model_repo="org/model",
        model_revision="main",
        policy_type="pi0",
        action_names=["j0.pos"],
        expected_cameras=["front"],
        state_dim=6,
        chunk_size=50,
        trained_fps=30.0,
        supports_rtc=True,
        rtc_execution_horizon=25,
        serving_mode="shared",
        warmed_up=True,
        server_load=0.5,
    )
    out = decode_session_ack(encode_session_ack(msg))
    assert out == msg


def test_status_roundtrip():
    msg = StatusMsg(
        model_repo="org/model",
        model_revision="abc123",
        policy_type="act",
        action_names=["j0.pos", "j1.pos"],
        expected_cameras=["front"],
        state_dim=6,
        chunk_size=100,
        trained_fps=30.0,
        supports_rtc=False,
        rtc_execution_horizon=0,
        serving_mode="exclusive",
        warmed_up=False,
        active_sessions=2,
        max_sessions=4,
    )
    out = decode_status(encode_status(msg))
    assert out == msg


def test_reset_and_reset_ack_roundtrip():
    out = decode_reset(encode_reset(ResetMsg(client_uuid="uuid-1", episode_id=5)))
    assert out == ResetMsg(client_uuid="uuid-1", episode_id=5)
    out_ack = decode_reset_ack(encode_reset_ack(ResetAckMsg(ok=False, error="busy")))
    assert out_ack == ResetAckMsg(ok=False, error="busy")


def test_session_close_roundtrip():
    msg = SessionCloseMsg(client_uuid="uuid-1", session_id="sess-1")
    out = decode_session_close(encode_session_close(msg))
    assert out == msg


# ---------------------------------------------------------------------------
# Schema evolution (additive-only contract)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("encoded", "decoder", "expected"),
    [
        (
            encode_session_ack(SessionAckMsg(accepted=True, session_id="s")),
            decode_session_ack,
            SessionAckMsg(accepted=True, session_id="s"),
        ),
        (
            encode_reset(ResetMsg(client_uuid="u", episode_id=1)),
            decode_reset,
            ResetMsg(client_uuid="u", episode_id=1),
        ),
        (
            encode_session_open(SessionOpenMsg(client_uuid="u")),
            decode_session_open,
            SessionOpenMsg(client_uuid="u"),
        ),
    ],
    ids=["session_ack", "reset", "session_open"],
)
def test_unknown_keys_ignored(encoded, decoder, expected):
    obj = msgpack.unpackb(encoded, raw=False)
    obj["a_future_field"] = {"nested": [1, 2, 3]}
    out = decoder(msgpack.packb(obj, use_bin_type=True))
    assert out == expected


def test_missing_optional_keys_take_defaults():
    minimal = msgpack.packb({"accepted": True}, use_bin_type=True)
    out = decode_session_ack(minimal)
    assert out.accepted is True
    assert out.error == ""
    assert out.warnings == []
    assert out.chunk_size == 0
    assert out.server_load == 0.0

    out_chunk = decode_action_chunk(msgpack.packb({"seq_id_echo": 9}, use_bin_type=True))
    assert out_chunk.seq_id_echo == 9
    assert out_chunk.chunk_model is None
    assert out_chunk.chunk_robot is None
    assert out_chunk.queue_wait_ms == 0.0

    out_obs = decode_observation(msgpack.packb({"task": "t"}, use_bin_type=True))
    assert out_obs.task == "t"
    assert out_obs.state is None
    assert out_obs.images == {}
    assert out_obs.episode_start is False


# ---------------------------------------------------------------------------
# decode_raw
# ---------------------------------------------------------------------------


def test_decode_raw_returns_plain_dict_with_op():
    open_obj = decode_raw(encode_session_open(SessionOpenMsg(client_uuid="u")))
    assert isinstance(open_obj, dict)
    assert open_obj["op"] == "open"

    close_obj = decode_raw(encode_session_close(SessionCloseMsg(client_uuid="u")))
    assert isinstance(close_obj, dict)
    assert close_obj["op"] == "close"
