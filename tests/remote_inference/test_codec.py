# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import replace

import numpy as np
import pytest
import torch

pytest.importorskip("msgpack")
import msgpack

from lerobot.remote_inference import codec
from lerobot.remote_inference.codec import CodecLimits, decode_message, encode_message
from lerobot.remote_inference.protocol import (
    Envelope,
    ErrorCode,
    MessageType,
    ProtocolError,
    deployment_prefix,
)
from lerobot.transport.wire import codec as wire_codec


def message(body=None):
    return Envelope(MessageType.ACTION, "instance", "session", 4, "request", body or {})


def test_nested_torch_and_scalars():
    body = {
        "features": {"state": torch.arange(6, dtype=torch.float32)},
        "scalar": np.float32(1.5),
        "task": "move",
    }
    actual = decode_message(encode_message(message(body))).body
    np.testing.assert_array_equal(actual["features"]["state"], np.arange(6))
    assert actual["scalar"] == 1.5
    assert actual["task"] == "move"


@pytest.mark.parametrize("deployment", ["../x", "user/model", "*", "a b", "a?b", ""])
def test_invalid_routing_segment(deployment):
    with pytest.raises(ProtocolError):
        deployment_prefix(deployment)


def test_boolean_encoding_normalizes_valid_local_backing_bytes():
    storage = np.array([[0, 9, 2], [255, 9, 1]], dtype=np.uint8)
    mask = storage.view(np.bool_)[:, ::2]  # Valid bool values, noncanonical and noncontiguous storage.
    values = torch.from_numpy(mask)
    encoded = codec.encode_message(Envelope(MessageType.DESCRIBE, body={"mask": values}))
    envelope = msgpack.unpackb(encoded, raw=False)
    assert envelope["body"]["mask"]["data"] == b"\x00\x01\x01\x01"
    decoded = codec.decode_message(encoded).body["mask"]
    np.testing.assert_array_equal(decoded, [[False, True], [True, True]])
    assert decoded.dtype == np.bool_
    assert decoded.flags.writeable
    assert decoded.tobytes() == b"\x00\x01\x01\x01"
    torch.testing.assert_close(torch.from_numpy(decoded).eq(True), torch.from_numpy(decoded))
    np.testing.assert_array_equal(storage, [[0, 9, 2], [255, 9, 1]])


@pytest.mark.parametrize("dtype", ["float32", "bool", "uint8"])
def test_torch_adapter_matches_numpy_wire_bytes_and_cross_decoding(dtype):
    array = np.arange(12).astype(dtype).reshape(3, 4).T
    numpy_message = message({"values": {"state": array}, "task": "reach"})
    torch_message = message({"values": {"state": torch.from_numpy(array)}, "task": "reach"})
    numpy_bytes = wire_codec.encode_message(numpy_message)
    assert encode_message(numpy_message) == numpy_bytes
    assert encode_message(torch_message) == numpy_bytes
    for decode in (decode_message, wire_codec.decode_message):
        actual = decode(numpy_bytes).body["values"]["state"]
        np.testing.assert_array_equal(actual, array)
        assert actual.flags.writeable
        assert not np.shares_memory(actual, array)


@pytest.mark.parametrize("encoder", [encode_message, wire_codec.encode_message])
def test_both_encoders_enforce_payload_limits(encoder):
    with pytest.raises(ProtocolError) as exc:
        encoder(message({"state": np.zeros(32, np.float32)}), replace(CodecLimits(), max_payload_bytes=64))
    assert exc.value.code is ErrorCode.MALFORMED


def test_non_finite_torch_rejected():
    with pytest.raises(ProtocolError, match="finite"):
        encode_message(message({"action": torch.tensor([float("nan")])}))
