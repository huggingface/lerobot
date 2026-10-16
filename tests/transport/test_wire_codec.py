# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("msgpack")
import msgpack

from lerobot.transport.wire import codec
from lerobot.transport.wire.codec import CodecLimits, RGBImage, decode_message, encode_message
from lerobot.transport.wire.protocol import Envelope, ErrorCode, MessageType, ProtocolError


def message(body=None):
    return Envelope(MessageType.ACTION, "instance", "session", 4, "request", body or {})


def packed_record(value):
    envelope = msgpack.unpackb(encode_message(message()), raw=False)
    envelope["body"] = {"value": value}
    return msgpack.packb(envelope, use_bin_type=True)


@pytest.mark.parametrize(
    "dtype", ["bool", "uint8", "int16", "uint64", "float16", "float32", "float64", ">f4"]
)
def test_raw_tensor_exactness_and_ownership(dtype):
    source = np.arange(12).astype(dtype).reshape(3, 4).T  # deliberately noncontiguous
    result = decode_message(encode_message(message({"action": source})))
    actual = result.body["action"]
    np.testing.assert_array_equal(actual, source)
    assert actual.dtype.name == source.dtype.name
    assert actual.flags.writeable
    actual[0, 0] = 1
    assert source[0, 0] == 0
    assert (result.instance_id, result.session_id, result.generation, result.request_id) == (
        "instance",
        "session",
        4,
        "request",
    )


@pytest.mark.parametrize("encoding", ["raw", "jpeg"])
def test_rgb_channel_order(encoding):
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255
    decoded = decode_message(encode_message(message({"camera": RGBImage(rgb, encoding)}))).body["camera"]
    assert decoded.shape == rgb.shape
    assert decoded.dtype == rgb.dtype
    if encoding == "raw":
        np.testing.assert_array_equal(decoded, rgb)
    else:
        assert np.max(abs(decoded.astype(int) - rgb.astype(int))) <= 2


@pytest.mark.parametrize(
    "changes",
    [
        {"dtype": "object"},
        {"dtype": "V10000000"},
        {"shape": [True, 3]},
        {"shape": [-1]},
        {"shape": [1] * 9},
        {"shape": [1000000]},
        {"shape": [3]},
        {"endianness": "native"},
        {"data": b""},
        {"data": np.array([np.nan], np.float32).tobytes()},
    ],
)
def test_tensor_metadata_rejected_before_allocation(changes):
    record = {
        "__lerobot_type__": "tensor",
        "dtype": "float32",
        "endianness": "little",
        "shape": [1],
        "data": np.array([1], np.float32).tobytes(),
    }
    record.update(changes)
    with pytest.raises(ProtocolError, match="dtype|dimensions|rank|size|count|finite|byte|Tensor"):
        decode_message(packed_record(record))


def test_jpeg_header_checked_before_decode(monkeypatch):
    rgb = np.zeros((32, 32, 3), np.uint8)
    wire = msgpack.unpackb(encode_message(message({"image": RGBImage(rgb, "jpeg")})), raw=False)
    wire["body"]["image"]["shape"] = [1, 1, 3]
    # Header comparison must reject the lie without entering NumPy pixel allocation.
    monkeypatch.setattr(np, "array", lambda *a, **k: pytest.fail("allocated malformed image"))
    with pytest.raises(ProtocolError, match="header"):
        decode_message(msgpack.packb(wire, use_bin_type=True))


def test_rgb_declared_limit_before_decode():
    record = {
        "__lerobot_type__": "rgb",
        "dtype": "uint8",
        "encoding": "jpeg",
        "channel_order": "RGB",
        "shape": [32, 32, 3],
        "data": b"not a jpeg",
    }
    with pytest.raises(ProtocolError, match="pixel limit"):
        decode_message(packed_record(record), replace(CodecLimits(), max_image_pixels=16))


@pytest.mark.parametrize("value", [np.array([np.nan]), np.array([np.inf]), float("inf")])
def test_non_finite_rejected(value):
    with pytest.raises(ProtocolError, match="finite"):
        encode_message(message({"action": value}))


def test_protocol_major_validated_before_tensors(monkeypatch):
    wire = msgpack.unpackb(encode_message(message({"action": np.ones(4)})), raw=False)
    wire["version"] = 9000
    monkeypatch.setattr(np, "frombuffer", lambda *a, **k: pytest.fail("allocated incompatible protocol"))
    with pytest.raises(ProtocolError) as exc:
        decode_message(msgpack.packb(wire, use_bin_type=True))
    assert exc.value.code == ErrorCode.PROTOCOL


def test_payload_string_node_and_decoded_limits():
    with pytest.raises(ProtocolError, match="byte limit"):
        decode_message(b"x" * 100, replace(CodecLimits(), max_payload_bytes=32))
    with pytest.raises(ProtocolError, match="String"):
        encode_message(message({"task": "é" * 20}), replace(CodecLimits(), max_string_bytes=32))
    with pytest.raises(ProtocolError, match="node count"):
        decode_message(
            encode_message(message({"list": list(range(20))})), replace(CodecLimits(), max_nodes=8)
        )
    with pytest.raises(ProtocolError, match="Total decoded"):
        decode_message(
            encode_message(message({"x": np.zeros(8), "y": np.zeros(8)})),
            replace(CodecLimits(), max_decoded_bytes=100),
        )


@pytest.mark.parametrize(
    "payload", [b"\x81\xa1x", b"\xc1", b"\x82\xa1x\xc0\xa1x\xc0", msgpack.packb(msgpack.ExtType(1, b"x"))]
)
def test_malformed_messagepack(payload):
    with pytest.raises(ProtocolError):
        decode_message(payload)


def test_session_message_requires_identity():
    with pytest.raises(ProtocolError, match="require instance"):
        Envelope(MessageType.OBSERVATION)


def test_peek_correlates_malformed_obsolete_body_without_allocating(monkeypatch):
    from lerobot.transport.wire.codec import peek_envelope

    invalid = {"__lerobot_type__": "tensor", "dtype": "object", "shape": [1], "data": b""}
    monkeypatch.setattr(np, "frombuffer", lambda *a, **k: pytest.fail("allocated in header peek"))
    header = peek_envelope(packed_record(invalid))
    assert header.body == {}
    assert header.request_id == "request"
    with pytest.raises(ProtocolError):
        decode_message(packed_record(invalid))


@pytest.mark.parametrize("read", [codec.decode_message, codec.peek_envelope])
@pytest.mark.parametrize("nested", [[], {}])
def test_aggregate_nodes_are_bounded_during_unpacking(read, nested, monkeypatch):
    envelope = msgpack.unpackb(codec.encode_message(Envelope(MessageType.DESCRIBE)), raw=False)
    # Every individual container is within the item bound. The aggregate is not.
    envelope["body"] = {"groups": [[nested for _ in range(8)] for _ in range(8)]}
    payload = msgpack.packb(envelope, use_bin_type=True)
    limits = replace(codec.CodecLimits(), max_container_items=16, max_nodes=24)
    completed = []
    from lerobot.transport.wire import codec as wire_codec

    sequence = wire_codec._UnpackBudget.sequence

    def count_sequence(self, items):
        completed.append(len(items))
        return sequence(self, items)

    monkeypatch.setattr(wire_codec._UnpackBudget, "sequence", count_sequence)
    monkeypatch.setattr(wire_codec, "_decode_value", lambda *args: pytest.fail("reached full body decoding"))
    with pytest.raises(ProtocolError, match="node count.*parsing"):
        read(payload, limits)
    # The outer groups list is never completed: parsing stopped within its children.
    assert len(completed) < 64


def test_wire_node_budget_accepts_small_mixed_containers():
    source = Envelope(MessageType.DESCRIBE, body={"values": [[], {}, [1], {"value": 2}]})
    encoded = codec.encode_message(source)
    decoded = codec.decode_message(encoded, replace(codec.CodecLimits(), max_nodes=32))
    assert decoded == source


def test_noncanonical_boolean_wire_bytes_are_rejected():
    source = Envelope(MessageType.DESCRIBE, body={"mask": np.array([False, True])})
    envelope = msgpack.unpackb(codec.encode_message(source), raw=False)
    envelope["body"]["mask"]["data"] = bytes([0, 2])
    with pytest.raises(ProtocolError, match="Boolean tensor bytes must be 0 or 1") as exc:
        codec.decode_message(msgpack.packb(envelope, use_bin_type=True))
    assert exc.value.code is ErrorCode.MALFORMED
