# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Wire parsing must reject aggregate allocations before decoding typed records."""

from dataclasses import replace

import numpy as np
import pytest
import torch

pytest.importorskip("msgpack")
import msgpack

from lerobot.remote_inference import codec
from lerobot.remote_inference.protocol import Envelope, ErrorCode, MessageType, ProtocolError


@pytest.mark.parametrize("read", [codec.decode_message, codec.peek_envelope])
@pytest.mark.parametrize("nested", [[], {}, [0], {"scalar": 0}])
def test_aggregate_nodes_are_bounded_during_unpacking(read, nested, monkeypatch):
    envelope = msgpack.unpackb(codec.encode_message(Envelope(MessageType.DESCRIBE)), raw=False)
    # Every individual container is within the item bound. The aggregate is not.
    envelope["body"] = {"groups": [[nested for _ in range(8)] for _ in range(8)]}
    payload = msgpack.packb(envelope, use_bin_type=True)
    limits = replace(codec.CodecLimits(), max_container_items=16, max_nodes=24)
    completed = []
    sequence = codec._UnpackBudget.sequence

    def count_sequence(self, items):
        completed.append(len(items))
        return sequence(self, items)

    monkeypatch.setattr(codec._UnpackBudget, "sequence", count_sequence)
    monkeypatch.setattr(codec, "_decode_value", lambda *args: pytest.fail("reached full body decoding"))
    with pytest.raises(ProtocolError, match="node count.*parsing"):
        read(payload, limits)
    # The outer groups list is never completed: parsing stopped within its children.
    assert len(completed) < 64


def test_wire_node_budget_accepts_small_mixed_containers():
    source = Envelope(MessageType.DESCRIBE, body={"values": [[], {}, [1], {"value": 2}]})
    encoded = codec.encode_message(source)
    decoded = codec.decode_message(encoded, replace(codec.CodecLimits(), max_nodes=32))
    assert decoded == source


@pytest.mark.parametrize("invalid_byte", [2, 127, 255])
def test_noncanonical_boolean_wire_bytes_are_rejected(invalid_byte):
    source = Envelope(MessageType.DESCRIBE, body={"mask": np.array([False, True])})
    envelope = msgpack.unpackb(codec.encode_message(source), raw=False)
    envelope["body"]["mask"]["data"] = bytes([0, invalid_byte])
    with pytest.raises(ProtocolError, match="Boolean tensor bytes must be 0 or 1") as exc:
        codec.decode_message(msgpack.packb(envelope, use_bin_type=True))
    assert exc.value.code is ErrorCode.MALFORMED


@pytest.mark.parametrize("as_tensor", [False, True])
def test_boolean_encoding_normalizes_valid_local_backing_bytes(as_tensor):
    storage = np.array([[0, 9, 2], [255, 9, 1]], dtype=np.uint8)
    mask = storage.view(np.bool_)[:, ::2]  # Valid bool values, noncanonical and noncontiguous storage.
    values = torch.from_numpy(mask) if as_tensor else mask
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
