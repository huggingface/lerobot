# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Wire parsing must reject aggregate allocations before decoding typed records."""

from dataclasses import replace

import pytest

pytest.importorskip("msgpack")
import msgpack

from lerobot.remote_inference import codec
from lerobot.remote_inference.protocol import Envelope, MessageType, ProtocolError


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
