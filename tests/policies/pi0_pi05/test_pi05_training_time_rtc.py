#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import (  # noqa: E402
    _build_flow_matching_inputs,
    _reduce_training_rtc_loss,
)


def test_pi05_training_rtc_uses_clean_prefix_and_per_token_time():
    actions = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    noise = torch.tensor([[[10.0], [20.0], [30.0], [40.0]]])
    time = torch.tensor([0.25])
    prefix_mask = torch.tensor([[True, True, False, False]])

    x_t, model_time = _build_flow_matching_inputs(actions, noise, time, prefix_mask)

    assert model_time.tolist() == [[0.0, 0.0, 0.25, 0.25]]
    assert torch.equal(x_t[:, :2], actions[:, :2])
    assert torch.equal(x_t[:, 2:], 0.25 * noise[:, 2:] + 0.75 * actions[:, 2:])


def test_pi05_training_rtc_loss_excludes_clean_prefix():
    losses = torch.tensor([[[100.0], [100.0], [2.0], [4.0]]])
    prefix_mask = torch.tensor([[True, True, False, False]])

    loss = _reduce_training_rtc_loss(losses, prefix_mask, reduction="mean")

    assert loss.item() == pytest.approx(3.0)


@pytest.mark.parametrize("max_delay", [-1, 5])
def test_pi05_config_rejects_invalid_training_rtc_delay(max_delay):
    with pytest.raises(ValueError, match="rtc_training_max_delay"):
        PI05Config(chunk_size=5, n_action_steps=5, rtc_training_max_delay=max_delay)
