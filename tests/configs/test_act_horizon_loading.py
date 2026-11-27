# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Test cases contributed by Jonathan D Faeth (@JonnyFaeth).
import json
from pathlib import Path

import pytest
from draccus.utils import ParsingError

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@pytest.mark.parametrize("field", ["chunk_size", "n_action_steps"])
@pytest.mark.parametrize("value", [0, -1])
@pytest.mark.parametrize("via_override", [False, True])
def test_loaded_act_rejects_invalid_horizon(tmp_path: Path, field: str, value: int, via_override: bool):
    ACTConfig(device="cpu", chunk_size=1, n_action_steps=1).save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    args = []
    if via_override:
        args = [f"--{field}={value}"]
    else:
        config = json.loads(config_path.read_text())
        config[field] = value
        config_path.write_text(json.dumps(config))
    with pytest.raises(ParsingError) as error:
        PreTrainedConfig.from_pretrained(tmp_path, local_files_only=True, cli_overrides=args)
    assert isinstance(error.value.__cause__, ValueError)
    assert str(error.value.__cause__) == f"`{field}` must be positive. Got {value}."


@pytest.mark.parametrize("chunk,steps", [(1, 1), (20, 1), (20, 20)])
def test_loaded_act_positive_horizons_roundtrip(tmp_path: Path, chunk: int, steps: int):
    ACTConfig(device="cpu", chunk_size=chunk, n_action_steps=steps).save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path, local_files_only=True)
    assert isinstance(loaded, ACTConfig)
    assert (loaded.chunk_size, loaded.n_action_steps) == (chunk, steps)
