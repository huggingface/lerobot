#!/usr/bin/env python

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

"""A fine-tuned GR00T checkpoint loads without NVIDIA's base weights, with the same result as before."""

import hashlib
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path

import pytest
import torch
from huggingface_hub import constants as hub_constants
from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled
from huggingface_hub.utils import filter_repo_objects, httpx
from safetensors import safe_open
from safetensors.torch import load_file, save_file

pytest.importorskip("transformers")
pytest.importorskip("diffusers")

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies import pretrained
from lerobot.policies.groot import groot_n1_7, modeling_groot
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.groot_n1_7 import GR00TN17, GR00TN17Config
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.utils.import_utils import _require_package_cache


def _write_tiny_base(path, tune_backbone=True):
    from transformers import Qwen3VLConfig

    backbone = path / "backbone"
    Qwen3VLConfig(
        tie_word_embeddings=True,
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "vocab_size": 64,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True},
            "tie_word_embeddings": True,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 32,
            "deepstack_visual_indexes": [0],
            "num_position_embeddings": 16,
        },
    ).save_pretrained(backbone)
    config = GR00TN17Config(
        model_name=str(backbone),
        select_layer=2,
        backbone_embedding_dim=32,
        hidden_size=32,
        input_embedding_dim=32,
        max_state_dim=7,
        max_action_dim=5,
        action_horizon=4,
        max_num_embodiments=4,
        use_alternate_vl_dit=False,
        vl_self_attention_cfg={"num_layers": 0},
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 1,
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 32,
            "interleave_self_attention": False,
        },
        tune_llm=tune_backbone,
        tune_visual=tune_backbone,
        load_bf16=True,
    )
    base = path / "base"
    # Saved in bf16 like NVIDIA's base, so its config asks for bf16 parameters.
    GR00TN17(config, load_backbone_weights=False).to(torch.bfloat16).save_pretrained(base)
    return base


def _policy_config(base, dtype=torch.float32):
    return GrootConfig(
        base_model_path=str(base),
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(5,))},
        device="cpu",
        dtype=dtype,
        chunk_size=4,
        n_action_steps=4,
        # Not the defaults, so a load that drops the model config overrides gives a different model.
        num_inference_timesteps=3,
        tune_top_llm_layers=1,
    )


def _cap_action_horizon(base, delta_indices):
    action = {"delta_indices": delta_indices}
    processor_kwargs = {"modality_configs": {"new_embodiment": {"action": action}}}
    (base / "processor_config.json").write_text(json.dumps({"processor_kwargs": processor_kwargs}))


def _match_copy(checkpoint, copy):
    """Record the SHA-256 of `copy` in the checkpoint's config.json, as a save that wrote that copy does."""
    config = json.loads((checkpoint / "config.json").read_text())
    config["base_config_sha256"] = hashlib.sha256(copy.read_bytes()).hexdigest()
    (checkpoint / "config.json").write_text(json.dumps(config))


def _save_finetune(tmp_path, dtype=torch.float32, tune_backbone=True):
    base = _write_tiny_base(tmp_path, tune_backbone)
    policy = GrootPolicy(_policy_config(base, dtype))
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in policy.parameters():
            if parameter.is_floating_point():
                noise = torch.randn(parameter.shape, generator=generator).to(parameter.dtype)
                parameter.add_(noise, alpha=1e-3)
    policy.save_pretrained(tmp_path / "finetune")
    return base, tmp_path / "finetune"


def _tensors(policy):
    return {
        name: (tensor.detach().cpu(), tensor.dtype, tensor.device, tensor.requires_grad)
        for name, tensor in [*policy.named_parameters(remove_duplicate=False), *policy.named_buffers()]
    }


def _assert_same(expected, actual):
    assert expected.keys() == actual.keys()
    for name, (tensor, dtype, device, requires_grad) in expected.items():
        other, other_dtype, other_device, other_requires_grad = actual[name]
        assert (dtype, device, requires_grad) == (other_dtype, other_device, other_requires_grad), name
        assert torch.equal(tensor, other), name


def _overwrite_weights_in_file(path):
    with open(path, "r+b") as file:
        header_size = int.from_bytes(file.read(8), "little")
        file.seek(8 + header_size)
        file.write(bytes(path.stat().st_size - 8 - header_size))


def _load_regular(finetune, monkeypatch, policy_cls=GrootPolicy):
    with monkeypatch.context() as patch:
        patch.setattr(GrootPolicy, "_build_groot_model_from_checkpoint", classmethod(lambda cls, *args: None))
        return policy_cls.from_pretrained(finetune)


def _cache_as_hub_repo(monkeypatch, tmp_path, finetune):
    """Caches the fine-tune as the Hub repo someone/groot-finetune without its base config copies, which only
    a snapshot_download that may reach the Hub gets. The other files load from the cache, offline. The Hub's
    branches are the cache's refs, so a test can move one. Returns the loading options and the files it
    fetched."""
    commit = "0" * 40
    repo = tmp_path / "hub" / "models--someone--groot-finetune"
    shutil.copytree(
        finetune, repo / "snapshots" / commit, ignore=shutil.ignore_patterns(modeling_groot._BASE_CONFIG_DIR)
    )
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(commit)
    (repo / "blobs").mkdir()
    files = [path.relative_to(finetune).as_posix() for path in finetune.rglob("*") if path.is_file()]
    fetched = []

    def snapshot_download(
        repo_id, allow_patterns=None, cache_dir=None, revision=None, local_files_only=False, **kwargs
    ):
        # Like the real one, it reads only the cache it is given, else the default one.
        if Path(cache_dir or hub_constants.HF_HUB_CACHE) != tmp_path / "hub":
            raise LocalEntryNotFoundError("Not in this cache.")
        if not re.fullmatch("[0-9a-f]{40}", revision or ""):
            revision = (repo / "refs" / (revision or "main")).read_text()
        snapshot = repo / "snapshots" / revision
        for name in [] if local_files_only else filter_repo_objects(files, allow_patterns=allow_patterns):
            fetched.append(name)
            # Like the Hugging Face cache, the snapshot links to a blob of the repo.
            blob = repo / "blobs" / name.replace("/", "--")
            shutil.copyfile(finetune / name, blob)
            (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
            (snapshot / name).unlink(missing_ok=True)
            (snapshot / name).symlink_to(os.path.relpath(blob, (snapshot / name).parent))
        return str(snapshot)

    monkeypatch.setattr(modeling_groot, "snapshot_download", snapshot_download)
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)
    return {"cache_dir": tmp_path / "hub"}, fetched


def _record_builds(monkeypatch):
    built = []
    real_from_pretrained = GR00TN17.from_pretrained

    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        built.append(pretrained_model_name_or_path)
        return real_from_pretrained(pretrained_model_name_or_path, **kwargs)

    monkeypatch.setattr(GR00TN17, "from_pretrained", classmethod(from_pretrained))
    return built


def _name_base(finetune, base_model_path):
    """Make the fine-tune name another base model, without a copy of its config."""
    shutil.rmtree(finetune / modeling_groot._BASE_CONFIG_DIR)
    config = json.loads((finetune / "config.json").read_text())
    (finetune / "config.json").write_text(json.dumps({**config, "base_model_path": base_model_path}))


def _hub_base(monkeypatch, tmp_path, files, on_download=False):
    """Serves `files` (name to source file) as the Hub repo someone/groot-base, read offline from a cache.
    With on_download, a file enters the cache only when a download asks for it, as on a first run. Returns
    the repo id."""
    commit = "1" * 40
    cache = tmp_path / "hub_base"
    repo = cache / "models--someone--groot-base"

    def download(names):
        snapshot = repo / "snapshots" / commit
        snapshot.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copyfile(files[name], snapshot / name)
        (repo / "refs").mkdir(exist_ok=True)
        (repo / "refs" / "main").write_text(commit)
        return snapshot

    def snapshot_download(repo_id, allow_patterns=None, **kwargs):
        return str(download(filter_repo_objects(files, allow_patterns=allow_patterns)))

    cache.mkdir()
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)
    if on_download:
        monkeypatch.setattr(groot_n1_7, "snapshot_download", snapshot_download)
        monkeypatch.setattr(modeling_groot, "snapshot_download", snapshot_download)
    else:
        download(files)
    return "someone/groot-base"


def test_finetune_matches_regular_load_without_base_weights(tmp_path, monkeypatch):
    base, finetune = _save_finetune(tmp_path, dtype=torch.bfloat16)
    regular = _load_regular(finetune, monkeypatch)
    assert regular._groot_model.action_head.num_inference_timesteps == 3
    expected = _tensors(regular)

    (base / "model.safetensors").unlink()
    policy = GrootPolicy.from_pretrained(finetune)
    # The weights must be copies, not views of the checkpoint file.
    _overwrite_weights_in_file(finetune / "model.safetensors")

    _assert_same(expected, _tensors(policy))
    assert not policy.training
    assert policy._groot_model.action_head.num_inference_timesteps == 3
    lm_head, embedding = modeling_groot._TIED_WEIGHT_NAMES
    assert policy._groot_model.get_parameter(lm_head) is policy._groot_model.get_parameter(embedding)
    # Both loads share the code that applies tune_top_llm_layers, so check its effect directly.
    first, top = policy._groot_model.backbone.language_model.layers
    assert all(parameter.requires_grad for parameter in top.parameters())
    assert not any(parameter.requires_grad for parameter in first.parameters())


def test_loading_leaves_the_global_random_state_as_it_was(tmp_path, monkeypatch):
    _, finetune = _save_finetune(tmp_path)
    state = torch.get_rng_state()

    # The regular load draws random numbers to initialize the weights that the checkpoint then replaces.
    _load_regular(finetune, monkeypatch)

    assert torch.equal(torch.get_rng_state(), state)


def test_finetune_loads_by_hub_repo_id(tmp_path, monkeypatch):
    base, finetune = _save_finetune(tmp_path)
    expected = _tensors(_load_regular(finetune, monkeypatch))
    hub, fetched = _cache_as_hub_repo(monkeypatch, tmp_path, finetune)

    shutil.rmtree(base)
    policy = GrootPolicy.from_pretrained("someone/groot-finetune", **hub)

    _assert_same(expected, _tensors(policy))
    assert fetched == [f"{modeling_groot._BASE_CONFIG_DIR}/config.json"]


def test_load_by_hub_repo_id_reads_the_saved_copies(tmp_path, monkeypatch):
    base = _write_tiny_base(tmp_path)
    _cap_action_horizon(base, [0, 1])
    GrootPolicy(_policy_config(base)).save_pretrained(tmp_path / "finetune")
    hub, _ = _cache_as_hub_repo(monkeypatch, tmp_path, tmp_path / "finetune")
    # The base changes after the save, so a load that reads it instead of the copies builds another model.
    model_config = json.loads((base / "config.json").read_text())
    (base / "config.json").write_text(json.dumps({**model_config, "action_horizon": 2}))
    _cap_action_horizon(base, [0])

    class Subclass(GrootPolicy):
        # Takes only the config, so the folder of the copies cannot reach it as an argument.
        def __init__(self, config):
            super().__init__(config)

    policy = Subclass.from_pretrained("someone/groot-finetune", **hub)
    policy.save_pretrained(tmp_path / "resaved")
    shutil.rmtree(base)

    assert policy._groot_model.config.action_horizon == 4
    assert policy._action_queue_steps == 2
    # The config holds the horizon the policy was built with, so it reads neither the base nor the copies.
    assert policy.config.action_delta_indices == [0, 1]
    # A save after the load keeps the copies, not the changed base.
    resaved = GrootPolicy.from_pretrained(tmp_path / "resaved")
    assert resaved._groot_model.config.action_horizon == 4
    assert resaved._action_queue_steps == 2


def test_load_reads_the_copies_of_the_checkpoint_it_loads(tmp_path, monkeypatch):
    base, finetune = _save_finetune(tmp_path)
    # The base changes after the save, so a load that reads it instead of the copies builds another model.
    model_config = json.loads((base / "config.json").read_text())
    (base / "config.json").write_text(json.dumps({**model_config, "action_horizon": 2}))
    config = GrootConfig.from_pretrained(finetune)
    # As make_policy passes it to load a PEFT adapter's base policy.
    config.pretrained_path = tmp_path / "adapter"

    class Subclass(GrootPolicy):
        pass

    policy = Subclass.from_pretrained(finetune, config=config)

    assert policy._groot_model.config.action_horizon == 4
    assert config.pretrained_path == tmp_path / "adapter"


@pytest.mark.parametrize(
    ("failure", "cached"),
    [
        # Offline, huggingface_hub asks the Hub for a file list it has not cached, even when the files are.
        pytest.param(OfflineModeIsEnabled("Outgoing traffic has been disabled."), True, id="offline_cached"),
        pytest.param(httpx.ConnectError("The Hub cannot be reached."), False, id="unreachable"),
        # Offline, huggingface_hub raises a subclass of this when its cached file list names files it lacks.
        pytest.param(LocalEntryNotFoundError("Some files are not cached."), False, id="incomplete_cache"),
        pytest.param(None, False, id="local_files_only"),
    ],
)
def test_hub_checkpoint_reads_the_copies_it_has(tmp_path, monkeypatch, caplog, failure, cached):
    base, finetune = _save_finetune(tmp_path)
    expected = _tensors(_load_regular(finetune, monkeypatch))
    hub, _ = _cache_as_hub_repo(monkeypatch, tmp_path, finetune)
    if cached:
        # An earlier load downloaded them.
        GrootPolicy.from_pretrained("someone/groot-finetune", **hub)
    if failure is None:
        hub["local_files_only"] = True
    else:

        def snapshot_download(repo_id, **kwargs):
            raise failure

        monkeypatch.setattr(modeling_groot, "snapshot_download", snapshot_download)
    (base / "model.safetensors").unlink()
    with caplog.at_level(logging.WARNING):
        policy = GrootPolicy.from_pretrained("someone/groot-finetune", **hub)

    _assert_same(expected, _tensors(policy))
    assert ("No base config copies" in caplog.text) is not cached


@pytest.mark.parametrize("subclass", [False, True])
def test_copies_come_from_the_commit_of_the_weights(tmp_path, monkeypatch, subclass):
    base, finetune = _save_finetune(tmp_path)
    hub, _ = _cache_as_hub_repo(monkeypatch, tmp_path, finetune)
    config = PreTrainedConfig.from_pretrained("someone/groot-finetune", **hub)
    # The base changes after the save, so a load that reads it instead of the copies builds another model.
    model_config = json.loads((base / "config.json").read_text())
    (base / "config.json").write_text(json.dumps({**model_config, "action_horizon": 2}))
    download = modeling_groot.hf_hub_download

    def download_while_main_moves(*args, **kwargs):
        model_file = download(*args, **kwargs)
        # A push while the weights download, so main names another commit by the next download.
        (tmp_path / "hub" / "models--someone--groot-finetune" / "refs" / "main").write_text("1" * 40)
        return model_file

    class Subclass(GrootPolicy):
        pass

    monkeypatch.setattr(modeling_groot, "hf_hub_download", download_while_main_moves)
    policy = (Subclass if subclass else GrootPolicy).from_pretrained(
        "someone/groot-finetune", config=config, **hub
    )

    assert policy._groot_model.config.action_horizon == 4


def test_config_comes_from_the_commit_of_the_weights(tmp_path, monkeypatch):
    base, finetune = _save_finetune(tmp_path)
    hub, _ = _cache_as_hub_repo(monkeypatch, tmp_path, finetune)
    repo = tmp_path / "hub" / "models--someone--groot-finetune"
    # A later push of a fine-tune from another base model, so its config.json records another base config.
    later = repo / "snapshots" / ("1" * 40)
    later.mkdir()
    saved = json.loads((finetune / "config.json").read_text())
    (later / "config.json").write_text(json.dumps({**saved, "base_config_sha256": "0" * 64}))
    # The base changes after the save, so a load that reads it instead of the copies builds another model.
    model_config = json.loads((base / "config.json").read_text())
    (base / "config.json").write_text(json.dumps({**model_config, "action_horizon": 2}))
    download = modeling_groot.hf_hub_download

    def download_while_main_moves(*args, **kwargs):
        model_file = download(*args, **kwargs)
        # The push lands while the weights download, so main names it when config.json is read.
        (repo / "refs" / "main").write_text("1" * 40)
        return model_file

    monkeypatch.setattr(modeling_groot, "hf_hub_download", download_while_main_moves)
    policy = GrootPolicy.from_pretrained("someone/groot-finetune", **hub)

    assert policy._groot_model.config.action_horizon == 4


def test_finetune_loads_without_the_base_model(tmp_path, monkeypatch):
    base = _write_tiny_base(tmp_path)
    # Caps the action horizon at 2 steps, so a load that loses this file gives 4.
    processor_kwargs = {"modality_configs": {"new_embodiment": {"action": {"delta_indices": [0, 1]}}}}
    (base / "processor_config.json").write_text(json.dumps({"processor_kwargs": processor_kwargs}))
    GrootPolicy(_policy_config(base)).save_pretrained(tmp_path / "finetune")
    finetune = tmp_path / "finetune"
    regular = _load_regular(finetune, monkeypatch)
    assert regular._action_queue_steps == 2
    expected = _tensors(regular)

    shutil.rmtree(base)
    policy = GrootPolicy.from_pretrained(finetune)

    _assert_same(expected, _tensors(policy))
    assert policy._action_queue_steps == 2
    assert policy.chunk_inference_spec().prediction_steps == 2
    assert policy.config.action_delta_indices == [0, 1]
    assert policy.config.drop_n_last_frames == 1
    # Saving again keeps the fine-tune self-contained, also over the folder it was loaded from or after
    # that folder is gone.
    policy.save_pretrained(finetune)
    shutil.rmtree(finetune)
    policy.save_pretrained(tmp_path / "resaved")
    assert GrootPolicy.from_pretrained(tmp_path / "resaved")._action_queue_steps == 2


def test_checkpoint_saved_before_the_copies_loads_from_the_base_model(tmp_path, caplog):
    base = _write_tiny_base(tmp_path)
    _cap_action_horizon(base, [0, 1])
    GrootPolicy(_policy_config(base)).save_pretrained(tmp_path / "finetune")
    finetune = tmp_path / "finetune"
    expected = _tensors(GrootPolicy.from_pretrained(finetune))
    # As a save before this change wrote it: no copy, and neither key.
    shutil.rmtree(finetune / modeling_groot._BASE_CONFIG_DIR)
    config = json.loads((finetune / "config.json").read_text())
    del config["embodiment_action_horizons"], config["base_config_sha256"]
    (finetune / "config.json").write_text(json.dumps(config))

    with caplog.at_level(logging.WARNING):
        policy = GrootPolicy.from_pretrained(finetune)

    _assert_same(expected, _tensors(policy))
    assert policy._action_queue_steps == 2
    assert policy.config.action_delta_indices == [0, 1]
    # It records no base config, so it is not warned about one.
    assert "may differ from the one trained" not in caplog.text


def test_hub_checkpoint_saved_before_the_copies_downloads_no_copy(tmp_path, monkeypatch):
    _, finetune = _save_finetune(tmp_path)
    # As a save before this change wrote it: no copy, and neither key.
    shutil.rmtree(finetune / modeling_groot._BASE_CONFIG_DIR)
    config = json.loads((finetune / "config.json").read_text())
    del config["embodiment_action_horizons"], config["base_config_sha256"]
    (finetune / "config.json").write_text(json.dumps(config))
    hub, _ = _cache_as_hub_repo(monkeypatch, tmp_path, finetune)
    download = modeling_groot.snapshot_download
    requested = []

    def recording_download(repo_id, **kwargs):
        requested.append(kwargs.get("allow_patterns"))
        return download(repo_id, **kwargs)

    monkeypatch.setattr(modeling_groot, "snapshot_download", recording_download)
    GrootPolicy.from_pretrained("someone/groot-finetune", **hub)

    # Without a recorded SHA-256 no copy would be used, so none is asked of the Hub.
    assert requested == []


def test_base_config_that_changed_since_training_is_warned_about(tmp_path, caplog):
    base, finetune = _save_finetune(tmp_path)
    # config.json records the SHA-256 of the base config, but the checkpoint holds no copy, like a DCP one.
    shutil.rmtree(finetune / modeling_groot._BASE_CONFIG_DIR)
    model_config = json.loads((base / "config.json").read_text())
    (base / "config.json").write_text(json.dumps({**model_config, "action_horizon": 2}))

    with caplog.at_level(logging.WARNING):
        policy = GrootPolicy.from_pretrained(finetune)

    assert policy._groot_model.config.action_horizon == 2
    assert f"base_model_path {base} gives another one or none, so the model may differ" in caplog.text
    # A new run with that config builds from the base model, not from the checkpoint, so it is not warned.
    caplog.clear()
    config = GrootConfig.from_pretrained(finetune)
    config.pretrained_path = None
    with caplog.at_level(logging.WARNING):
        GrootPolicy(config)
    assert "may differ from the one trained" not in caplog.text


def test_saved_horizons_follow_the_embodiment_tag(tmp_path):
    base = _write_tiny_base(tmp_path)
    steps = {"new_embodiment": 2, "other": 1}
    modality_configs = {tag: {"action": {"delta_indices": list(range(n))}} for tag, n in steps.items()}
    processor_config = {"processor_kwargs": {"modality_configs": modality_configs}}
    (base / "processor_config.json").write_text(json.dumps(processor_config))
    GrootPolicy(_policy_config(base)).save_pretrained(tmp_path / "finetune")
    shutil.rmtree(base)
    # As lerobot-train reads --policy.path=<checkpoint> --policy.embodiment_tag=other.
    config = PreTrainedConfig.from_pretrained(tmp_path / "finetune", cli_overrides=["--embodiment_tag=other"])

    assert config.action_delta_indices == [0]
    policy = GrootPolicy.from_pretrained(tmp_path / "finetune", config=config)
    assert policy._action_queue_steps == 1
    policy.save_pretrained(tmp_path / "resaved")
    assert GrootConfig.from_pretrained(tmp_path / "resaved").action_delta_indices == [0]


def test_loading_a_base_model_reads_its_horizons(tmp_path):
    base = _write_tiny_base(tmp_path)
    _cap_action_horizon(base, [0, 1])
    GrootPolicy(_policy_config(base)).save_pretrained(tmp_path / "finetune")
    # Holds the horizon of the checkpoint it names, whose copies give it too.
    config = GrootConfig.from_pretrained(tmp_path / "finetune")
    # Another base model, in shards like NVIDIA's, so loading it by path takes the base model branch.
    other_base = shutil.copytree(base, tmp_path / "other_base")
    _cap_action_horizon(other_base, [0])
    shard = (other_base / "model.safetensors").rename(other_base / "model-00001-of-00001.safetensors")

    with safe_open(shard, framework="pt") as weights:
        index = {"metadata": {}, "weight_map": dict.fromkeys(weights.keys(), shard.name)}
    (other_base / "model.safetensors.index.json").write_text(json.dumps(index))

    policy = GrootPolicy.from_pretrained(other_base, config=config)

    assert policy._action_queue_steps == 1
    assert policy.config.action_delta_indices == [0]


def test_horizons_saved_for_one_base_model_are_read_again_for_another(tmp_path):
    base = _write_tiny_base(tmp_path)
    _cap_action_horizon(base, [0, 1])
    GrootPolicy(_policy_config(base)).save_pretrained(tmp_path / "finetune")
    other_base = shutil.copytree(base, tmp_path / "other_base")
    _cap_action_horizon(other_base, [0])
    # The saved config, reused for a run from another base model without the checkpoint.
    config = GrootConfig.from_pretrained(tmp_path / "finetune")
    config.pretrained_path = None
    config.base_model_path = str(other_base)

    assert config.action_delta_indices == [0]
    assert GrootPolicy(config)._action_queue_steps == 1


def test_dataset_without_the_checkpoint_fails_when_the_base_model_is_gone(tmp_path):
    base = _write_tiny_base(tmp_path)
    _cap_action_horizon(base, [0, 1])
    # A training run from this base model, whose train config keeps the horizons but names no checkpoint.
    config = GrootPolicy(_policy_config(base)).config
    shutil.rmtree(base)

    # As that run resumes: lerobot-train builds the dataset before the policy that loads the checkpoint.
    with pytest.raises(ValueError, match="gives no action horizons"):
        _ = config.action_delta_indices


def test_horizons_must_be_positive():
    with pytest.raises(ValueError, match="must be positive"):
        GrootConfig(embodiment_action_horizons={"new_embodiment": 0})


def test_hub_base_model_saves_its_config(tmp_path, monkeypatch):
    base = _write_tiny_base(tmp_path)
    files = {name: base / name for name in ("config.json", "model.safetensors")}
    policy = GrootPolicy(_policy_config(_hub_base(monkeypatch, tmp_path, files, on_download=True)))
    policy.save_pretrained(tmp_path / "finetune")
    shutil.rmtree(tmp_path / "hub_base")
    shutil.rmtree(base)

    _assert_same(_tensors(policy), _tensors(GrootPolicy.from_pretrained(tmp_path / "finetune")))


@pytest.mark.parametrize("regular_load", [False, True])
def test_base_weights_come_from_the_commit_of_the_base_config(tmp_path, monkeypatch, regular_load):
    base, finetune = _save_finetune(tmp_path)
    repo = tmp_path / "hub_base" / "models--someone--groot-base"
    # Three commits of the base, each with other weights and another action horizon. The cache is behind the
    # Hub: its main names the first, the Hub's main names the second when config.json downloads, and a push
    # makes it name the third right after.
    commits = [str(index) * 40 for index in range(3)]
    for index, commit in enumerate(commits):
        snapshot = Path(shutil.copytree(base, repo / "snapshots" / commit))
        model_config = json.loads((snapshot / "config.json").read_text())
        (snapshot / "config.json").write_text(json.dumps({**model_config, "action_horizon": 4 - index}))
        weights = load_file(snapshot / "model.safetensors")
        save_file({key: tensor + index for key, tensor in weights.items()}, snapshot / "model.safetensors")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(commits[0])

    def download_config(repo_id, **kwargs):
        (repo / "refs" / "main").write_text(commits[1])
        return str(repo / "snapshots" / commits[1])

    def snapshot_download(repo_id, revision=None, **kwargs):
        return str(repo / "snapshots" / (revision or commits[2]))

    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path / "hub_base"))
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(modeling_groot, "snapshot_download", download_config)
    monkeypatch.setattr(groot_n1_7, "snapshot_download", snapshot_download)

    if regular_load:
        # Without strict, a missing weight makes the load fall back to the regular one, which fills it from
        # the base weights.
        _name_base(finetune, "someone/groot-base")
        weights = load_file(finetune / "model.safetensors")
        name = next(key for key in weights if "action_head" in key)
        weights.pop(name)
        save_file(weights, finetune / "model.safetensors")
        model = GrootPolicy.from_pretrained(finetune, strict=False)._groot_model
        name = name.removeprefix("_groot_model.")
    else:
        model = GrootPolicy(_policy_config("someone/groot-base"))._groot_model
        name = next(name for name, _ in model.named_parameters() if name.startswith("action_head."))

    assert model.config.action_horizon == 3
    weights = load_file(repo / "snapshots" / commits[1] / "model.safetensors")
    assert torch.equal(model.get_parameter(name), weights[name].float())


@pytest.mark.parametrize(
    ("failure", "cached"),
    [(httpx.ConnectError, True), (httpx.ConnectTimeout, True), (503, True), (httpx.ConnectError, False)],
)
def test_base_model_is_read_from_the_cache_when_the_hub_fails(tmp_path, monkeypatch, failure, cached):
    base = _write_tiny_base(tmp_path)
    files = {name: base / name for name in ("config.json", "model.safetensors")}
    # Cached without the file list that snapshot_download keeps, as when another tool downloaded it.
    repo_id = _hub_base(monkeypatch, tmp_path, files)
    if not cached:
        monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path / "empty_cache"))

    def hub(transport, request):
        # A status is the Hub's answer when it is down; an error fails the request as without a network.
        if isinstance(failure, int):
            return httpx.Response(failure, request=request)
        raise failure("No network.", request=request)

    # HF_HUB_OFFLINE unset, so huggingface_hub tries the Hub first.
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", hub)

    if cached:
        model = GrootPolicy(_policy_config(repo_id))._groot_model
        name = next(name for name, _ in model.named_parameters() if name.startswith("action_head."))
        assert torch.equal(model.get_parameter(name), load_file(files["model.safetensors"])[name].float())
    else:
        waits = []
        monkeypatch.setattr(time, "sleep", waits.append)
        with pytest.raises(LocalEntryNotFoundError, match="check your internet connection"):
            GrootPolicy(_policy_config(repo_id))
        # At once, as main fails: nothing is retried.
        assert waits == []


@pytest.mark.parametrize("subclass", [False, True])
def test_force_download_downloads_each_file_once(tmp_path, monkeypatch, subclass):
    base, finetune = _save_finetune(tmp_path)
    expected = _tensors(_load_regular(finetune, monkeypatch))
    hub, _ = _cache_as_hub_repo(monkeypatch, tmp_path, finetune)
    config = PreTrainedConfig.from_pretrained("someone/groot-finetune", **hub)
    forced = []
    serve_copies = modeling_groot.snapshot_download
    serve_file = modeling_groot.hf_hub_download

    def snapshot_download(repo_id, force_download=False, **kwargs):
        forced.append(force_download)
        return serve_copies(repo_id, **kwargs)

    def download(repo_id, filename, force_download=False, **kwargs):
        forced.append(force_download)
        return serve_file(repo_id, filename, **kwargs)

    class Subclass(GrootPolicy):
        pass

    monkeypatch.setattr(modeling_groot, "snapshot_download", snapshot_download)
    monkeypatch.setattr(modeling_groot, "hf_hub_download", download)
    monkeypatch.setattr(pretrained, "hf_hub_download", download)
    built = _record_builds(monkeypatch)
    if not subclass:
        (base / "model.safetensors").unlink()

    policy = (Subclass if subclass else GrootPolicy).from_pretrained(
        "someone/groot-finetune", config=config, force_download=True, **hub
    )

    # The check for the weights file, the weights file, the copies, and for the regular load its second look
    # for the weights file.
    assert forced == ([False, True, True, False] if subclass else [False, True, True])
    assert built == [str(base) if subclass else None]
    _assert_same(expected, _tensors(policy))


def test_failed_forced_download_reaches_the_caller(tmp_path, monkeypatch):
    _, finetune = _save_finetune(tmp_path)

    def download(repo_id, filename, force_download=False, **kwargs):
        if force_download:
            raise OSError("No space left on device")
        return str(finetune / filename)

    monkeypatch.setattr(modeling_groot, "hf_hub_download", download)

    # As on main, rather than taking the fine-tune for a base model.
    with pytest.raises(OSError, match="No space left on device"):
        GrootPolicy.from_pretrained("someone/groot-finetune", force_download=True)


def test_checkpoint_with_only_the_embedding_name_needs_no_base_weights(tmp_path, monkeypatch):
    base, finetune = _save_finetune(tmp_path)

    weights = load_file(finetune / "model.safetensors")
    lm_head, embedding = (f"_groot_model.{name}" for name in modeling_groot._TIED_WEIGHT_NAMES)
    weights[embedding] = weights.pop(lm_head)
    save_file(weights, finetune / "model.safetensors")
    expected = _tensors(_load_regular(finetune, monkeypatch))

    (base / "model.safetensors").unlink()
    policy = GrootPolicy.from_pretrained(finetune)

    _assert_same(expected, _tensors(policy))


def test_invalid_features_are_reported_before_building(tmp_path, monkeypatch):
    _, finetune = _save_finetune(tmp_path)
    config = GrootConfig.from_pretrained(finetune)
    config.input_features["observation.state"] = PolicyFeature(
        type=FeatureType.STATE, shape=(config.max_state_dim + 1,)
    )
    built = _record_builds(monkeypatch)

    with pytest.raises(ValueError, match="exceeds max_state_dim"):
        GrootPolicy.from_pretrained(finetune, config=config)
    assert built == []


def test_checkpoint_with_weights_outside_groot_is_rejected_without_the_base_weights(tmp_path, monkeypatch):
    _, finetune = _save_finetune(tmp_path)

    weights = load_file(finetune / "model.safetensors")
    lm_head, embedding = modeling_groot._TIED_WEIGHT_NAMES
    # Outside GR00T, under the name that a GR00T weight has without its prefix.
    weights[lm_head] = weights[f"_groot_model.{lm_head}"] + 1
    # And both names of the tied pair, so the error lists two weights.
    weights[f"_groot_model.{embedding}"] = weights[f"_groot_model.{lm_head}"].clone()
    save_file(weights, finetune / "model.safetensors")
    built = _record_builds(monkeypatch)

    with pytest.raises(RuntimeError) as error:
        GrootPolicy.from_pretrained(finetune)
    assert built == [None]
    # The regular load raises the same error, after it reads the base weights.
    with pytest.raises(RuntimeError, match=f"^{re.escape(str(error.value))}$"):
        _load_regular(finetune, monkeypatch)


def test_wrong_sized_checkpoint_is_rejected_without_the_base_weights(tmp_path, monkeypatch):
    _, finetune = _save_finetune(tmp_path)

    weights = load_file(finetune / "model.safetensors")
    key = next(k for k in weights if "action_head" in k and weights[k].dim() == 2)
    weights[key] = torch.cat([weights[key], weights[key][:1]])
    save_file(weights, finetune / "model.safetensors")
    built = _record_builds(monkeypatch)

    with pytest.raises(RuntimeError, match="size mismatch") as error:
        GrootPolicy.from_pretrained(finetune, strict=False)
    assert built == [None]
    # The regular load raises the same error, after it reads the base weights.
    with pytest.raises(RuntimeError, match=f"^{re.escape(str(error.value))}$"):
        _load_regular(finetune, monkeypatch)


@pytest.mark.parametrize("strict", [True, False])
def test_checkpoint_missing_a_weight_loads_the_regular_way_only_when_not_strict(
    tmp_path, monkeypatch, caplog, strict
):
    base, finetune = _save_finetune(tmp_path)

    weights = load_file(finetune / "model.safetensors")
    weights.pop(next(k for k in weights if "action_head" in k))
    save_file(weights, finetune / "model.safetensors")
    built = _record_builds(monkeypatch)

    if strict:
        with pytest.raises(RuntimeError) as error:
            GrootPolicy.from_pretrained(finetune)
        assert built == [None]
        # The regular load raises the same error, after it reads the base weights.
        with pytest.raises(RuntimeError, match=f"^{re.escape(str(error.value))}$"):
            _load_regular(finetune, monkeypatch)
    else:
        # The base weights fill the missing one.
        with caplog.at_level(logging.WARNING):
            GrootPolicy.from_pretrained(finetune, strict=False)
        assert built == [None, str(base)]
        assert "keys differ" in caplog.text


def test_checkpoint_without_the_tied_weight_is_rejected_in_the_same_words(tmp_path, monkeypatch):
    _, finetune = _save_finetune(tmp_path)

    weights = load_file(finetune / "model.safetensors")
    # A save keeps one name of the tied LM head and embedding, so this drops both.
    weights.pop(f"_groot_model.{modeling_groot._TIED_WEIGHT_NAMES[0]}")
    save_file(weights, finetune / "model.safetensors")

    with pytest.raises(RuntimeError) as error:
        GrootPolicy.from_pretrained(finetune)
    # The regular load raises the same error, which names the tied weight once.
    with pytest.raises(RuntimeError, match=f"^{re.escape(str(error.value))}$"):
        _load_regular(finetune, monkeypatch)


@pytest.mark.parametrize(
    "faults",
    [
        "extra_and_missing",
        "two_wrong_sizes",
        "wrong_sized_lm_head",
        "both_tied_names_wrong_sized_embedding",
    ],
)
def test_faults_are_reported_in_the_order_and_names_of_the_regular_load(tmp_path, monkeypatch, faults):
    _, finetune = _save_finetune(tmp_path)

    weights = load_file(finetune / "model.safetensors")
    lm_head, embedding = (f"_groot_model.{name}" for name in modeling_groot._TIED_WEIGHT_NAMES)
    head = next(k for k in weights if "action_head" in k and weights[k].dim() == 2)
    backbone = next(k for k in weights if "backbone" in k and weights[k].dim() == 2 and k != lm_head)

    def grow(key, source=None):
        tensor = weights[source or key]
        weights[key] = torch.cat([tensor, tensor[:1]])

    if faults == "extra_and_missing":
        weights["extra.weight"] = torch.zeros(3)
        weights.pop(head)
    elif faults == "two_wrong_sizes":
        grow(head)
        grow(backbone)
    elif faults == "wrong_sized_lm_head":
        grow(lm_head)
    else:
        grow(embedding, source=lm_head)
    save_file(weights, finetune / "model.safetensors")

    with pytest.raises(RuntimeError) as error:
        GrootPolicy.from_pretrained(finetune)
    # The regular load raises the same error: torch reports weights of another shape first, in the model's
    # order and under the names the checkpoint holds them, and safetensors lists missing and extra weights
    # together.
    with pytest.raises(RuntimeError, match=f"^{re.escape(str(error.value))}$"):
        _load_regular(finetune, monkeypatch)


def test_checkpoint_with_both_tied_names_loads_the_regular_way(tmp_path, monkeypatch, caplog):
    base, finetune = _save_finetune(tmp_path)

    weights = load_file(finetune / "model.safetensors")
    lm_head = weights["_groot_model.backbone.model.lm_head.weight"].clone()
    weights["_groot_model.backbone.model.model.language_model.embed_tokens.weight"] = lm_head + 1
    save_file(weights, finetune / "model.safetensors")
    built = _record_builds(monkeypatch)

    with caplog.at_level(logging.WARNING):
        policy = GrootPolicy.from_pretrained(finetune, strict=False)
        assert torch.equal(policy.get_parameter("_groot_model.backbone.model.lm_head.weight"), lm_head)
    assert built == [str(base)]
    assert "weights that GR00T lacks" in caplog.text


def test_bfloat16_parts_of_the_base_keep_an_fp32_checkpoint_exact(tmp_path, monkeypatch, caplog):
    base, finetune = _save_finetune(tmp_path, tune_backbone=False)
    expected = _tensors(_load_regular(finetune, monkeypatch))
    built = _record_builds(monkeypatch)

    with caplog.at_level(logging.WARNING):
        policy = GrootPolicy.from_pretrained(finetune)

    _assert_same(expected, _tensors(policy))
    assert built == [None, str(base)]
    assert "builds some weights in bfloat16" in caplog.text


def test_base_config_without_dtype_loads_bfloat16_the_regular_way(tmp_path, monkeypatch, caplog):
    base, finetune = _save_finetune(tmp_path)
    for path in (base / "config.json", finetune / modeling_groot._BASE_CONFIG_DIR / "config.json"):
        path.write_text(json.dumps({**json.loads(path.read_text()), "dtype": None}))
    config = GrootConfig.from_pretrained(finetune)
    config.dtype = torch.bfloat16
    with monkeypatch.context() as patch:
        patch.setattr(GrootPolicy, "_build_groot_model_from_checkpoint", classmethod(lambda cls, *args: None))
        expected = _tensors(GrootPolicy.from_pretrained(finetune, config=config))
    built = _record_builds(monkeypatch)

    with caplog.at_level(logging.WARNING):
        policy = GrootPolicy.from_pretrained(finetune, config=config)

    _assert_same(expected, _tensors(policy))
    assert built == [str(base)]
    assert "does not name a dtype" in caplog.text


def test_subclass_loads_the_regular_way(tmp_path, monkeypatch, caplog):
    base, finetune = _save_finetune(tmp_path)

    class TunesLLM(GrootPolicy):
        def __init__(self, config):
            config.tune_llm = True
            super().__init__(config)

    expected = _tensors(_load_regular(finetune, monkeypatch, TunesLLM))
    built = _record_builds(monkeypatch)

    with caplog.at_level(logging.WARNING):
        policy = TunesLLM.from_pretrained(finetune)

    _assert_same(expected, _tensors(policy))
    assert built == [str(base)]
    assert "TunesLLM is a subclass" in caplog.text


def test_missing_groot_extra_gives_the_install_hint(tmp_path, monkeypatch):
    _, finetune = _save_finetune(tmp_path)
    # Without transformers, the package check fails and the config class has no loader.
    monkeypatch.setitem(_require_package_cache, "transformers", False)
    monkeypatch.setattr(modeling_groot, "GR00TN17Config", type("GR00TN17Config", (), {}))

    with pytest.raises(ImportError, match=r"lerobot\[groot\]"):
        GrootPolicy.from_pretrained(finetune)


def test_unset_device_is_reported(tmp_path):
    _, finetune = _save_finetune(tmp_path)
    config = GrootConfig.from_pretrained(finetune)
    config.device = None

    with pytest.raises(ValueError, match="device is unset"):
        GrootPolicy.from_pretrained(finetune, config=config)


def test_hub_options_reach_the_base_config(tmp_path, monkeypatch):
    base, finetune = _save_finetune(tmp_path)
    files = {name: base / name for name in ("config.json", "model.safetensors")}
    _name_base(finetune, _hub_base(monkeypatch, tmp_path, files))
    # Without strict, a missing weight makes the load fall back to the regular one, which builds the model
    # from the base weights.
    weights = load_file(finetune / "model.safetensors")
    weights.pop(next(key for key in weights if "action_head" in key))
    save_file(weights, finetune / "model.safetensors")
    download = modeling_groot.snapshot_download
    options = []

    def recording_download(repo_id, **kwargs):
        options.append(
            (repo_id, kwargs.get("cache_dir"), kwargs.get("local_files_only"), kwargs.get("token"))
        )
        return download(repo_id, **kwargs)

    monkeypatch.setattr(modeling_groot, "snapshot_download", recording_download)
    GrootPolicy.from_pretrained(
        finetune, cache_dir=tmp_path / "hub_base", local_files_only=True, token="hf_test", strict=False
    )

    # So a base model given by Hub id is read once, with the caller's cache, cache only or token options, also
    # when the load falls back to the regular one.
    assert options == [("someone/groot-base", tmp_path / "hub_base", True, "hf_test")]


def test_loader_options_reach_the_model(tmp_path, monkeypatch):
    base, finetune = _save_finetune(tmp_path)
    config = _policy_config(base)
    # None of these are the defaults, nor the defaults of GR00TN17.from_pretrained.
    config.tune_llm = True
    config.tune_visual = False
    config.tune_projector = False
    config.tune_diffusion_model = False
    config.tune_vlln = False
    config.rtc_ramp_rate = 2.5
    built = _record_builds(monkeypatch)

    policy = GrootPolicy.from_pretrained(finetune, config=config)

    assert built == [None]
    model = policy._groot_model
    assert all(parameter.requires_grad for parameter in model.backbone.language_model.parameters())
    assert not any(parameter.requires_grad for parameter in model.backbone.visual.parameters())
    for module in (model.action_head.state_encoder, model.action_head.model, model.action_head.vlln):
        assert not any(parameter.requires_grad for parameter in module.parameters())
    assert model.config.rtc_ramp_rate == 2.5


def test_save_that_stops_halfway_leaves_no_copies_in_use(tmp_path, monkeypatch):
    base = _write_tiny_base(tmp_path)
    other_base = shutil.copytree(base, tmp_path / "other_base")
    model_config = json.loads((other_base / "config.json").read_text())
    (other_base / "config.json").write_text(json.dumps({**model_config, "action_horizon": 2}))
    GrootPolicy(_policy_config(other_base)).save_pretrained(tmp_path / "finetune")
    policy = GrootPolicy(_policy_config(base))
    save_weights = pretrained.save_torch_state_dict

    def fail_after_the_weights(*args, **kwargs):
        save_weights(*args, **kwargs)
        raise OSError("No space left on device")

    with monkeypatch.context() as patch:
        patch.setattr(pretrained, "save_torch_state_dict", fail_after_the_weights)
        with pytest.raises(OSError, match="No space left"):
            policy.save_pretrained(tmp_path / "finetune")

    # The copy of the other base is still there, but config.json records another SHA-256, so the load reads
    # the base, as for a copy that another writer of the checkpoint left.
    assert GrootPolicy.from_pretrained(tmp_path / "finetune")._groot_model.config.action_horizon == 4


def test_save_on_another_rank_writes_nothing(tmp_path, monkeypatch):
    base, finetune = _save_finetune(tmp_path)
    saved = {path: path.read_bytes() for path in finetune.rglob("*") if path.is_file()}
    # Another policy, whose copy of the base config would differ from the saved one.
    other_base = shutil.copytree(base, tmp_path / "other_base")
    model_config = json.loads((other_base / "config.json").read_text())
    (other_base / "config.json").write_text(json.dumps({**model_config, "action_horizon": 2}))
    policy = GrootPolicy(_policy_config(other_base))
    monkeypatch.setattr("lerobot.distributed.utils.is_main_process", lambda: False)

    policy.save_pretrained(finetune)

    assert {path: path.read_bytes() for path in finetune.rglob("*") if path.is_file()} == saved


@pytest.mark.parametrize("layout", ["file_link", "folder_link"])
def test_copy_that_links_out_of_the_checkpoint_is_saved_as_built(tmp_path, layout):
    _, finetune = _save_finetune(tmp_path)
    copies = finetune / modeling_groot._BASE_CONFIG_DIR
    outside = tmp_path / "outside.json"
    model_config = json.loads((copies / "config.json").read_text())
    outside.write_text(json.dumps({**model_config, "action_horizon": 2}))
    if layout == "folder_link":
        elsewhere = Path(shutil.move(copies, tmp_path / "elsewhere"))
        copies.symlink_to(elsewhere)
        shutil.copyfile(outside, elsewhere / "config.json")
        linked = [outside, *elsewhere.iterdir()]
    else:
        (copies / "config.json").unlink()
        (copies / "config.json").symlink_to(outside)
        linked = [outside]
    _match_copy(finetune, outside)

    policy = GrootPolicy.from_pretrained(finetune)
    policy.save_pretrained(tmp_path / "resaved")

    assert policy._groot_model.config.action_horizon == 2
    # The save copies the config the model was built from, so loading it again builds the same model.
    assert GrootPolicy.from_pretrained(tmp_path / "resaved")._groot_model.config.action_horizon == 2
    # Saving back into the folder it was loaded from leaves the files the links point to as they were.
    before = {path: path.read_bytes() for path in linked}
    policy.save_pretrained(finetune)
    assert {path: path.read_bytes() for path in linked} == before
    assert not copies.is_symlink()


@pytest.mark.parametrize("named_by", ["base_model_path", "a_link_in_the_copies"])
def test_a_local_config_that_is_not_groot_is_never_read(tmp_path, monkeypatch, named_by):
    base, finetune = _save_finetune(tmp_path)
    copies = finetune / modeling_groot._BASE_CONFIG_DIR
    # Missing keys take the N1.7 defaults, so any JSON object builds a GR00T. This one builds the tiny one,
    # as credentials would next to a real fine-tune, but names no GR00T model.
    model_config = json.loads((copies / "config.json").read_text())
    del model_config["model_type"]
    secret = tmp_path / "secret" / "config.json"
    secret.parent.mkdir()
    secret_config = {**model_config, "action_horizon": 2, "auths": {"example": {"auth": "SECRET"}}}
    secret.write_text(json.dumps(secret_config))

    if named_by == "base_model_path":
        _name_base(finetune, str(secret.parent))
        # With base weights, so a load that read that config.json to build a model could finish.
        shutil.copyfile(base / "model.safetensors", secret.parent / "model.safetensors")
        built = _record_builds(monkeypatch)
        with pytest.raises(ValueError, match="is not a GR00T N1.7 model"):
            GrootPolicy.from_pretrained(finetune)
        assert built == []
    else:
        (copies / "config.json").unlink()
        (copies / "config.json").symlink_to(secret)
        _match_copy(finetune, secret)
        policy = GrootPolicy.from_pretrained(finetune)
        assert policy._groot_model.config.action_horizon == 4
        # A save replaces the link, so the file it points to stays as it was.
        policy.save_pretrained(finetune)
        assert json.loads(secret.read_text()) == secret_config


def test_hub_base_config_that_is_not_groot_is_refused(tmp_path, monkeypatch):
    base, finetune = _save_finetune(tmp_path)
    # Like the local one above: builds the tiny GR00T, but names no GR00T model.
    model_config = json.loads((base / "config.json").read_text())
    del model_config["model_type"]
    served = tmp_path / "served.json"
    served.write_text(json.dumps({**model_config, "auths": {"example": {"auth": "SECRET"}}}))
    _name_base(finetune, _hub_base(monkeypatch, tmp_path, {"config.json": served}))
    built = _record_builds(monkeypatch)

    with pytest.raises(ValueError, match="is not a GR00T N1.7 model"):
        GrootPolicy.from_pretrained(finetune)
    assert built == []


def test_hub_base_config_that_is_not_groot_is_refused_before_its_weights_download(tmp_path, monkeypatch):
    base = _write_tiny_base(tmp_path)
    model_config = json.loads((base / "config.json").read_text())
    del model_config["model_type"]
    served = tmp_path / "served.json"
    served.write_text(json.dumps(model_config))
    files = {"config.json": served, "model.safetensors": base / "model.safetensors"}
    built = _record_builds(monkeypatch)

    # A first run from it: nothing is cached yet.
    with pytest.raises(ValueError, match="is not a GR00T N1.7 model"):
        GrootPolicy(_policy_config(_hub_base(monkeypatch, tmp_path, files, on_download=True)))
    assert built == []
    assert not list((tmp_path / "hub_base").rglob("model.safetensors"))


def test_hub_base_config_that_cannot_be_downloaded_is_left_to_the_build(tmp_path, monkeypatch):
    # Offline, with nothing cached.
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path / "empty_cache"))
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)
    built = _record_builds(monkeypatch)

    with pytest.raises(OSError):
        GrootPolicy(_policy_config("someone/groot-base"))
    # The early read gives up quietly, so the build reads base_model_path itself and raises its own error.
    assert built == ["someone/groot-base"]


def test_finetune_without_a_copy_or_its_base_model_fails_the_regular_way(tmp_path, caplog):
    _, finetune = _save_finetune(tmp_path)
    # A base model on another machine, which is not a repo id either.
    _name_base(finetune, str(tmp_path / "elsewhere" / "base"))

    with caplog.at_level(logging.WARNING), pytest.raises(OSError):
        GrootPolicy.from_pretrained(finetune)
    assert "No config.json to read" in caplog.text


def test_save_copies_a_base_model_that_links_to_a_cache_snapshot(tmp_path):
    base = _write_tiny_base(tmp_path)
    # The base laid out like the Hugging Face cache, and given by a link to its snapshot.
    repo = tmp_path / "hub" / "models--someone--groot-base"
    snapshot = repo / "snapshots" / ("0" * 40)
    (repo / "blobs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    for path in list(base.iterdir()):
        blob = shutil.move(path, repo / "blobs" / path.name)
        (snapshot / path.name).symlink_to(os.path.relpath(blob, snapshot))
    (tmp_path / "linked_base").symlink_to(snapshot)

    GrootPolicy(_policy_config(tmp_path / "linked_base")).save_pretrained(tmp_path / "finetune")

    saved = tmp_path / "finetune" / modeling_groot._BASE_CONFIG_DIR
    assert sorted(path.name for path in saved.iterdir()) == ["config.json"]


def test_backbone_named_by_the_copies_runs_no_custom_code(tmp_path):
    base, finetune = _save_finetune(tmp_path)
    # A backbone folder whose config asks transformers to run a module that leaves a mark when imported. Its
    # name is unique, as transformers imports a module of the same folder name only once per process.
    backbone = shutil.copytree(base.parent / "backbone", tmp_path / f"backbone_{tmp_path.name}")
    mark = tmp_path / "custom_code_ran"
    (backbone / "custom_config.py").write_text(
        f"from transformers import Qwen3VLConfig\nopen({str(mark)!r}, 'w').close()\n"
        "class CustomConfig(Qwen3VLConfig):\n    pass\n"
    )
    backbone_config = json.loads((backbone / "config.json").read_text())
    backbone_config["auto_map"] = {"AutoConfig": "custom_config.CustomConfig"}
    (backbone / "config.json").write_text(json.dumps(backbone_config))
    copy = finetune / modeling_groot._BASE_CONFIG_DIR / "config.json"
    copy.write_text(json.dumps({**json.loads(copy.read_text()), "model_name": str(backbone)}))
    _match_copy(finetune, copy)

    GrootPolicy.from_pretrained(finetune)

    assert not mark.exists()
