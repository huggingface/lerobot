"""Regression checks for layout-only changes; no checkpoints or GPU required."""

import copy
import importlib
import pickle

import pytest
import torch

from lerobot.optim.lingbot import LingbotMuonConfig


def test_muon_legacy_import_is_same_module(monkeypatch):
    legacy = importlib.import_module("lerobot.optim.lingbot_muon")
    canonical = importlib.import_module("lerobot.optim.muon")
    assert legacy is canonical
    assert legacy.DistributedMuon is canonical.DistributedMuon
    assert legacy._get_dtensor_shard_info is canonical._get_dtensor_shard_info
    monkeypatch.setattr(legacy, "_MEGABATCH_MAX_GROUP_SIZE", 7)
    assert canonical._MEGABATCH_MAX_GROUP_SIZE == 7
    # A class reference pickled under the old import path remains resolvable.
    assert pickle.loads(b"clerobot.optim.lingbot_muon\nDistributedMuon\n.") is canonical.DistributedMuon


@pytest.mark.parametrize("shape", [(7, 5), (3, 7, 5)])
def test_muon_state_roundtrip(shape):
    module = importlib.import_module("lerobot.optim.muon")
    torch.manual_seed(123)
    param = torch.nn.Parameter(torch.randn(shape))
    opt = module.DistributedMuon([param], lr=1e-4)
    param.grad = torch.randn_like(param)
    opt.step()
    restored_param = torch.nn.Parameter(param.detach().clone())
    restored = module.DistributedMuon([restored_param], lr=1e-4)
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    param.grad = torch.randn_like(param)
    restored_param.grad = param.grad.clone()
    opt.step()
    restored.step()
    torch.testing.assert_close(param, restored_param, rtol=0, atol=0)


def test_combined_optimizer_group_order_and_shared_state():
    params = {
        "model.linear.weight": torch.nn.Parameter(torch.randn(7, 5)),
        "model.linear.bias": torch.nn.Parameter(torch.randn(7)),
        "model.layers.0.mlp.experts.weight": torch.nn.Parameter(torch.randn(3, 7, 5)),
    }
    scale = (32 / 4) ** 0.5
    opt = LingbotMuonConfig(expert_lr_scale=scale).build(params)
    expected = [("adamw", 1e-4), ("muon", 1e-4), ("muon", 1e-4 * scale)]
    assert [(g["kind"], g["lr"]) for g in opt.param_groups] == expected
    for param in params.values():
        param.grad = torch.ones_like(param)
    opt.step()
    opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    assert [(g["kind"], g["lr"]) for g in opt.param_groups] == expected
    assert all(inner.state is opt.state for inner in opt._inner)
    assert all(
        any(group is parent for parent in opt.param_groups)
        for inner in opt._inner
        for group in inner.param_groups
    )
