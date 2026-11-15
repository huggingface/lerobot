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

"""LoRA fine-tuning of the praxis policy through PEFT, on the tiny random model."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lerobot.policies.praxis.modeling_praxis import PraxisPolicy

from .test_praxis_training import load_example, training_batch
from .tiny_model import FakeAutoencoder, FakeTextEncoder, make_tiny_policy

pytest.importorskip("transformers", reason="praxis requires the praxis extra (transformers)")
peft = pytest.importorskip("peft")


def _wrap(policy: PraxisPolicy, tmp_path: Path, **overrides):
    policy.config.pretrained_path = tmp_path  # PEFT training needs a checkpoint to start from
    return policy.wrap_with_peft(peft_cli_overrides={"r": 4, "lora_alpha": 8, **overrides})


def test_default_targets_are_the_action_expert_projections(tmp_path: Path) -> None:
    policy = make_tiny_policy(seed=1)
    n_blocks = len(policy.model.action_expert.blocks)
    peft_model = _wrap(policy, tmp_path)
    lora_layers = [n for n, m in peft_model.named_modules() if isinstance(m, peft.tuners.lora.Linear)]
    assert len(lora_layers) == 5 * n_blocks
    assert all("action_expert.blocks." in n for n in lora_layers)
    assert all(n.rsplit(".", 1)[1] in {"to_q", "to_kv", "to_out", "mlp_in", "mlp_out"} for n in lora_layers)
    trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    params = peft_model.get_optim_params()
    assert len(params) == len(trainable)


def test_lora_starts_at_the_base_model_and_trains(tmp_path: Path) -> None:
    base = make_tiny_policy(seed=2, text_dropout=0.0, proprio_dropout=0.0, activation_checkpointing=False)
    base.train()
    batch = training_batch(base, seed=3)
    loss_base, _ = base.forward(batch)

    policy = make_tiny_policy(seed=2, text_dropout=0.0, proprio_dropout=0.0, activation_checkpointing=False)
    peft_model = _wrap(policy, tmp_path)
    peft_model.train()
    loss_lora, _ = peft_model(batch)
    assert torch.equal(loss_base.detach(), loss_lora.detach())  # B = 0: the adapter is the identity
    loss_lora.backward()
    grads = [p.grad for n, p in peft_model.named_parameters() if p.requires_grad]
    assert all(g is not None for g in grads) and any(g.abs().sum() > 0 for g in grads)
    base_grads = [
        p.grad for n, p in peft_model.named_parameters() if not p.requires_grad and p.grad is not None
    ]
    assert not base_grads
    peft_model.update()  # the routing-bias hook reaches the base policy through PEFT


def test_merge_folds_the_adapter_into_the_weights(tmp_path: Path) -> None:
    merge_adapter = load_example("merge_praxis_adapter").merge_adapter

    policy = make_tiny_policy(seed=4)
    base_dir = tmp_path / "base"
    policy.save_pretrained(base_dir)
    peft_model = _wrap(policy, tmp_path)
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn_like(param) * 0.1)
    adapter_dir = tmp_path / "adapter"
    peft_model.save_pretrained(adapter_dir)
    assert (adapter_dir / "adapter_config.json").is_file()
    expected = {k: v.clone() for k, v in peft_model.merge_and_unload().state_dict().items()}
    peft_model = None

    out = merge_adapter(base_dir, adapter_dir, tmp_path / "merged")
    merged = PraxisPolicy.from_pretrained(out, strict=True, load_frozen_parts=False)
    merged.vae, merged.text_encoder = FakeAutoencoder(), FakeTextEncoder()
    base_state = PraxisPolicy.from_pretrained(base_dir, strict=True, load_frozen_parts=False).state_dict()
    got = merged.state_dict()
    assert set(got) == set(expected)
    changed = 0
    for name in expected:
        assert torch.equal(got[name], expected[name]), name
        if "action_expert" in name and not torch.equal(got[name], base_state[name]):
            changed += 1
    assert changed > 0


def test_adapter_fine_tune_keeps_the_router_biases_and_saves_the_counter(tmp_path) -> None:
    """Under PEFT the MoE routing biases of the base stay fixed (an adapter checkpoint could not carry them)
    and the training counter is part of the adapter (`modules_to_save`)."""
    from safetensors.torch import load_file

    from lerobot.policies.praxis.praxis_core.layers import MoEFeedForward

    policy = make_tiny_policy(seed=30)
    peft_policy = _wrap(policy, tmp_path, r=2, lora_alpha=4)
    base = peft_policy.get_base_model()
    assert base._peft_active()
    biases = [m.expert_bias.clone() for m in base.model.modules() if isinstance(m, MoEFeedForward)]
    assert biases, "the tiny model has MoE blocks"
    batch = training_batch(base, seed=31)
    base.train()
    loss, _ = base(batch)
    loss.backward()
    base.update()
    for module, before in zip(
        [m for m in base.model.modules() if isinstance(m, MoEFeedForward)], biases, strict=True
    ):
        assert torch.equal(module.expert_bias, before)
    assert int(base.train_counter.calls) == 1
    peft_policy.save_pretrained(tmp_path)
    saved = load_file(str(tmp_path / "adapter_model.safetensors"))
    counter_keys = [k for k in saved if "train_counter" in k]
    assert counter_keys and int(saved[counter_keys[0]]) == 1


def test_peft_config_must_keep_the_counter(tmp_path: Path) -> None:
    """`--peft.full_training_modules=[]` would leave the training counter out of the adapter."""
    policy = make_tiny_policy(seed=30)
    with pytest.raises(ValueError, match="train_counter"):
        _wrap(policy, tmp_path, full_training_modules=[])
    with pytest.raises(ValueError, match="training counter and the rig heads"):
        _wrap(
            make_tiny_policy(seed=30),
            tmp_path,
            full_training_modules=["train_counter", "model.action_expert"],
        )


def test_lora_from_the_base_trains_the_rig_heads(tmp_path: Path) -> None:
    """A fine-tune from the base checkpoint names the six new head tensors in `full_training_modules`:
    they train in full next to the adapters, the adapter carries them, and the merge keeps them."""
    from safetensors.torch import load_file

    from lerobot.policies.praxis.praxis_core.checkpoint import BASE_HEAD_TENSORS

    merge_adapter = load_example("merge_praxis_adapter").merge_adapter
    policy = make_tiny_policy(seed=40)
    base_dir = tmp_path / "base"
    policy.save_pretrained(base_dir)
    heads = list(PraxisPolicy.PEFT_HEAD_MODULES)
    peft_model = _wrap(policy, tmp_path, full_training_modules=["train_counter", *heads])
    trainable = {n for n, p in peft_model.named_parameters() if p.requires_grad}
    head_params = {n for n in trainable if "modules_to_save" in n and "train_counter" not in n}
    assert len(head_params) == 6, sorted(head_params)
    assert all(any(f"{h}.modules_to_save" in n for h in heads) for n in head_params)
    assert all("lora_" in n or "modules_to_save" in n for n in trainable)
    assert len(peft_model.get_optim_params()) == len(trainable)

    base = peft_model.get_base_model()
    before = {n: p.detach().clone() for n, p in peft_model.named_parameters() if n in head_params}
    optimizer = torch.optim.SGD(peft_model.get_optim_params(), lr=0.1)
    base.train()
    loss, _ = base(training_batch(base, seed=41))
    loss.backward()
    optimizer.step()
    base.update()
    after = dict(peft_model.named_parameters())
    assert all(not torch.equal(before[n], after[n]) for n in head_params), "the heads train"

    adapter_dir = tmp_path / "adapter"
    peft_model.save_pretrained(adapter_dir)
    saved = load_file(str(adapter_dir / "adapter_model.safetensors"))
    assert sum(1 for k in saved if any(h in k for h in heads)) == 6, "the adapter carries the heads"

    expected = {k: v.clone() for k, v in peft_model.merge_and_unload().state_dict().items()}
    peft_model = None
    out = merge_adapter(base_dir, adapter_dir, tmp_path / "merged")
    merged = PraxisPolicy.from_pretrained(out, strict=True, load_frozen_parts=False)
    base_state = PraxisPolicy.from_pretrained(base_dir, strict=True, load_frozen_parts=False).state_dict()
    got = merged.state_dict()
    head_keys = [n for n in got if any(n.endswith(t) for t in BASE_HEAD_TENSORS)]
    assert len(head_keys) == 6
    for name in head_keys:
        assert torch.equal(got[name], expected[name]), name
        assert not torch.equal(got[name], base_state[name]), name


def test_trainable_parameters_stay_the_adapters_under_peft(tmp_path: Path) -> None:
    """With adapters injected, a model whose parameters are all frozen (an adapter loaded for
    inference) does not turn the 27B base weights trainable."""
    policy = make_tiny_policy(seed=31)
    peft_model = _wrap(policy, tmp_path)
    base = peft_model.get_base_model()
    assert base._peft_active()
    for parameter in base.model.parameters():
        parameter.requires_grad_(False)
    assert base._trainable_parameters() == []
    assert not any(parameter.requires_grad for parameter in base.model.parameters())
