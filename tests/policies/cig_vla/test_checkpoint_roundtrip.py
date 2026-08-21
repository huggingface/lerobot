from unittest.mock import patch

import torch
from conftest import MockBackbone, make_batch, make_policy

from lerobot.policies.cig_vla.modeling_cig_vla import CIGVLAPolicy


def test_checkpoint_roundtrip(tmp_path):
    policy = make_policy(detach_main=False, detach_causal=True).eval()
    batch = make_batch(labels=False)
    batch = {key: value[:1] if isinstance(value, torch.Tensor) else value[:1] for key, value in batch.items()}
    before_geometry = policy.predict_geometric_bottleneck(batch)
    torch.manual_seed(17)
    before_action = policy.predict_action_chunk(batch)
    policy.save_pretrained(tmp_path)
    with patch(
        "lerobot.policies.cig_vla.modeling_interaction_cig_vla.Qwen3VLGroundingBackbone", MockBackbone
    ):
        loaded = CIGVLAPolicy.from_pretrained(tmp_path, strict=True)
    assert loaded.config.type == "cig_vla"
    assert loaded.config.grounding_architecture == policy.config.grounding_architecture
    assert loaded.config.detach_bottleneck_for_main_action is False
    assert loaded.config.detach_bottleneck_for_causal_branch is True
    after_geometry = loaded.predict_geometric_bottleneck(batch)
    torch.testing.assert_close(before_geometry.translation_goal, after_geometry.translation_goal)
    torch.manual_seed(17)
    after_action = loaded.predict_action_chunk(batch)
    torch.testing.assert_close(before_action, after_action)
