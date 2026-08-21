import torch
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.cig_vla.configuration_cig_vla import CIGVLAConfig
from lerobot.policies.cig_vla.modeling_cig_vla import CIGVLAPolicy


class MockBackbone(nn.Module):
    hidden_size = 16

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.projection = nn.Linear(1, self.hidden_size)

    def encode_multimodal(self, images, instructions):
        values = torch.stack([sample[0].mean().reshape(1) for sample in images])
        token = self.projection(values)
        hidden = token[:, None].expand(-1, 5, -1)
        return hidden, torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)


def make_config(detach_main=True, detach_causal=True):
    return CIGVLAConfig(
        device="cpu",
        input_features={
            "observation.image": PolicyFeature(FeatureType.VISUAL, (3, 8, 8)),
            "observation.wrist_image": PolicyFeature(FeatureType.VISUAL, (3, 8, 8)),
            "observation.state": PolicyFeature(FeatureType.STATE, (5,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        chunk_size=4,
        n_action_steps=2,
        grounding_hidden_dim=16,
        grounding_num_heads=4,
        grounding_num_layers=1,
        controller_hidden_dim=16,
        controller_num_layers=1,
        controller_num_heads=4,
        num_inference_steps=2,
        detach_bottleneck_for_main_action=detach_main,
        detach_bottleneck_for_causal_branch=detach_causal,
        causal_response_margin=10.0,
    )


def make_policy(detach_main=True, detach_causal=True):
    return CIGVLAPolicy(
        make_config(detach_main, detach_causal),
        backbone=MockBackbone(),
        dataset_stats={"action": {"mean": torch.zeros(7), "std": torch.ones(7)}},
    )


def make_batch(labels=True):
    batch = {
        "observation.image": torch.randn(2, 3, 8, 8),
        "observation.wrist_image": torch.randn(2, 3, 8, 8),
        "observation.state": torch.randn(2, 5),
        "task": ["pick bowl", "pick bowl"],
        "action": torch.randn(2, 4, 7),
        "action_is_pad": torch.tensor([[False, False, False, True], [False, False, True, True]]),
    }
    del labels
    return batch
