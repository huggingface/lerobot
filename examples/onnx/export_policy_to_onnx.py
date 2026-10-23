# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Export a LeRobot policy to ONNX and verify the exported graph with onnxruntime.

The exported graph takes flat tensors — the observation state followed by one tensor
per camera image — and returns the predicted action chunk. Inputs and outputs live in
the policy's normalized observation/action space; see `docs/source/onnx_export.mdx`
for how to apply dataset statistics around the graph.

Examples:
    # Export a randomly initialized ACT policy (no download needed)
    python examples/onnx/export_policy_to_onnx.py --output act_pusht.onnx

    # Export a pretrained policy from the hub
    python examples/onnx/export_policy_to_onnx.py \
        --from-pretrained lerobot/act_aloha_sim_insertion_human \
        --output act_aloha.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features


class PolicyWrapper(torch.nn.Module):
    """Expose a flat forward signature so the policy can be traced with torch.onnx.export.

    The policy expects a batch dict keyed by feature names; positional tensors are
    easier to bind in ONNX runtimes, so this wrapper rebuilds the dict.
    """

    def __init__(self, policy: ACTPolicy, input_keys: list[str]):
        super().__init__()
        self.policy = policy
        self.input_keys = input_keys

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        batch = dict(zip(self.input_keys, inputs, strict=True))
        # Returns the full (batch, chunk_size, action_dim) action chunk. Temporal
        # ensembling is a queueing/weighting scheme applied by ACTPolicy at runtime
        # and is intentionally not part of the exported graph.
        return self.policy.predict_action_chunk(batch)


def input_feature_keys(policy: ACTPolicy) -> list[str]:
    """Ordered flat input keys: the state feature, then camera images."""
    state_keys = [key for key, ft in policy.config.input_features.items() if ft.type is FeatureType.STATE]
    image_keys = [key for key, ft in policy.config.input_features.items() if ft.type is FeatureType.VISUAL]
    other_keys = [
        key
        for key, ft in policy.config.input_features.items()
        if ft.type not in (FeatureType.STATE, FeatureType.VISUAL)
    ]
    if other_keys:
        raise NotImplementedError(
            f"This example only handles state and visual features, got: {other_keys}. "
            "Extend the wrapper to pass them as additional flat inputs."
        )
    if len(state_keys) != 1:
        raise NotImplementedError(
            "ACT expects exactly one state feature (its forward pass reads "
            f"observation.state unconditionally), got: {state_keys}."
        )
    return state_keys + image_keys


def build_policy(args) -> ACTPolicy:
    if args.from_pretrained is not None:
        # A pretrained policy carries its own features; no dataset metadata needed.
        return ACTPolicy.from_pretrained(args.from_pretrained)

    # When starting from scratch we size the policy from a dataset's features, the
    # same way `examples/training/train_policy.py` does before training.
    dataset_metadata = LeRobotDatasetMetadata(args.dataset)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        # Skip the ImageNet weight download for this self-contained example; pass
        # `pretrained_backbone_weights` (or load a trained checkpoint) for real use.
        pretrained_backbone_weights=None,
    )
    return ACTPolicy(config)


def make_dummy_inputs(policy: ACTPolicy, input_keys: list[str]) -> tuple[torch.Tensor, ...]:
    """One-sample dummy tensors matching the feature shapes, used to trace the graph."""
    return tuple(
        torch.zeros(1, *policy.config.input_features[key].shape, dtype=torch.float32) for key in input_keys
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-pretrained",
        type=str,
        default=None,
        help="Hub id or local path of a pretrained ACT policy",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="lerobot/pusht",
        help="Dataset whose features size the policy when not starting from a checkpoint",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/onnx/act_policy.onnx", help="Output .onnx path"
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--atol", type=float, default=1e-4, help="Max allowed absolute diff between PyTorch and ONNX outputs"
    )
    args = parser.parse_args()

    policy = build_policy(args)
    policy.eval()

    input_names = input_feature_keys(policy)
    wrapper = PolicyWrapper(policy, input_names).eval()
    sample_inputs = make_dummy_inputs(policy, input_names)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = {name: {0: "batch"} for name in input_names}
    dynamic_axes["action_chunk"] = {0: "batch"}
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            sample_inputs,
            str(output_path),
            opset_version=args.opset,
            input_names=input_names,
            output_names=["action_chunk"],
            dynamic_axes=dynamic_axes,
            # The classic TorchScript-based exporter handles the wrapper's
            # tuple-of-tensors signature; the dynamo exporter (default in newer
            # torch) requires stricter input structure.
            dynamo=False,
        )

    # Verify: onnxruntime must reproduce the PyTorch output, on random inputs
    # (zeros would make an untrained graph coincidentally look correct).
    torch.manual_seed(42)
    random_inputs = tuple(torch.randn_like(tensor) for tensor in sample_inputs)
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        name: tensor.numpy() for name, tensor in zip(session.get_inputs(), random_inputs, strict=True)
    }
    ort_output = session.run(None, ort_inputs)[0]
    with torch.inference_mode():
        torch_output = wrapper(*random_inputs).numpy()
    max_abs_diff = float(np.max(np.abs(ort_output - torch_output)))
    if max_abs_diff > args.atol:
        raise RuntimeError(
            f"ONNX output diverges from PyTorch by {max_abs_diff:.2e} (tolerance {args.atol:.2e})"
        )

    print(f"Exported policy to {output_path}")
    print(f"Inputs: {input_names}")
    print(f"PyTorch vs onnxruntime max abs diff: {max_abs_diff:.2e} (atol {args.atol:.2e})")


if __name__ == "__main__":
    main()
