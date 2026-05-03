import importlib
import os
from unittest.mock import MagicMock, patch

import pytest
from safetensors.torch import load_file

from .utils import skip_if_package_missing

# Skip this entire module in CI
pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="This test requires peft and is very slow, not meant for CI",
)


def run_command(cmd, module, args):
    module = importlib.import_module(f"lerobot.scripts.{module}")
    with patch("sys.argv", [cmd] + args):
        module.main()


def lerobot_train(args):
    return run_command(cmd="lerobot-train", module="lerobot_train", args=args)


def resolve_model_id_for_peft_training(policy_type):
    """PEFT training needs pretrained models, this finds the pretrained model of a policy type for PEFT training."""
    if policy_type == "smolvla":
        return "lerobot/smolvla_base"

    raise ValueError(f"No pretrained model known for {policy_type}. PEFT training will not work.")


@pytest.mark.parametrize("policy_type", ["smolvla"])
@skip_if_package_missing("peft")
def test_peft_training_push_to_hub_works(policy_type, tmp_path):
    """Ensure that push to hub stores PEFT only the adapter, not the full model weights."""
    output_dir = tmp_path / f"output_{policy_type}"
    upload_folder_contents = set()

    model_id = resolve_model_id_for_peft_training(policy_type)

    def mock_upload_folder(*args, **kwargs):
        folder_path = kwargs["folder_path"]
        # we include more than is actually uploaded since we ignore {allow,ignore}_patterns of upload_folders()
        upload_folder_contents.update(os.listdir(folder_path))
        return MagicMock()

    with (
        patch("huggingface_hub.HfApi.create_repo"),
        patch("huggingface_hub.HfApi.upload_folder", mock_upload_folder),
    ):
        lerobot_train(
            [
                f"--policy.path={model_id}",
                "--policy.push_to_hub=true",
                "--policy.repo_id=foo/bar",
                "--policy.input_features=null",
                "--policy.output_features=null",
                "--peft.method=LORA",
                "--dataset.repo_id=lerobot/pusht",
                "--dataset.episodes=[0, 1]",
                "--steps=1",
                f"--output_dir={output_dir}",
            ]
        )

        assert "adapter_model.safetensors" in upload_folder_contents
        assert "config.json" in upload_folder_contents
        assert "adapter_config.json" in upload_folder_contents


@pytest.mark.parametrize("policy_type", ["smolvla"])
@skip_if_package_missing("peft")
def test_peft_training_works(policy_type, tmp_path):
    """Check whether the standard case of fine-tuning a (partially) pre-trained policy with PEFT works."""
    output_dir = tmp_path / f"output_{policy_type}"
    model_id = resolve_model_id_for_peft_training(policy_type)

    lerobot_train(
        [
            f"--policy.path={model_id}",
            "--policy.push_to_hub=false",
            "--policy.input_features=null",
            "--policy.output_features=null",
            "--peft.method=LORA",
            "--dataset.repo_id=lerobot/pusht",
            "--dataset.episodes=[0, 1]",
            "--steps=1",
            f"--output_dir={output_dir}",
        ]
    )

    policy_dir = output_dir / "checkpoints" / "last" / "pretrained_model"

    for file in ["adapter_config.json", "adapter_model.safetensors", "config.json"]:
        assert (policy_dir / file).exists()

    # This is the default case where we train a pre-trained policy from scratch with new data.
    # We assume that we target policy-specific modules but fully fine-tune action and state projections
    # so these must be part of the trained state dict.
    state_dict = load_file(policy_dir / "adapter_model.safetensors")

    adapted_keys = [
        "state_proj",
        "action_in_proj",
        "action_out_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
    ]

    found_keys = [
        module_key
        for module_key in adapted_keys
        for state_dict_key in state_dict
        if f".{module_key}." in state_dict_key
    ]

    assert set(found_keys) == set(adapted_keys)


@pytest.mark.parametrize("policy_type", ["smolvla"])
@skip_if_package_missing("peft")
def test_peft_training_params_are_fewer(policy_type, tmp_path):
    """Check whether the standard case of fine-tuning a (partially) pre-trained policy with PEFT works."""
    output_dir = tmp_path / f"output_{policy_type}"
    model_id = resolve_model_id_for_peft_training(policy_type)

    def dummy_update_policy(
        train_metrics, policy, batch, optimizer, grad_clip_norm: float, accelerator, **kwargs
    ):
        params_total = sum(p.numel() for p in policy.parameters())
        params_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)

        assert params_total > params_trainable

        return train_metrics, {}

    with patch("lerobot.scripts.lerobot_train.update_policy", dummy_update_policy):
        lerobot_train(
            [
                f"--policy.path={model_id}",
                "--policy.push_to_hub=false",
                "--policy.input_features=null",
                "--policy.output_features=null",
                "--peft.method=LORA",
                "--dataset.repo_id=lerobot/pusht",
                "--dataset.episodes=[0, 1]",
                "--steps=1",
                f"--output_dir={output_dir}",
            ]
        )
