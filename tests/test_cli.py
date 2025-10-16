import importlib
from unittest.mock import patch

import pytest
from safetensors.torch import load_file


def run_command(cmd, module, args):
    module = importlib.import_module(f"lerobot.scripts.{module}")
    with patch("sys.argv", [cmd] + args):
        module.main()


def lerobot_train(args):
    return run_command(cmd="lerobot-train", module="lerobot_train", args=args)


def lerobot_record(args):
    return run_command(cmd="lerobot-record", module="lerobot_record", args=args)


@pytest.mark.parametrize("policy_type", ["smolvla"])
def test_peft_training_works(policy_type, tmp_path):
    """Check whether the standard case of fine-tuning a (partially) pre-trained policy with PEFT works."""
    output_dir = tmp_path / f"output_{policy_type}"

    lerobot_train(
        [
            f"--policy.type={policy_type}",
            "--policy.push_to_hub=false",
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

    fully_trained_keys = [
        "state_proj",
        "action_in_proj",
        "action_out_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
    ]

    found_keys = [
        module_key
        for module_key in fully_trained_keys
        for state_dict_key in state_dict
        if f".{module_key}." in state_dict_key
    ]

    assert set(found_keys) == set(fully_trained_keys)


class DummyRobot:
    name = "dummy"
    cameras = []
    action_features = {"foo": 1.0, "bar": 2.0}
    observation_features = {"obs1": 1.0, "obs2": 2.0}

    def connect(self, *args):
        pass

    def disconnect(self):
        pass


def dummy_make_robot_from_config(*args, **kwargs):
    return DummyRobot()


@pytest.mark.parametrize("policy_type", ["smolvla"])
def test_peft_record_loads_policy(policy_type, tmp_path):
    """Train a policy with PEFT and attempt to load it with `lerobot-record`."""
    from peft import PeftModel

    output_dir = tmp_path / f"output_{policy_type}"

    lerobot_train(
        [
            f"--policy.type={policy_type}",
            "--policy.push_to_hub=false",
            "--peft.method=LORA",
            "--dataset.repo_id=lerobot/pusht",
            "--dataset.episodes=[0, 1]",
            "--steps=1",
            f"--output_dir={output_dir}",
        ]
    )

    policy_dir = output_dir / "checkpoints" / "last" / "pretrained_model"
    dataset_dir = tmp_path / "eval_pusht"
    single_task = "move the table"
    loaded_policy = None

    def dummy_record_loop(*args, **kwargs):
        nonlocal loaded_policy

        if "dataset" not in kwargs:
            return

        dataset = kwargs["dataset"]
        dataset.add_frame({"task": single_task})
        loaded_policy = kwargs["policy"]

    with (
        patch("lerobot.robots.make_robot_from_config", dummy_make_robot_from_config),
        # disable record loop since we're only interested in successful loading of the policy.
        patch("lerobot.scripts.lerobot_record.record_loop", dummy_record_loop),
        # disable speech output
        patch("lerobot.utils.utils.say"),
    ):
        lerobot_record(
            [
                f"--policy.path={policy_dir}",
                "--robot.type=so101_follower",
                "--robot.port=/dev/null",
                "--dataset.repo_id=lerobot/eval_pusht",
                f'--dataset.single_task="{single_task}"',
                f"--dataset.root={dataset_dir}",
                "--dataset.push_to_hub=false",
            ]
        )

        assert isinstance(loaded_policy, PeftModel)
