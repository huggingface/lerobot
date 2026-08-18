from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("datasets", reason="lerobot_train imports the dataset module")
pytest.importorskip("accelerate", reason="train() requires the training extra")

from lerobot.scripts import lerobot_train


class StopAfterProcessorCreationError(Exception):
    pass


class DummyAccelerator:
    is_main_process = True
    device = torch.device("cpu")

    def wait_for_everyone(self):
        pass


def test_train_passes_action_tokenizer_name_to_preprocessor(monkeypatch):
    active_cfg = SimpleNamespace(
        pretrained_path="pi0fast-base",
        pretrained_revision=None,
        device="cpu",
        action_tokenizer_name="lerobot/fast-action-tokenizer",
    )
    cfg = SimpleNamespace(
        policy=active_cfg,
        trainable_config=active_cfg,
        reward_model=None,
        is_reward_model_training=False,
        resume=False,
        peft=None,
        rename_map={},
        wandb=SimpleNamespace(enable=False, project=None),
        seed=None,
        cudnn_deterministic=False,
        eval_freq=0,
        env=None,
        validate=lambda: None,
        to_dict=lambda: {},
    )
    dataset = SimpleNamespace(meta=SimpleNamespace(stats={"observation.state": {"mean": 0.0}}))
    policy = SimpleNamespace(
        config=SimpleNamespace(
            input_features={"observation.state": None},
            output_features={"action": None},
            normalization_mapping={},
        )
    )

    monkeypatch.setattr(lerobot_train, "init_logging", lambda accelerator: None)
    monkeypatch.setattr(lerobot_train, "make_dataset", lambda cfg: dataset)
    monkeypatch.setattr(lerobot_train, "make_policy", lambda **kwargs: policy)

    def assert_preprocessor_overrides(**kwargs):
        assert kwargs["policy_cfg"] is active_cfg
        assert kwargs["pretrained_path"] == "pi0fast-base"
        assert kwargs["preprocessor_overrides"]["action_tokenizer_processor"] == {
            "action_tokenizer_name": "lerobot/fast-action-tokenizer",
        }
        raise StopAfterProcessorCreationError

    monkeypatch.setattr(lerobot_train, "make_pre_post_processors", assert_preprocessor_overrides)

    with pytest.raises(StopAfterProcessorCreationError):
        lerobot_train.train.__wrapped__(cfg, accelerator=DummyAccelerator())
