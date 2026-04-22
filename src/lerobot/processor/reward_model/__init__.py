"""Reward model processor steps for HIL-SERL.

Pluggable reward signals:
- ``height_gripper``: state-based (EE height + gripper closed)
- ``cnn``: vision-based CNN classifier
- ``sarm``: stage-aware reward model (continuous progress 0→1)

Importing this module installs a monkey-patch on upstream
``Classifier.predict_reward`` so checkpoint loading works with the external
``classifier_preprocessor`` pipeline. See ``_classifier_patch``.
"""

from lerobot.processor.reward_model._classifier_patch import apply as _apply_classifier_patch
from lerobot.processor.reward_model.base import (
    BaseRewardProcessorStep,
    RewardModelConfig,
)
from lerobot.processor.reward_model.cnn_classifier import (
    CNNRewardConfig,
    CNNRewardProcessorStep,
)
from lerobot.processor.reward_model.height_gripper import (
    HeightGripperRewardConfig,
    HeightGripperRewardStep,
)
from lerobot.processor.reward_model.sarm import (
    SARMRewardConfig,
    SARMRewardProcessorStep,
)

_apply_classifier_patch()


def _filter_cfg(cfg_dict: dict, target_cls) -> dict:
    import dataclasses

    fields = {f.name for f in dataclasses.fields(target_cls)}
    return {k: v for k, v in cfg_dict.items() if k in fields}


def build_reward_model_step(cfg: dict | None):
    """Dispatch a reward model config dict to its processor step.

    Args:
        cfg: dict with a ``type`` key in {"manual", "height_gripper", "cnn", "sarm"}
             and (possibly extra) type-specific fields. Extra keys are ignored.
             If ``None`` / empty / ``type in {"manual", "none"}``, returns ``None``.

    Returns:
        A ``BaseRewardProcessorStep`` subclass instance, or ``None``.
    """
    if not cfg:
        return None
    cfg = dict(cfg)
    rtype = cfg.get("type")
    terminate = bool(cfg.pop("terminate_on_success", True))

    if rtype in (None, "manual", "none"):
        return None
    if rtype == "height_gripper":
        return HeightGripperRewardStep(
            config=HeightGripperRewardConfig(**_filter_cfg(cfg, HeightGripperRewardConfig)),
            terminate_on_success=terminate,
        )
    if rtype == "cnn":
        return CNNRewardProcessorStep(
            config=CNNRewardConfig(**_filter_cfg(cfg, CNNRewardConfig)),
            terminate_on_success=terminate,
        )
    if rtype == "sarm":
        return SARMRewardProcessorStep(
            config=SARMRewardConfig(**_filter_cfg(cfg, SARMRewardConfig)),
            terminate_on_success=terminate,
        )
    raise ValueError(f"Unknown reward_model.type: {rtype!r}")


__all__ = [
    "BaseRewardProcessorStep",
    "CNNRewardConfig",
    "CNNRewardProcessorStep",
    "HeightGripperRewardConfig",
    "HeightGripperRewardStep",
    "RewardModelConfig",
    "SARMRewardConfig",
    "SARMRewardProcessorStep",
    "build_reward_model_step",
]
