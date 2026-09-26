"""The dataset's relative-action declaration flows into the policy config, once."""

from types import SimpleNamespace

import pytest

from lerobot.policies.factory import _apply_dataset_relative_action_provenance

GROUPS = [[0, 1, 2, 3, 4, 5]]


def _cfg(**overrides):
    return SimpleNamespace(**{"use_relative_actions": True, "relative_se3_pose_groups": [], **overrides})


def _meta(groups=GROUPS):
    declaration = {"chunk_size": 50, "exclude_joints": ["gripper"], "se3_pose_groups": groups}
    return SimpleNamespace(info=SimpleNamespace(relative_action=declaration))


def test_an_unset_config_inherits_the_declaration_and_a_contradicting_one_raises():
    cfg = _cfg()
    _apply_dataset_relative_action_provenance(cfg, _meta())
    assert cfg.relative_se3_pose_groups == GROUPS

    # Stating the same layout again is allowed; stating a different one is not, because the
    # statistics only describe the declared groups.
    _apply_dataset_relative_action_provenance(_cfg(relative_se3_pose_groups=GROUPS), _meta())
    with pytest.raises(ValueError, match="contradicts the dataset"):
        _apply_dataset_relative_action_provenance(
            _cfg(relative_se3_pose_groups=[[1, 2, 3, 4, 5, 6]]), _meta()
        )


@pytest.mark.parametrize(
    ("cfg", "ds_meta"),
    [
        (_cfg(use_relative_actions=False), _meta()),
        (_cfg(), SimpleNamespace(info=SimpleNamespace(relative_action=None))),
    ],
    ids=["absolute policy", "dataset without a declaration"],
)
def test_left_alone(cfg, ds_meta):
    _apply_dataset_relative_action_provenance(cfg, ds_meta)
    assert cfg.relative_se3_pose_groups == []
