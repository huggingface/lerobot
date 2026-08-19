import pytest
import torch

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.xvla.processor_xvla import (
    LiberoProcessorStep,
    XVLAAddDomainIdProcessorStep,
    XVLARotation6DToAxisAngleProcessorStep,
)
from lerobot.policies.xvla_rmoe.configuration_xvla_rmoe import XVLARMoEConfig
from lerobot.policies.xvla_rmoe.processor_xvlarmoe import (
    DEFAULT_LIBERO_RENAME_MAP,
    XVLARMoEDatasetStateToPretrainedProcessorStep,
    XVLARMoEEE6DToLiberoActionProcessorStep,
    XVLARMoEImageProcessorStep,
    XVLARMoELiberoActionToEE6DProcessorStep,
    make_xvlarmoe_libero_pre_post_processors,
    reconcile_xvlarmoe_processors,
)
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
)
from lerobot.lerobot_types import TransitionKey
from lerobot.utils.constants import IMAGENET_STATS, OBS_STATE


def test_xvla_loads_action_aligned_state_without_future_images():
    config = XVLARMoEConfig(chunk_size=3, n_action_steps=3)
    metadata = type(
        "Metadata",
        (),
        {
            "fps": 20,
            "features": {
                "observation.state": {},
                "observation.images.front": {},
                "action": {},
            },
        },
    )()

    timestamps = resolve_delta_timestamps(config, metadata)

    assert timestamps == {
        "observation.state": [0.0, 0.05, 0.1],
        "action": [0.0, 0.05, 0.1],
    }


def test_relative_libero_actions_use_action_aligned_states_for_absolute_targets():
    raw = torch.tensor(
        [[[0.2, -0.4, 0.6, 0.0, 0.0, 0.2, -1.0], [0.4, 0.2, -0.2, 0.0, 0.0, 0.4, 1.0]]]
    )
    state = torch.tensor(
        [[[0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 0.04, -0.04],
          [0.51, -0.01, 0.22, 0.0, 0.0, 0.05, 0.04, -0.04]]]
    )
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: state},
        TransitionKey.ACTION: raw,
    }

    ee6d = XVLARMoELiberoActionToEE6DProcessorStep()(transition)[TransitionKey.ACTION]
    restored = XVLARMoEEE6DToLiberoActionProcessorStep()(
        {TransitionKey.ACTION: ee6d}
    )[TransitionKey.ACTION]

    expected_xyz = state[..., :3] + raw[..., :3] * 0.05
    expected_axis_angle = torch.tensor([[[0.0, 0.0, 0.1], [0.0, 0.0, 0.25]]])
    assert ee6d.shape == (1, 2, 20)
    torch.testing.assert_close(restored[..., :3], expected_xyz)
    torch.testing.assert_close(restored[..., 3:6], expected_axis_angle, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(restored[..., 6], raw[..., 6])
    torch.testing.assert_close(ee6d[..., 9], torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(
        XVLARMoELiberoActionToEE6DProcessorStep()(transition)[TransitionKey.OBSERVATION][OBS_STATE],
        state[:, 0],
    )


def test_absolute_action_chunk_rejects_a_single_unaligned_state():
    raw = torch.zeros(1, 2, 7)
    state = torch.zeros(1, 8)
    transition = {TransitionKey.OBSERVATION: {OBS_STATE: state}, TransitionKey.ACTION: raw}

    with pytest.raises(ValueError, match="action-aligned"):
        XVLARMoELiberoActionToEE6DProcessorStep()(transition)


def test_dataset_and_inference_states_match_pretrained_20d_contract():
    dataset_state = torch.tensor([[0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.4, -0.4]])
    dataset_transition = {TransitionKey.OBSERVATION: {OBS_STATE: dataset_state}}
    training_state = XVLARMoEDatasetStateToPretrainedProcessorStep()(dataset_transition)[
        TransitionKey.OBSERVATION
    ][OBS_STATE]

    env_preprocessor, _ = make_xvlarmoe_libero_pre_post_processors()
    assert isinstance(env_preprocessor.steps[0], LiberoProcessorStep)
    observation = {
        "observation.robot_state": {
            "eef": {
                "pos": torch.tensor([[0.1, -0.2, 0.3]]),
                "mat": torch.eye(3).unsqueeze(0),
            },
            "gripper": {"qpos": torch.tensor([[0.4, -0.4]])},
        }
    }
    inference_state = env_preprocessor.steps[0].observation(observation)[OBS_STATE]

    assert training_state.shape == (1, 20)
    assert inference_state.shape == (1, 20)
    torch.testing.assert_close(training_state, inference_state)


def test_rmoe_postprocessor_matches_original_xvla():
    action = torch.zeros(2, 20)
    action[:, :3] = torch.tensor([[0.1, 0.2, 0.3], [-0.2, 0.4, 0.1]])
    action[:, 3] = 1
    action[:, 7] = 1
    action[:, 9] = torch.tensor([0.2, 0.8])
    transition = {TransitionKey.ACTION: action}

    original = XVLARotation6DToAxisAngleProcessorStep()(transition)[TransitionKey.ACTION]
    rmoe = XVLARMoEEE6DToLiberoActionProcessorStep()(transition)[TransitionKey.ACTION]
    torch.testing.assert_close(rmoe, original)


def test_train_uint8_and_inference_float_images_have_identical_model_values():
    step = XVLARMoEImageProcessorStep()
    uint8_image = torch.tensor([[[[0, 127], [255, 64]]] * 3], dtype=torch.uint8)
    float_image = uint8_image.float() / 255.0

    train = step({TransitionKey.OBSERVATION: {"observation.images.image": uint8_image}})[
        TransitionKey.OBSERVATION
    ]["observation.images.image"]
    inference = step({TransitionKey.OBSERVATION: {"observation.images.image": float_image}})[
        TransitionKey.OBSERVATION
    ]["observation.images.image"]
    repeated = step({TransitionKey.OBSERVATION: {"observation.images.image": inference}})[
        TransitionKey.OBSERVATION
    ]["observation.images.image"]

    torch.testing.assert_close(train, inference)
    torch.testing.assert_close(repeated, inference)
    mean = torch.tensor(IMAGENET_STATS["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STATS["std"]).view(1, 3, 1, 1)
    torch.testing.assert_close(train, (float_image - mean) / std)


def test_reconcile_builds_one_canonical_train_and_inference_policy_path():
    preprocessor = PolicyProcessorPipeline(
        steps=[
            RenameObservationsProcessorStep(rename_map={}),
            AddBatchDimensionProcessorStep(),
            XVLAAddDomainIdProcessorStep(domain_id=3),
            DeviceProcessorStep(device="cpu"),
        ]
    )
    postprocessor = PolicyProcessorPipeline(steps=[])

    class Config:
        action_mode = "ee6d"

    reconciled, _ = reconcile_xvlarmoe_processors(Config(), preprocessor, postprocessor)
    step_types = [type(step) for step in reconciled.steps]

    assert reconciled.steps[0].rename_map == DEFAULT_LIBERO_RENAME_MAP
    assert step_types.count(XVLARMoEImageProcessorStep) == 1
    assert step_types.count(XVLARMoELiberoActionToEE6DProcessorStep) == 1
    assert step_types.count(XVLARMoEDatasetStateToPretrainedProcessorStep) == 1
    domain_step = next(step for step in reconciled.steps if isinstance(step, XVLAAddDomainIdProcessorStep))
    assert domain_step.domain_id == 0
    assert step_types.index(XVLARMoEImageProcessorStep) < step_types.index(DeviceProcessorStep)
    assert step_types.index(XVLARMoELiberoActionToEE6DProcessorStep) < step_types.index(DeviceProcessorStep)
