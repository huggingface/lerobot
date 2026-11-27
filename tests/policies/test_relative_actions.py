"""Tests for relative action transforms — full pipeline validation.

Tests the complete flow matching OpenPI:
  raw actions → RelativeActions → Normalize(relative_stats) → model → Unnormalize → AbsoluteActions

Uses real dataset: lerobot-data-collection/dagger_final_1_21
"""

import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.compute_stats import get_feature_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import (
    TransitionKey,
    batch_to_transition,
    create_transition,
)
from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.processor.relative_action_processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
    to_absolute_actions,
    to_relative_actions,
)
from lerobot.utils.constants import ACTION, ACTION_ANCHOR, OBS_STATE

pytestmark = pytest.mark.dataset

CHUNK_SIZE = 10
REPO_ID = "lerobot-data-collection/dagger_final_1_21"


@pytest.fixture(scope="module")
def dataset():
    return LeRobotDataset(REPO_ID, episodes=[0])


@pytest.fixture(scope="module")
def action_dim(dataset):
    return dataset.meta.features["action"]["shape"][0]


def _build_action_chunks(dataset, chunk_size, max_chunks=50):
    """Build action chunks from hf_dataset, like the training script does."""
    hf = dataset.hf_dataset
    total = len(hf)
    all_ep = torch.tensor([int(hf[i]["episode_index"]) for i in range(total)])
    chunks, states = [], []
    for i in range(total - chunk_size + 1):
        if all_ep[i] != all_ep[i + chunk_size - 1]:
            continue
        chunk_actions = torch.stack([hf[i + k]["action"] for k in range(chunk_size)]).float()
        state = hf[i]["observation.state"].float()
        chunks.append(chunk_actions)
        states.append(state)
        if len(chunks) >= max_chunks:
            break
    assert len(chunks) > 0, f"No valid chunks found. total={total}, ep_indices={all_ep.tolist()}"
    return torch.stack(chunks), torch.stack(states)


def _compute_relative_chunk_stats(action_chunks, states, mask):
    all_chunks = []
    for actions, state in zip(action_chunks, states, strict=True):
        relative = to_relative_actions(actions.unsqueeze(0), state.unsqueeze(0), mask).squeeze(0)
        all_chunks.append(relative.numpy())
    all_relative = np.concatenate(all_chunks, axis=0)
    return get_feature_stats(all_relative, axis=0, keepdims=all_relative.ndim == 1)


# Basic roundtrip tests


def test_roundtrip_3d(action_dim):
    actions = torch.randn(4, CHUNK_SIZE, action_dim)
    state = torch.randn(4, action_dim)
    mask = [True] * action_dim
    recovered = to_absolute_actions(to_relative_actions(actions, state, mask), state, mask)
    torch.testing.assert_close(recovered, actions)


def test_roundtrip_2d(action_dim):
    actions = torch.randn(4, action_dim)
    state = torch.randn(4, action_dim)
    mask = [True] * action_dim
    recovered = to_absolute_actions(to_relative_actions(actions, state, mask), state, mask)
    torch.testing.assert_close(recovered, actions)


def test_stacked_state_anchors_on_the_current_frame(action_dim):
    """A (B, T_obs, state_dim) state collapses to frame 0, not to some mix of the stack."""
    actions = torch.randn(4, CHUNK_SIZE, action_dim)
    stacked = torch.randn(4, 3, action_dim)
    mask = [True] * action_dim

    relative = to_relative_actions(actions, stacked, mask)
    torch.testing.assert_close(relative, to_relative_actions(actions, stacked[:, 0], mask))
    torch.testing.assert_close(to_absolute_actions(relative, stacked, mask), actions)


def test_no_mutation(action_dim):
    actions = torch.randn(2, CHUNK_SIZE, action_dim)
    original = actions.clone()
    state = torch.randn(2, action_dim)
    to_relative_actions(actions, state, [True] * action_dim)
    torch.testing.assert_close(actions, original)


def test_exclude_joints_supports_partial_name_matching():
    names = [
        "right_joint_1.pos",
        "right_gripper.pos",
        "left_joint_1.pos",
        "left_gripper.pos",
    ]
    step = RelativeActionsProcessorStep(enabled=True, exclude_joints=["gripper"], action_names=names)
    assert step._build_mask(len(names)) == [True, False, True, False]


# Chunk-level relative stats test


def test_chunk_stats_have_larger_std_than_frame_stats(dataset, action_dim):
    """Chunk-level relative stats should have larger std than per-frame relative stats."""
    action_chunks, states = _build_action_chunks(dataset, CHUNK_SIZE)
    mask = [True] * action_dim

    chunk_stats = _compute_relative_chunk_stats(action_chunks, states, mask)

    # Per-frame stats
    hf = dataset.hf_dataset
    n = min(500, len(hf))
    frame_actions = torch.stack([hf[i]["action"] for i in range(n)]).float()
    frame_states = torch.stack([hf[i]["observation.state"] for i in range(n)]).float()
    frame_relatives = to_relative_actions(frame_actions, frame_states, mask).numpy()
    frame_stats = get_feature_stats(frame_relatives, axis=0, keepdims=frame_relatives.ndim == 1)

    assert chunk_stats["std"].mean() >= frame_stats["std"].mean(), (
        f"Chunk std ({chunk_stats['std'].mean():.4f}) should be >= "
        f"frame std ({frame_stats['std'].mean():.4f})"
    )


# Full pipeline roundtrip: relative → normalize → unnormalize → absolute


def test_full_pipeline_roundtrip(dataset, action_dim):
    """Test the complete OpenPI pipeline: relative → normalize → unnormalize → absolute."""
    action_chunks, states = _build_action_chunks(dataset, CHUNK_SIZE)
    mask = [True] * action_dim

    relative_stats = _compute_relative_chunk_stats(action_chunks, states, mask)
    stats = {ACTION: dict(relative_stats.items())}

    features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
    norm_map = {FeatureType.ACTION: NormalizationMode.MEAN_STD}

    relative_step = RelativeActionsProcessorStep(enabled=True)
    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
    unnormalizer = UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
    absolute_step = AbsoluteActionsProcessorStep(enabled=True)

    original_actions = action_chunks[0].unsqueeze(0)
    state = states[0].unsqueeze(0)

    batch = {ACTION: original_actions, OBS_STATE: state}
    transition = batch_to_transition(batch)

    # Forward: relative → normalize
    t1 = relative_step(transition)
    t2 = normalizer(t1)

    normalized_action = t2[TransitionKey.ACTION]
    assert normalized_action.abs().mean() < 10, (
        f"Normalized actions should be in reasonable range, got mean abs {normalized_action.abs().mean():.2f}"
    )

    # Reverse: unnormalize → absolute
    t3 = unnormalizer(t2)
    t4 = absolute_step(t3)

    recovered_actions = t4[TransitionKey.ACTION]
    torch.testing.assert_close(recovered_actions, original_actions, atol=1e-4, rtol=1e-4)


def test_normalized_relative_values_are_reasonable(dataset, action_dim):
    """With correct chunk stats, normalized relative actions should be in a reasonable range."""
    action_chunks, states = _build_action_chunks(dataset, CHUNK_SIZE)
    mask = [True] * action_dim

    relative_stats = _compute_relative_chunk_stats(action_chunks, states, mask)
    mean = torch.tensor(relative_stats["mean"]).float()
    std = torch.tensor(relative_stats["std"]).float()

    all_normalized = []
    for actions, state in zip(action_chunks, states, strict=True):
        relative = to_relative_actions(actions.unsqueeze(0), state.unsqueeze(0), mask).squeeze(0)
        normalized = (relative - mean) / (std + 1e-6)
        all_normalized.append(normalized)

    all_normalized = torch.cat(all_normalized, dim=0)

    pct_in_range = (all_normalized.abs() < 5).float().mean()
    assert pct_in_range > 0.9, (
        f"Only {pct_in_range * 100:.1f}% of normalized values in [-5, 5], expected >90%"
    )

    assert all_normalized.mean().abs() < 1.0, (
        f"Mean of normalized relative actions is {all_normalized.mean():.2f}, expected near 0"
    )


def test_processor_step_roundtrip(dataset, action_dim):
    """RelativeActionsProcessorStep applies relative offsets; to_absolute_actions recovers original."""
    hf = dataset.hf_dataset
    batch = {
        ACTION: torch.stack([hf[i]["action"] for i in range(4)]),
        OBS_STATE: torch.stack([hf[i]["observation.state"] for i in range(4)]),
    }
    original_actions = batch[ACTION].clone()
    transition = batch_to_transition(batch)

    step = RelativeActionsProcessorStep(enabled=True)
    relative_transition = step(transition)
    assert not torch.allclose(relative_transition[TransitionKey.ACTION], original_actions)

    state = transition[TransitionKey.OBSERVATION][OBS_STATE]
    mask = [True] * action_dim
    recovered = to_absolute_actions(relative_transition[TransitionKey.ACTION], state, mask)
    torch.testing.assert_close(recovered, original_actions)


def test_processor_step_disabled_is_noop(dataset, action_dim):
    """enabled=False should be a no-op."""
    hf = dataset.hf_dataset
    batch = {
        ACTION: torch.stack([hf[i]["action"] for i in range(2)]),
        OBS_STATE: torch.stack([hf[i]["observation.state"] for i in range(2)]),
    }
    original = batch[ACTION].clone()
    transition = batch_to_transition(batch)
    result = RelativeActionsProcessorStep(enabled=False)(transition)
    torch.testing.assert_close(result[TransitionKey.ACTION], original)


# Training batch shape validation


def test_relative_with_action_chunks(dataset, action_dim):
    """Verify relative actions work correctly with (B, chunk_size, action_dim) shaped actions."""
    action_chunks, states = _build_action_chunks(dataset, CHUNK_SIZE)

    # Simulate a training batch: actions=(B, chunk_size, action_dim), state=(B, state_dim)
    batch_actions = action_chunks[:4]  # (4, chunk_size, action_dim)
    batch_states = states[:4]  # (4, state_dim)

    mask = [True] * action_dim
    relative = to_relative_actions(batch_actions, batch_states, mask)

    # First action in each chunk should be close to zero (action[t] - state[t] ≈ small)
    first_relatives = relative[:, 0, :]  # (B, action_dim)
    assert first_relatives.abs().mean() < relative.abs().mean(), (
        f"First action in chunk should have smaller relative offset than average. "
        f"First: {first_relatives.abs().mean():.4f}, Average: {relative.abs().mean():.4f}"
    )

    # Later actions should have larger relative offsets
    last_relatives = relative[:, -1, :]  # (B, action_dim)
    assert last_relatives.abs().mean() >= first_relatives.abs().mean(), (
        f"Last action in chunk should have >= relative offset than first. "
        f"Last: {last_relatives.abs().mean():.4f}, First: {first_relatives.abs().mean():.4f}"
    )

    # Roundtrip
    recovered = to_absolute_actions(relative, batch_states, mask)
    torch.testing.assert_close(recovered, batch_actions)


def test_relative_stats_match_actual_data_distribution(dataset, action_dim):
    """Verify computed stats match the actual relative-action distribution."""
    action_chunks, states = _build_action_chunks(dataset, CHUNK_SIZE)
    mask = [True] * action_dim

    # Compute stats like the training script does
    relative_stats = _compute_relative_chunk_stats(action_chunks, states, mask)

    # Also compute directly
    all_relatives = []
    for actions, state in zip(action_chunks, states, strict=True):
        rel = to_relative_actions(actions.unsqueeze(0), state.unsqueeze(0), mask).squeeze(0)
        all_relatives.append(rel)
    all_relatives_tensor = torch.cat(all_relatives, dim=0)

    # Compare mean
    actual_mean = all_relatives_tensor.mean(dim=0).numpy()
    np.testing.assert_allclose(relative_stats["mean"], actual_mean, atol=0.01)

    # Compare std
    actual_std = all_relatives_tensor.std(dim=0).numpy()
    np.testing.assert_allclose(relative_stats["std"], actual_std, atol=0.1)

    # Verify q01 < mean < q99
    assert (relative_stats["q01"] < relative_stats["mean"]).all(), "q01 should be < mean"
    assert (relative_stats["mean"] < relative_stats["q99"]).all(), "mean should be < q99"


def test_quantile_normalization_roundtrip(dataset, action_dim):
    """Full roundtrip with QUANTILES normalization (what OpenPI uses for pi05)."""
    action_chunks, states = _build_action_chunks(dataset, CHUNK_SIZE)
    mask = [True] * action_dim

    relative_stats = _compute_relative_chunk_stats(action_chunks, states, mask)
    stats = {ACTION: dict(relative_stats.items())}

    features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
    norm_map = {FeatureType.ACTION: NormalizationMode.QUANTILES}

    relative_step = RelativeActionsProcessorStep(enabled=True)
    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
    unnormalizer = UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
    absolute_step = AbsoluteActionsProcessorStep(enabled=True)

    original_actions = action_chunks[0].unsqueeze(0)
    state = states[0].unsqueeze(0)

    batch = {ACTION: original_actions, OBS_STATE: state}
    transition = batch_to_transition(batch)

    # Forward: relative → quantile normalize
    t1 = relative_step(transition)
    t2 = normalizer(t1)

    normalized = t2[TransitionKey.ACTION]
    # Most values should be in [-1, 1] with quantile normalization
    pct_in_range = (normalized.abs() < 2).float().mean()
    assert pct_in_range > 0.5, f"Only {pct_in_range * 100:.1f}% in [-2, 2] after quantile norm, expected >50%"

    # Reverse: unnormalize → absolute
    t3 = unnormalizer(t2)
    t4 = absolute_step(t3)

    recovered = t4[TransitionKey.ACTION]
    torch.testing.assert_close(recovered, original_actions, atol=1e-3, rtol=1e-3)


def test_state_not_modified_by_relative_processor(dataset, action_dim):
    """State should never be modified by the relative-action processor."""
    hf = dataset.hf_dataset
    batch = {
        ACTION: torch.stack([hf[i]["action"] for i in range(4)]),
        OBS_STATE: torch.stack([hf[i]["observation.state"] for i in range(4)]),
    }
    original_state = batch[OBS_STATE].clone()
    transition = batch_to_transition(batch)

    step = RelativeActionsProcessorStep(enabled=True)
    result = step(transition)

    result_state = result[TransitionKey.OBSERVATION][OBS_STATE]
    torch.testing.assert_close(result_state, original_state)


def test_anchor_not_in_config():
    """The anchor travels with the batch and must not leak into the config."""
    step = RelativeActionsProcessorStep(enabled=True)
    result = step(create_transition(observation={OBS_STATE: torch.tensor([[1.0, 2.0, 3.0, 4.0]])}))
    assert result[TransitionKey.COMPLEMENTARY_DATA][ACTION_ANCHOR] is not None
    assert set(step.get_config()) == {"enabled", "exclude_joints", "action_names"}


# --------------------------------------------------------------------------------------
# Chunk anchoring
# --------------------------------------------------------------------------------------
# A chunk generated at tick t must be un-relativised against state(t) for every action it
# yields. The preprocessor writes the anchor into the batch it hands the policy, and the
# postprocessor reads it back from that batch, so the anchor is always the state the chunk
# was predicted from.

_ANCHOR_NAMES = [f"j{i}.pos" for i in range(4)]


def _anchor_step(**kwargs):
    return RelativeActionsProcessorStep(enabled=True, action_names=list(_ANCHOR_NAMES), **kwargs)


def _feed(step, value):
    """Run one preprocess call at a uniform state, and report the anchor it wrote."""
    result = step(create_transition(observation={OBS_STATE: torch.full((1, 4), value)}))
    return result[TransitionKey.COMPLEMENTARY_DATA][ACTION_ANCHOR]


def test_each_preprocess_call_re_anchors():
    step = _anchor_step()
    _feed(step, 1.0)
    torch.testing.assert_close(_feed(step, 5.0), torch.full((1, 4), 5.0))


def test_a_whole_chunk_is_anchored_on_the_state_it_was_predicted_from():
    relative_step = _anchor_step()
    absolute_step = AbsoluteActionsProcessorStep(enabled=True)
    # A distinct offset per step, so a wrong anchor cannot be masked by a flat chunk.
    chunk = torch.stack([torch.full((4,), 0.1 * (i + 1)) for i in range(3)])[None]  # [1, 3, 4]
    anchor = torch.full((1, 4), 10.0)
    preprocessed = relative_step(create_transition(observation={OBS_STATE: anchor}))
    complementary = preprocessed[TransitionKey.COMPLEMENTARY_DATA]
    actions = absolute_step(create_transition(action=chunk, complementary_data=complementary))[
        TransitionKey.ACTION
    ]
    torch.testing.assert_close(actions, anchor[:, None] + chunk)
