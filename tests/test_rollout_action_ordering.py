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

"""Joint-ordering tests for the rollout action path.

``dataset_features["observation.state"]["names"]`` must equal ``ordered_action_keys``:
``send_next_action`` labels the action tensor positionally, and for relative actions
``AbsoluteActionsProcessorStep`` adds the cached state positionally too.

These drive the real engines and ``send_next_action`` against a bimanual stub whose
hardware order is the reverse of the checkpoint's, and are parameterized over the wiring
variants so the broken ones are reproduced rather than merely absent.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: E402
from lerobot.processor import (  # noqa: E402
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
)
from lerobot.utils.feature_utils import (  # noqa: E402
    build_dataset_frame,
    combine_feature_dicts,
    hw_to_dataset_features,
)

# ---------------------------------------------------------------------------
# The two orders under test
# ---------------------------------------------------------------------------

_JOINTS = [f"joint_{i}" for i in range(1, 8)] + ["gripper"]

# Hardware enumeration: `bi_openarm_follower._motors_ft` is `{**left_*, **right_*}`.
HW_ORDER = [f"left_{j}.pos" for j in _JOINTS] + [f"right_{j}.pos" for j in _JOINTS]
# Checkpoint enumeration: `policy.config.action_feature_names` from a right-arm-first dataset.
CKPT_ORDER = [f"right_{j}.pos" for j in _JOINTS] + [f"left_{j}.pos" for j in _JOINTS]

_CAMERAS = {"base": (224, 224, 3), "left_wrist": (224, 224, 3), "right_wrist": (224, 224, 3)}

# Measurements and offsets live on separate decimal scales so that a commanded value
# ``anchor + delta`` decomposes back into exactly one (joint, output dimension) pair. With
# overlapping scales the decomposition is ambiguous and the tests silently prove nothing.
MEASUREMENTS = {name: 1000.0 * (idx + 1) for idx, name in enumerate(HW_ORDER)}

GRIPPERS = frozenset(name for name in HW_ORDER if "gripper" in name)


def _observation(offset: float = 0.0) -> dict:
    """Raw robot observation: a measurement per joint plus a frame per camera."""
    import numpy as np

    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    return {
        **{name: value + offset for name, value in MEASUREMENTS.items()},
        **dict.fromkeys(_CAMERAS, frame),
    }


def _delta(dim: int) -> float:
    """Relative offset for output dimension *dim*, kept below the measurement spacing."""
    return float(dim + 1)


def _expected_command(joint: str) -> float:
    """What a correctly wired pipeline commands for *joint* (grippers carry no anchor)."""
    dim = CKPT_ORDER.index(joint)
    anchor = 0.0 if joint in GRIPPERS else MEASUREMENTS[joint]
    return _delta(dim) + anchor


def test_orders_are_a_pure_block_swap():
    """Same set, identical within-arm order: the permutation can only swap arm twins."""
    assert set(HW_ORDER) == set(CKPT_ORDER)
    assert HW_ORDER != CKPT_ORDER
    assert [HW_ORDER.index(name) for name in CKPT_ORDER] == list(range(8, 16)) + list(range(0, 8))
    assert [i for i, n in enumerate(HW_ORDER) if "gripper" in n] == [7, 15]
    assert [i for i, n in enumerate(CKPT_ORDER) if "gripper" in n] == [7, 15]


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubRelativePolicy:
    """Relative-action chunk policy emitting a recognisable per-dimension offset."""

    def __init__(self, action_dim: int = 16):
        self.action_dim = action_dim
        self.config = SimpleNamespace(
            action_feature_names=list(CKPT_ORDER),
            use_amp=False,
            chunk_size=30,
            n_action_steps=30,
        )
        self.observed_states: list[list[float]] = []

    def predict_action_chunk(self, batch, **kwargs):
        return self.select_action(batch).unsqueeze(1)

    def select_action(self, batch, **kwargs):
        state = batch["observation.state"]
        self.observed_states.append(state.squeeze(0).tolist())
        return torch.tensor([[_delta(i) for i in range(self.action_dim)]], dtype=torch.float32)

    def reset(self):
        pass

    # Borrow the real queue accounting rather than restate it: the sync engine keys the
    # relative-action anchor hold off ``queued_action_count``, so a stub that answered it
    # by hand could drift from ``PreTrainedPolicy`` and hide a wiring break. This policy
    # keeps no queue attribute at all, so it reports 0 (fresh prediction every tick).
    _action_queue_attrs = PreTrainedPolicy._action_queue_attrs
    drop_queued_actions = PreTrainedPolicy.drop_queued_actions
    queued_action_count = PreTrainedPolicy.queued_action_count

    def supports_text_generation(self):
        return False


class _AnchorPipeline:
    """Stand-in for ``PolicyProcessorPipeline`` wrapping one real relative/absolute step."""

    def __init__(self, step, *, is_preprocessor: bool):
        self.steps = (step,)
        self._step = step
        self._is_preprocessor = is_preprocessor

    def __call__(self, data):
        from lerobot.lerobot_types import TransitionKey

        if self._is_preprocessor:
            batch = dict(data)
            state = batch["observation.state"]
            if not isinstance(state, torch.Tensor):
                state = torch.as_tensor(state, dtype=torch.float32)
            if state.ndim == 1:
                state = state.unsqueeze(0)
            batch["observation.state"] = state
            self._step({TransitionKey.OBSERVATION: batch})
            return batch

        transition = self._step({TransitionKey.ACTION: data})
        return transition[TransitionKey.ACTION]

    def reset(self):
        pass


def _make_pipelines():
    """A wired relative/absolute pair, sharing the anchor by reference as in production."""
    relative = RelativeActionsProcessorStep(
        enabled=True, exclude_joints=["gripper"], action_names=list(CKPT_ORDER)
    )
    absolute = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative)
    return (
        _AnchorPipeline(relative, is_preprocessor=True),
        _AnchorPipeline(absolute, is_preprocessor=False),
    )


# ---------------------------------------------------------------------------
# Feature construction, mirroring build_rollout_context
# ---------------------------------------------------------------------------


def _build_features(*, align_state: bool, align_action: bool):
    """Reproduce ``build_rollout_context``'s feature reconciliation for one variant."""
    from lerobot.rollout.context import (
        _align_action_feature_order,
        _align_state_feature_order,
        _resolve_action_key_order,
    )

    observation_features_hw = {**dict.fromkeys(HW_ORDER, float), **_CAMERAS}
    action_features_hw = dict.fromkeys(HW_ORDER, float)

    if align_state:
        observation_features_hw = _align_state_feature_order(observation_features_hw, list(CKPT_ORDER))
    if align_action:
        action_features_hw = _align_action_feature_order(action_features_hw, list(CKPT_ORDER))

    dataset_features = combine_feature_dicts(
        hw_to_dataset_features(action_features_hw, "action"),
        hw_to_dataset_features(observation_features_hw, "observation"),
    )
    ordered_action_keys = _resolve_action_key_order(list(CKPT_ORDER), list(action_features_hw))
    return dataset_features, ordered_action_keys


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

# (id, align_state, align_action, remap_in_get_action, anchor_ok, policy_gets_trained_order)
#
# `remap_in_get_action` models the pre-#4416 name round-trip the engines used to perform
# (`make_robot_action` on dataset_features[ACTION] + reindex to ordered_action_keys).
_VARIANTS = [
    ("pre_rebase", False, False, True, True, False),
    ("post_rebase_unfixed", True, False, True, False, True),
    ("drop_remap", True, False, False, True, True),
    ("head", True, True, False, True, True),
    ("revert_4416_only", False, True, False, False, False),
    ("revert_everything", False, False, True, True, False),
]

_VARIANT_IDS = [v[0] for v in _VARIANTS]


def _remap_like_pre_4416(action_tensor, dataset_features, ordered_action_keys):
    """The name round-trip both engines performed before ``0755b419`` removed it."""
    from lerobot.policies.utils import make_robot_action

    action_dict = make_robot_action(action_tensor, dataset_features)
    return torch.tensor([action_dict[k] for k in ordered_action_keys])


# ---------------------------------------------------------------------------
# Sync engine
# ---------------------------------------------------------------------------


def _run_sync_tick(*, align_state, align_action, remap):
    """One control tick through the real sync engine and ``send_next_action``."""
    from lerobot.rollout.inference.sync import SyncInferenceEngine
    from lerobot.rollout.strategies.core import send_next_action
    from lerobot.utils.action_interpolator import ActionInterpolator

    dataset_features, ordered_action_keys = _build_features(
        align_state=align_state, align_action=align_action
    )
    preprocessor, postprocessor = _make_pipelines()
    policy = _StubRelativePolicy()

    engine = SyncInferenceEngine(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset_features=dataset_features,
        ordered_action_keys=ordered_action_keys,
        task="fold the t-shirt",
        device="cpu",
        robot_type="bi_openarm_follower",
    )

    if remap:
        # Restore the historical round-trip without touching production code.
        inner = engine.get_action

        def get_action(obs_frame):
            action = inner(obs_frame)
            if action is None:
                return None
            return _remap_like_pre_4416(action, dataset_features, ordered_action_keys)

        engine.get_action = get_action

    obs_raw = _observation()
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(
            cfg=SimpleNamespace(max_action_jump_deg=None),
            shutdown_event=SimpleNamespace(set=lambda: None),
        ),
        hardware=SimpleNamespace(robot_wrapper=SimpleNamespace(send_action=lambda action: action)),
        processors=SimpleNamespace(robot_action_processor=lambda pair: pair[0]),
        policy=SimpleNamespace(inference=engine),
        data=SimpleNamespace(dataset_features=dataset_features, ordered_action_keys=ordered_action_keys),
    )

    action_dict = send_next_action(
        obs_processed=dict(obs_raw),
        obs_raw=obs_raw,
        ctx=ctx,
        interpolator=ActionInterpolator(multiplier=1),
    )
    assert action_dict is not None
    return action_dict, dataset_features, ordered_action_keys, policy


def _provenance(action_dict):
    """Decompose each commanded value into ``(anchor joint | None, policy output dim)``.

    Measurements are multiples of 1000 and offsets are 1..16, so the split is exact.
    """
    out = {}
    for joint, value in action_dict.items():
        thousands, remainder = divmod(round(value), 1000)
        anchor = HW_ORDER[thousands - 1] if thousands else None
        dim = remainder - 1
        assert 0 <= dim < len(CKPT_ORDER), (
            f"commanded value {value} for {joint} does not decompose into anchor + offset; "
            "the test's arithmetic model no longer matches the pipeline"
        )
        out[joint] = (anchor, dim)
    return out


def _misanchored(action_dict):
    """Joints whose commanded value was built from another joint's measurement."""
    return {
        joint: anchor
        for joint, (anchor, _) in _provenance(action_dict).items()
        if anchor is not None and anchor != joint
    }


def _twin(joint: str) -> str:
    """The same-numbered joint on the opposite arm."""
    return joint.replace("left_", "right_") if joint.startswith("left_") else joint.replace("right_", "left_")


@pytest.mark.parametrize(
    ("align_state", "align_action", "remap", "anchor_ok", "trained_order"),
    [v[1:] for v in _VARIANTS],
    ids=_VARIANT_IDS,
)
def test_sync_anchor_provenance_per_variant(align_state, align_action, remap, anchor_ok, trained_order):
    """Every joint must be anchored on its own measurement; ``anchor_ok=False`` is not."""
    action_dict, dataset_features, ordered_action_keys, policy = _run_sync_tick(
        align_state=align_state, align_action=align_action, remap=remap
    )

    mismatched = _misanchored(action_dict)
    if anchor_ok:
        assert not mismatched, f"anchor taken from the wrong joint: {mismatched}"
    else:
        # Not merely "some mismatch": every anchored joint must be anchored on its
        # opposite-arm twin, the only permutation these two orders can produce. The two
        # grippers are excluded from the relative mask, so they carry no anchor to break.
        assert set(mismatched) == set(HW_ORDER) - GRIPPERS
        for joint, src in mismatched.items():
            assert src == _twin(joint), f"{joint} anchored on {src}, expected its twin {_twin(joint)}"

    # Independently: did the policy receive the state layout it was trained on?
    assert len(policy.observed_states) == 1
    state_names = dataset_features["observation.state"]["names"]
    assert (state_names == CKPT_ORDER) is trained_order
    assert [MEASUREMENTS[n] for n in state_names] == policy.observed_states[0]


def test_sync_head_wiring_is_fully_correct():
    """The shipped configuration, asserted as exact values rather than by decomposition."""
    action_dict, dataset_features, ordered_action_keys, policy = _run_sync_tick(
        align_state=True, align_action=True, remap=False
    )

    assert dataset_features["observation.state"]["names"] == CKPT_ORDER
    assert dataset_features["action"]["names"] == CKPT_ORDER
    assert ordered_action_keys == CKPT_ORDER
    assert sorted(action_dict) == sorted(HW_ORDER)

    for joint in CKPT_ORDER:
        assert action_dict[joint] == pytest.approx(_expected_command(joint)), joint


def test_sync_grippers_swap_with_each_other_but_never_run_away():
    """Grippers carry no anchor so cannot diverge, but still swap with each other."""
    action_dict, _, _, _ = _run_sync_tick(align_state=True, align_action=False, remap=True)
    provenance = _provenance(action_dict)

    assert _misanchored(action_dict), "expected this variant to misanchor the arm joints"
    for joint in GRIPPERS:
        anchor, dim = provenance[joint]
        assert anchor is None, f"{joint} should carry no anchor: it is in relative_exclude_joints"
        assert dim == CKPT_ORDER.index(_twin(joint))
        assert action_dict[joint] == pytest.approx(_expected_command(_twin(joint)))


def test_sync_anchor_is_pinned_across_a_chunk():
    """A cached action must keep the anchor from the tick that predicted its chunk."""
    from lerobot.rollout.inference.sync import SyncInferenceEngine

    dataset_features, ordered_action_keys = _build_features(align_state=True, align_action=True)
    preprocessor, postprocessor = _make_pipelines()

    class _ChunkingPolicy(_StubRelativePolicy):
        """Serves a 3-step chunk out of ``_action_queue``, the deque name the base class reads."""

        chunk_len = 3

        def __init__(self):
            super().__init__()
            self._action_queue: deque[torch.Tensor] = deque()
            self.predictions = 0

        def predict_action_chunk(self, batch, **kwargs):
            self.predictions += 1
            self.observed_states.append(batch["observation.state"].squeeze(0).tolist())
            row = torch.tensor([[_delta(i) for i in range(self.action_dim)]], dtype=torch.float32)
            return row.unsqueeze(1).expand(1, self.chunk_len, self.action_dim)

        def select_action(self, batch, **kwargs):
            if not self._action_queue:
                self._action_queue.extend(self.predict_action_chunk(batch).squeeze(0))
            return self._action_queue.popleft().unsqueeze(0)

    policy = _ChunkingPolicy()
    engine = SyncInferenceEngine(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset_features=dataset_features,
        ordered_action_keys=ordered_action_keys,
        task="fold the t-shirt",
        device="cpu",
        robot_type="bi_openarm_follower",
    )

    # Tick 1 predicts the chunk and anchors on the pose it saw.
    first = engine.get_action(build_dataset_frame(dataset_features, _observation(), "observation"))

    # Ticks 2-3 serve cached actions while the robot has moved on. The anchor must stay
    # pinned to tick 1's pose, so the commanded values must not change.
    moved = _observation(offset=5.0)
    for _ in range(2):
        later = engine.get_action(build_dataset_frame(dataset_features, moved, "observation"))
        torch.testing.assert_close(later, first)

    assert policy.predictions == 1, "the chunk should have been predicted exactly once"

    # Tick 4 drains the chunk, so a fresh one is predicted and re-anchors on the current
    # pose: every anchored joint shifts by +5, and the two unanchored grippers do not.
    fresh = engine.get_action(build_dataset_frame(dataset_features, moved, "observation"))
    assert policy.predictions == 2
    shift = torch.tensor([0.0 if joint in GRIPPERS else 5.0 for joint in CKPT_ORDER], dtype=torch.float32)
    torch.testing.assert_close(fresh, first + shift)


# ---------------------------------------------------------------------------
# RTC engine
# ---------------------------------------------------------------------------


def _seed_rtc_engine(dataset_features, state_names):
    """An RTC engine whose queue holds the chunk this variant's postprocessor would emit."""
    from lerobot.policies.rtc import ActionQueue
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.rollout.inference import RTCInferenceEngine

    preprocessor, postprocessor = _make_pipelines()
    engine = RTCInferenceEngine(
        policy=_StubRelativePolicy(),
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=SimpleNamespace(robot_type="bi_openarm_follower", action_features={}),
        rtc_config=RTCConfig(enabled=True, execution_horizon=8, max_guidance_weight=1.0),
        dataset_features=dataset_features,
        task="fold the t-shirt",
        fps=30.0,
        device="cpu",
    )
    # `_rtc_loop` merges post-processed chunks, so the queue holds absolute actions in the
    # policy's dimension order, anchored on this variant's own state layout.
    absolute = torch.tensor(
        [
            [
                _delta(dim) + (0.0 if CKPT_ORDER[dim] in GRIPPERS else MEASUREMENTS[state_names[dim]])
                for dim in range(len(CKPT_ORDER))
            ]
        ],
        dtype=torch.float32,
    )
    engine._action_queue = ActionQueue(RTCConfig(enabled=True, execution_horizon=8))
    engine._action_queue.merge(absolute.clone(), absolute.clone(), 0, None, task="fold the t-shirt")
    return engine


@pytest.mark.parametrize(
    ("align_state", "align_action", "remap", "anchor_ok"),
    [(v[1], v[2], v[3], v[4]) for v in _VARIANTS],
    ids=_VARIANT_IDS,
)
def test_rtc_anchor_provenance_per_variant(align_state, align_action, remap, anchor_ok):
    """RTC must label its queued actions the same way sync does, variant for variant."""
    dataset_features, ordered_action_keys = _build_features(
        align_state=align_state, align_action=align_action
    )
    engine = _seed_rtc_engine(dataset_features, dataset_features["observation.state"]["names"])

    action = engine.get_action(None)
    assert action is not None
    if remap:
        action = _remap_like_pre_4416(action, dataset_features, ordered_action_keys)

    action_dict = {k: action[i].item() for i, k in enumerate(ordered_action_keys)}
    mismatched = _misanchored(action_dict)
    assert (not mismatched) is anchor_ok, f"unexpected anchor provenance: {mismatched}"


def test_rtc_and_sync_agree_on_the_shipped_wiring():
    """Same checkpoint and robot must give the same commanded dict on either backend."""
    sync_action_dict, dataset_features, ordered_action_keys, _ = _run_sync_tick(
        align_state=True, align_action=True, remap=False
    )
    engine = _seed_rtc_engine(dataset_features, dataset_features["observation.state"]["names"])

    action = engine.get_action(None)
    rtc_action_dict = {k: action[i].item() for i, k in enumerate(ordered_action_keys)}

    assert rtc_action_dict.keys() == sync_action_dict.keys()
    for joint in sync_action_dict:
        assert rtc_action_dict[joint] == pytest.approx(sync_action_dict[joint]), joint
