# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from lerobot.configs.observation_history import resolve_observation_delta_indices
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.rtc import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import (
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.rollout.inference.observation_history import ObservationHistory  # noqa: E402
from lerobot.rollout.inference.rtc import RTCInferenceEngine  # noqa: E402

CAMERA = "observation.images.camera"
HW_FEATURES = {
    OBS_STATE: {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
    CAMERA: {"dtype": "image", "shape": (2, 2, 3), "names": ["height", "width", "channels"]},
}


class RecordingPreprocessor:
    def __init__(self, steps=()):
        self.steps = list(steps)
        self.pipeline = PolicyProcessorPipeline(steps=self.steps)
        self.seen = []

    def __call__(self, batch):
        self.seen.append(float(batch[OBS_STATE][0, 0]))
        return self.pipeline(batch)

    def reset(self):
        self.pipeline.reset()


class RecordingPolicy:
    def __init__(self, config):
        self.config = config
        self.batches = []
        self.kwargs = []
        self.engine = None
        self.on_predict = None

    def predict_action_chunk(self, batch, **kwargs):
        self.batches.append(batch)
        self.kwargs.append(kwargs)
        if self.on_predict:
            self.on_predict()
        self.engine._shutdown_event.set()
        return torch.zeros(1, 4, 1)

    def reset(self):
        pass


def make_engine(indices=(-1, 0), image_indices=None, state_indices=None, steps=(), camera=CAMERA):
    config = SimpleNamespace(
        observation_delta_indices=list(indices) if indices is not None else None,
        image_observation_delta_indices=image_indices,
        state_observation_delta_indices=state_indices,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(1,)),
            camera: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 2, 2)),
        },
    )
    policy = RecordingPolicy(config)
    preprocessor = RecordingPreprocessor(steps)
    engine = RTCInferenceEngine(
        policy,
        preprocessor,
        lambda x: x,
        SimpleNamespace(robot_type="test", action_features={"joint.pos": float}),
        RTCConfig(enabled=True),
        HW_FEATURES,
        "old task",
        10,
        "cpu",
    )

    class Postprocessor:
        def __call__(self, actions):
            return actions

        def reset(self):
            pass

    engine._postprocessor = Postprocessor()
    policy.engine = engine
    engine._action_queue = ActionQueue(engine._rtc_config)
    engine._policy_active.set()
    return engine, policy, preprocessor


def notify(engine, value, image=None, policy_tick=None):
    observation = {
        "joint.pos": value,
        "camera": np.full((2, 2, 3), value, dtype=np.uint8) if image is None else image,
    }
    if policy_tick is None:
        engine.notify_observation(observation)
    else:
        engine.notify_observation(observation, policy_tick=policy_tick)


def infer(engine, policy):
    engine._shutdown_event.clear()
    engine._action_queue.clear()
    engine._rtc_loop()
    assert not engine.failed, engine.failure_traceback
    return policy.batches[-1]


def test_rtc_history_tracks_notifications_and_not_inference_calls():
    engine, policy, processor = make_engine()
    for value in (1, 2, 3):
        notify(engine, value)
    batch = infer(engine, policy)
    assert batch[OBS_STATE].shape == (1, 2, 1)
    assert batch[OBS_STATE][0, :, 0].tolist() == [2, 3]
    assert processor.seen == [2, 3]
    repeated = infer(engine, policy)
    torch.testing.assert_close(repeated[OBS_STATE], batch[OBS_STATE])
    assert repeated[f"{OBS_STATE}_is_pad"].tolist() == [[False, False]]


def test_rtc_modality_stride_padding_latest_task_and_relative_anchor():
    relative = RelativeActionsProcessorStep(enabled=True)
    engine, policy, processor = make_engine(None, [-4, -2, 0], [-1, 0], steps=[relative])
    for value in (1, 2, 3):
        notify(engine, value)
    engine.set_task("current task")
    batch = infer(engine, policy)
    assert batch[CAMERA].shape == (1, 3, 3, 2, 2)
    torch.testing.assert_close(batch[CAMERA][0, :, 0, 0, 0] * 255, torch.tensor([1.0, 1.0, 3.0]))
    assert batch[f"{CAMERA}_is_pad"].tolist() == [[True, False, False]]
    assert batch[f"{OBS_STATE}_is_pad"].tolist() == [[False, False]]
    assert batch[OBS_STATE][0, :, 0].tolist() == [2, 3]
    assert batch["task"] == ["current task"]
    assert relative.get_cached_state().tolist() == [[3]]
    assert processor.seen == [1, 2, 3]


def test_same_policy_tick_replaces_the_latest_history_frame_instead_of_appending():
    engine, policy, processor = make_engine()
    notify(engine, 1, policy_tick=0)
    notify(engine, 2, policy_tick=0)
    notify(engine, 3, policy_tick=0)
    notify(engine, 4, policy_tick=1)
    notify(engine, 5, policy_tick=1)
    batch = infer(engine, policy)
    assert batch[OBS_STATE][0, :, 0].tolist() == [3, 5]
    assert batch[f"{OBS_STATE}_is_pad"].tolist() == [[False, False]]
    assert processor.seen == [3, 5]


def test_history_snapshots_observations_without_holding_the_observation_lock(monkeypatch):
    import lerobot.rollout.inference.observation_history as history_module

    engine, _, _ = make_engine()
    original = history_module._cpu_snapshot

    def checked(value):
        assert engine._obs_lock.acquire(blocking=False), "snapshot must not run under the observation lock"
        engine._obs_lock.release()
        return original(value)

    monkeypatch.setattr(history_module, "_cpu_snapshot", checked)
    notify(engine, 1)
    assert engine._observation_history.snapshot()[-1].observation["joint.pos"] == 1


def test_engine_passes_the_true_leftover_length_alongside_the_padded_prefix():
    engine, policy, _ = make_engine()
    notify(engine, 1)
    notify(engine, 2)
    infer(engine, policy)
    leftover = engine._action_queue.get_left_over().shape[0]
    assert 0 < leftover < engine._rtc_config.execution_horizon
    engine._shutdown_event.clear()
    engine._rtc_loop()
    assert not engine.failed, engine.failure_traceback
    kwargs = policy.kwargs[-1]
    assert kwargs["prev_chunk_left_over"].shape == (engine._rtc_config.execution_horizon, 1)
    assert kwargs["execution_horizon"] == leftover


def test_engine_drops_an_empty_leftover_instead_of_padding_it_with_zeros():
    engine, policy, _ = make_engine()
    notify(engine, 1)
    notify(engine, 2)
    infer(engine, policy)
    while engine._action_queue.get() is not None:
        pass
    assert engine._action_queue.get_left_over().numel() == 0
    engine._shutdown_event.clear()
    engine._rtc_loop()
    assert not engine.failed, engine.failure_traceback
    assert policy.kwargs[-1]["prev_chunk_left_over"] is None


def test_engine_omits_execution_horizon_for_policies_that_do_not_accept_it():
    engine, policy, _ = make_engine()

    class NarrowPolicy(RecordingPolicy):
        def predict_action_chunk(self, batch, inference_delay=None, prev_chunk_left_over=None):
            return super().predict_action_chunk(
                batch, inference_delay=inference_delay, prev_chunk_left_over=prev_chunk_left_over
            )

    narrow = NarrowPolicy(policy.config)
    narrow.engine = engine
    engine._policy = narrow
    engine._passes_execution_horizon = engine._accepts_execution_horizon(narrow)
    notify(engine, 1)
    notify(engine, 2)
    infer(engine, narrow)
    engine._shutdown_event.clear()
    engine._rtc_loop()
    assert not engine.failed, engine.failure_traceback
    assert "execution_horizon" not in narrow.kwargs[-1]


def test_rtc_copies_mutable_camera_buffers_and_pads_first_frame_once():
    engine, policy, processor = make_engine()
    image = np.full((2, 2, 3), 7, dtype=np.uint8)
    notify(engine, 1, image)
    image.fill(99)
    batch = infer(engine, policy)
    torch.testing.assert_close(batch[CAMERA], torch.full((1, 2, 3, 2, 2), 7 / 255))
    assert batch[f"{CAMERA}_is_pad"].tolist() == [[True, False]]
    assert processor.seen == [1]
    engine.reset()
    notify(engine, 8)
    reset_batch = infer(engine, policy)
    assert reset_batch[OBS_STATE][0, :, 0].tolist() == [8, 8]
    assert reset_batch[f"{OBS_STATE}_is_pad"].tolist() == [[True, False]]


def test_rtc_stacks_policy_keys_after_rename():
    renamed = "observation.images.front"
    engine, policy, _ = make_engine(
        steps=[RenameObservationsProcessorStep({CAMERA: renamed})], camera=renamed
    )
    notify(engine, 1)
    notify(engine, 2)
    batch = infer(engine, policy)
    assert batch[renamed].shape == (1, 2, 3, 2, 2)
    assert batch[f"{renamed}_is_pad"].tolist() == [[False, False]]
    assert CAMERA not in batch


@pytest.mark.parametrize("indices", [None, [0]])
def test_single_frame_keeps_original_shapes_and_processor_cost(indices):
    engine, policy, processor = make_engine(indices)
    notify(engine, 1)
    notify(engine, 2)
    batch = infer(engine, policy)
    assert batch[OBS_STATE].shape == (1, 1)
    assert batch[CAMERA].shape == (1, 3, 2, 2)
    assert processor.seen == [2]
    assert f"{OBS_STATE}_is_pad" not in batch


def test_reset_during_inference_discards_old_history_chunk():
    engine, policy, _ = make_engine()
    notify(engine, 1)
    notify(engine, 2)
    policy.on_predict = engine.reset
    infer(engine, policy)
    assert engine.action_queue.qsize() == 0
    policy.on_predict = None
    notify(engine, 9)
    batch = infer(engine, policy)
    assert batch[OBS_STATE][0, :, 0].tolist() == [9, 9]


def test_history_retains_bounded_cpu_snapshots_with_monotonic_identity():
    engine, policy, _ = make_engine(None, [-4, -2, 0])
    tensor = torch.tensor([1.0], requires_grad=True)
    for value in range(8):
        observation = {
            "joint.pos": value,
            "camera": np.full((2, 2, 3), value, dtype=np.uint8),
            "extra": tensor,
        }
        engine.notify_observation(observation)
    frames = engine._observation_history.snapshot()
    assert len(frames) == 5
    assert [frame.observation_id for frame in frames] == [3, 4, 5, 6, 7]
    assert all(a.timestamp <= b.timestamp for a, b in zip(frames, frames[1:], strict=False))
    assert all(frame.observation["extra"].device.type == "cpu" for frame in frames)
    assert not frames[0].observation["extra"].requires_grad
    tensor.detach().fill_(9)
    assert frames[0].observation["extra"].item() == 1
    batch = infer(engine, policy)
    torch.testing.assert_close(batch[CAMERA][0, :, 0, 0, 0] * 255, torch.tensor([3.0, 5.0, 7.0]))
    assert batch[f"{CAMERA}_is_pad"].tolist() == [[False, False, False]]
    engine.reset()
    notify(engine, 10)
    assert engine._observation_history.snapshot()[0].observation_id == 8


def test_notifications_during_preprocessing_do_not_change_worker_snapshot():
    engine, policy, processor = make_engine()
    notify(engine, 1)
    notify(engine, 2)

    def publish_next_frame(step_index, transition):
        assert engine._obs_lock.acquire(blocking=False), "preprocessing must not hold the observation lock"
        engine._obs_lock.release()
        notify(engine, 9)
        return transition

    processor.pipeline.steps = [RelativeActionsProcessorStep()]
    processor.pipeline.before_step_hooks = [publish_next_frame]
    batch = infer(engine, policy)
    assert batch[OBS_STATE][0, :, 0].tolist() == [1, 2]
    assert processor.seen == [1, 2]
    processor.pipeline.before_step_hooks = []
    next_batch = infer(engine, policy)
    assert next_batch[OBS_STATE][0, :, 0].tolist() == [9, 9]


def test_start_clears_preexisting_history():
    engine, policy, _ = make_engine()
    notify(engine, 1)
    engine.pause()
    engine.start()
    engine.stop()
    notify(engine, 4)
    engine._policy_active.set()
    batch = infer(engine, policy)
    assert batch[OBS_STATE][0, :, 0].tolist() == [4, 4]
    assert batch[f"{OBS_STATE}_is_pad"].tolist() == [[True, False]]


def test_query_refreshes_observation_history_and_reset_epoch_before_chunk():
    engine, policy, _ = make_engine()
    policy.supports_text_generation = lambda: True

    def generate_text(batch):
        engine.reset()
        notify(engine, 7)
        notify(engine, 8)
        engine.set_task("new task")
        return "new task"

    policy.generate_text = generate_text
    notify(engine, 1)
    engine.ask("what next?")
    batch = infer(engine, policy)
    assert batch[OBS_STATE][0, :, 0].tolist() == [7, 8]
    assert batch["task"] == ["new task"]
    assert engine.action_queue.qsize() > 0


@pytest.mark.parametrize("visual,proprio", [(True, True), (True, False), (False, True)])
def test_rtc_temporal_batch_reaches_pi05_mem_without_advancing_policy_queue(visual, proprio):
    pi05_config = pytest.importorskip("lerobot.policies.pi05.configuration_pi05")
    pi05_model = pytest.importorskip("lerobot.policies.pi05.modeling_pi05")
    PI05Config, PI05Policy = pi05_config.PI05Config, pi05_model.PI05Policy  # noqa: N806

    engine, _, processor = make_engine()
    config = PI05Config(
        device="cpu",
        use_visual_memory=visual,
        use_proprioceptive_memory=proprio,
        memory_frames=3,
        memory_stride=2,
        image_resolution=(2, 2),
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(1,)),
            CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 2, 2)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(1,))},
    )

    class RecordingMEMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.inputs = []

        def sample_actions(self, images, img_masks, tokens, masks, *, states, state_masks, **kwargs):
            self.inputs.append((images, img_masks, tokens, states, state_masks))
            engine._shutdown_event.set()
            return torch.zeros(1, 4, 1)

    policy = PI05Policy.__new__(PI05Policy)
    nn.Module.__init__(policy)
    policy.config = config
    policy.model = RecordingMEMModel()
    policy.reset()
    engine._policy = policy
    engine._observation_history = ObservationHistory(config)

    class Tokenizer:
        steps = processor.steps

        def __call__(self, batch):
            batch = processor(batch)
            batch[OBS_LANGUAGE_TOKENS] = batch[OBS_STATE].to(torch.long)
            batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(1, 1, dtype=torch.bool)
            return batch

        def reset(self):
            processor.reset()

    engine._preprocessor = Tokenizer()
    for value in (1, 2, 3):
        notify(engine, value)
    for _ in range(2):
        engine._shutdown_event.clear()
        engine._action_queue.clear()
        engine._rtc_loop()
        assert not engine.failed, engine.failure_traceback
        images, image_masks, tokens, states, state_masks = policy.model.inputs[-1]
        if visual:
            assert images[0].shape == (1, 3, 3, 2, 2)
            torch.testing.assert_close(
                (images[0][0, :, 0, 0, 0] + 1) * 255 / 2, torch.tensor([1.0, 1.0, 3.0]), atol=1e-5, rtol=1e-5
            )
            assert image_masks[0].tolist() == [[False, True, True]]
        else:
            assert images[0].shape == (1, 3, 2, 2)
        if proprio:
            assert states[0, :, 0].tolist() == [1, 1, 3]
            assert state_masks.tolist() == [[False, True, True]]
        else:
            assert states is None
        assert tokens.tolist() == [[3]]
        assert policy._memory_steps_seen == 0
        assert all(not queue for queue in policy._memory_queues.values())


def test_history_without_current_offset_still_anchors_actions_on_latest_state():
    relative = RelativeActionsProcessorStep(enabled=True)
    engine, policy, processor = make_engine(None, [-2], [-2], steps=[relative])
    for value in (1, 2, 3):
        notify(engine, value)
    batch = infer(engine, policy)
    assert batch[OBS_STATE].tolist() == [[[1]]]
    assert relative.get_cached_state().tolist() == [[3]]
    assert processor.seen == [1, 3]


@pytest.mark.parametrize("indices", [[], [1], [-1.5, 0]])
def test_history_rejects_unavailable_or_invalid_offsets(indices):
    with pytest.raises(ValueError, match="past/current integers"):
        make_engine(indices)


def test_delta_index_resolution_does_not_swallow_errors_from_the_shared_property():
    class BrokenConfig:
        image_observation_delta_indices = None

        @property
        def observation_delta_indices(self):
            raise AttributeError("renamed field")

    with pytest.raises(AttributeError, match="renamed field"):
        resolve_observation_delta_indices(BrokenConfig(), OBS_STATE)
