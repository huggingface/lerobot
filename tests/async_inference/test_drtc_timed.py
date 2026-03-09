import pickle

import numpy as np

from lerobot.async_inference.drtc_timed import DrtcAction, DrtcObservation
from lerobot.utils.constants import OBS_STATE


def test_drtc_action_getters_and_pickle_roundtrip():
    action = np.arange(6, dtype=np.float32)
    timed_action = DrtcAction(timestamp=123.4, control_step=7, action_step=11, action=action)

    assert timed_action.get_timestamp() == 123.4
    assert timed_action.get_control_step() == 7
    assert timed_action.get_action_step() == 11
    np.testing.assert_array_equal(timed_action.get_action(), action)

    reloaded = pickle.loads(pickle.dumps(timed_action))  # nosec B301
    assert reloaded.get_control_step() == 7
    assert reloaded.get_action_step() == 11
    np.testing.assert_array_equal(reloaded.get_action(), action)


def test_drtc_observation_getters_and_pickle_roundtrip():
    observation = {OBS_STATE: [1.0, 2.0, 3.0]}
    timed_observation = DrtcObservation(
        timestamp=456.7,
        control_step=13,
        observation=observation,
        chunk_start_step=17,
        server_received_ts=460.0,
    )

    assert timed_observation.get_timestamp() == 456.7
    assert timed_observation.get_control_step() == 13
    assert timed_observation.get_observation() == observation
    assert timed_observation.chunk_start_step == 17
    assert timed_observation.server_received_ts == 460.0

    reloaded = pickle.loads(pickle.dumps(timed_observation))  # nosec B301
    assert reloaded.get_control_step() == 13
    assert reloaded.get_observation() == observation
    assert reloaded.chunk_start_step == 17
    assert reloaded.server_received_ts == 460.0
