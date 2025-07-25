#!/usr/bin/env python

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

import socket
import threading
import time

import pytest
import torch
from torch.multiprocessing import Event, Queue

from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.utils.transition import Transition
from tests.utils import require_package


def create_test_transitions(count: int = 3) -> list[Transition]:
    """Create test transitions for integration testing."""
    transitions = []
    for i in range(count):
        transition = Transition(
            state={"observation": torch.randn(3, 64, 64), "state": torch.randn(10)},
            action=torch.randn(5),
            reward=torch.tensor(1.0 + i),
            done=torch.tensor(i == count - 1),  # Last transition is done
            truncated=torch.tensor(False),
            next_state={"observation": torch.randn(3, 64, 64), "state": torch.randn(10)},
            complementary_info={"step": torch.tensor(i), "episode_id": i // 2},
        )
        transitions.append(transition)
    return transitions


def create_test_interactions(count: int = 3) -> list[dict]:
    """Create test interactions for integration testing."""
    interactions = []
    for i in range(count):
        interaction = {
            "episode_reward": 10.0 + i * 5,
            "step": i * 100,
            "policy_fps": 30.0 + i,
            "intervention_rate": 0.1 * i,
            "episode_length": 200 + i * 50,
        }
        interactions.append(interaction)
    return interactions


def find_free_port():
    """Finds a free port on the local machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to port 0 to let the OS choose a free port
        s.listen(1)
        port = s.getsockname()[1]
        return port


@pytest.fixture
def cfg():
    cfg = TrainRLServerPipelineConfig()

    port = find_free_port()

    policy_cfg = SACConfig()
    policy_cfg.actor_learner_config.learner_host = "127.0.0.1"
    policy_cfg.actor_learner_config.learner_port = port
    policy_cfg.concurrency.actor = "threads"
    policy_cfg.concurrency.learner = "threads"
    policy_cfg.actor_learner_config.queue_get_timeout = 0.1

    cfg.policy = policy_cfg

    return cfg


@require_package("grpc")
@pytest.mark.timeout(10)  # force cross-platform watchdog
def test_end_to_end_transitions_flow(cfg):
    from lerobot.scripts.rl.actor import (
        establish_learner_connection,
        learner_service_client,
        push_transitions_to_transport_queue,
        send_transitions,
    )
    from lerobot.scripts.rl.learner import start_learner
    from lerobot.transport.utils import bytes_to_transitions
    from tests.transport.test_transport_utils import assert_transitions_equal

    """Test complete transitions flow from actor to learner."""
    transitions_actor_queue = Queue()
    transitions_learner_queue = Queue()

    interactions_queue = Queue()
    parameters_queue = Queue()
    shutdown_event = Event()

    learner_thread = threading.Thread(
        target=start_learner,
        args=(parameters_queue, transitions_learner_queue, interactions_queue, shutdown_event, cfg),
    )
    learner_thread.start()

    policy_cfg = cfg.policy
    learner_client, channel = learner_service_client(
        host=policy_cfg.actor_learner_config.learner_host, port=policy_cfg.actor_learner_config.learner_port
    )

    assert establish_learner_connection(learner_client, shutdown_event, attempts=5)

    send_transitions_thread = threading.Thread(
        target=send_transitions, args=(cfg, transitions_actor_queue, shutdown_event, learner_client, channel)
    )
    send_transitions_thread.start()

    input_transitions = create_test_transitions(count=5)

    push_transitions_to_transport_queue(input_transitions, transitions_actor_queue)

    # Wait for learner to start
    time.sleep(0.1)

    shutdown_event.set()

    # Wait for learner to receive transitions
    learner_thread.join()
    send_transitions_thread.join()
    channel.close()

    received_transitions = []
    while not transitions_learner_queue.empty():
        received_transitions.extend(bytes_to_transitions(transitions_learner_queue.get()))

    assert len(received_transitions) == len(input_transitions)
    for i, transition in enumerate(received_transitions):
        assert_transitions_equal(transition, input_transitions[i])


@require_package("grpc")
@pytest.mark.timeout(10)
def test_end_to_end_interactions_flow(cfg):
    from lerobot.scripts.rl.actor import (
        establish_learner_connection,
        learner_service_client,
        send_interactions,
    )
    from lerobot.scripts.rl.learner import start_learner
    from lerobot.transport.utils import bytes_to_python_object, python_object_to_bytes

    """Test complete interactions flow from actor to learner."""
    # Queues for actor-learner communication
    interactions_actor_queue = Queue()
    interactions_learner_queue = Queue()

    # Other queues required by the learner
    parameters_queue = Queue()
    transitions_learner_queue = Queue()

    shutdown_event = Event()

    # Start the learner in a separate thread
    learner_thread = threading.Thread(
        target=start_learner,
        args=(parameters_queue, transitions_learner_queue, interactions_learner_queue, shutdown_event, cfg),
    )
    learner_thread.start()

    # Establish connection from actor to learner
    policy_cfg = cfg.policy
    learner_client, channel = learner_service_client(
        host=policy_cfg.actor_learner_config.learner_host, port=policy_cfg.actor_learner_config.learner_port
    )

    assert establish_learner_connection(learner_client, shutdown_event, attempts=5)

    # Start the actor's interaction sending process in a separate thread
    send_interactions_thread = threading.Thread(
        target=send_interactions,
        args=(cfg, interactions_actor_queue, shutdown_event, learner_client, channel),
    )
    send_interactions_thread.start()

    # Create and push test interactions to the actor's queue
    input_interactions = create_test_interactions(count=5)
    for interaction in input_interactions:
        interactions_actor_queue.put(python_object_to_bytes(interaction))

    # Wait for the communication to happen
    time.sleep(0.1)

    # Signal shutdown and wait for threads to complete
    shutdown_event.set()
    learner_thread.join()
    send_interactions_thread.join()
    channel.close()

    # Verify that the learner received the interactions
    received_interactions = []
    while not interactions_learner_queue.empty():
        received_interactions.append(bytes_to_python_object(interactions_learner_queue.get()))

    assert len(received_interactions) == len(input_interactions)

    # Sort by a unique key to handle potential reordering in queues
    received_interactions.sort(key=lambda x: x["step"])
    input_interactions.sort(key=lambda x: x["step"])

    for received, expected in zip(received_interactions, input_interactions, strict=False):
        assert received == expected


@require_package("grpc")
@pytest.mark.parametrize("data_size", ["small", "large"])
@pytest.mark.timeout(10)
def test_end_to_end_parameters_flow(cfg, data_size):
    from lerobot.scripts.rl.actor import establish_learner_connection, learner_service_client, receive_policy
    from lerobot.scripts.rl.learner import start_learner
    from lerobot.transport.utils import bytes_to_state_dict, state_to_bytes

    """Test complete parameter flow from learner to actor, with small and large data."""
    # Actor's local queue to receive params
    parameters_actor_queue = Queue()
    # Learner's queue to send params from
    parameters_learner_queue = Queue()

    # Other queues required by the learner
    transitions_learner_queue = Queue()
    interactions_learner_queue = Queue()

    shutdown_event = Event()

    # Start the learner in a separate thread
    learner_thread = threading.Thread(
        target=start_learner,
        args=(
            parameters_learner_queue,
            transitions_learner_queue,
            interactions_learner_queue,
            shutdown_event,
            cfg,
        ),
    )
    learner_thread.start()

    # Establish connection from actor to learner
    policy_cfg = cfg.policy
    learner_client, channel = learner_service_client(
        host=policy_cfg.actor_learner_config.learner_host, port=policy_cfg.actor_learner_config.learner_port
    )

    assert establish_learner_connection(learner_client, shutdown_event, attempts=5)

    # Start the actor's parameter receiving process in a separate thread
    receive_params_thread = threading.Thread(
        target=receive_policy,
        args=(cfg, parameters_actor_queue, shutdown_event, learner_client, channel),
    )
    receive_params_thread.start()

    # Create test parameters based on parametrization
    if data_size == "small":
        input_params = {"layer.weight": torch.randn(128, 64)}
    else:  # "large"
        # CHUNK_SIZE is 2MB, so this tensor (4MB) will force chunking
        input_params = {"large_layer.weight": torch.randn(1024, 1024)}

    # Simulate learner having new parameters to send
    parameters_learner_queue.put(state_to_bytes(input_params))

    # Wait for the actor to receive the parameters
    time.sleep(0.1)

    # Signal shutdown and wait for threads to complete
    shutdown_event.set()
    learner_thread.join()
    receive_params_thread.join()
    channel.close()

    # Verify that the actor received the parameters correctly
    received_params = bytes_to_state_dict(parameters_actor_queue.get())

    assert received_params.keys() == input_params.keys()
    for key in input_params:
        assert torch.allclose(received_params[key], input_params[key])
