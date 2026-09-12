#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import torch

from lerobot.rl.algorithms.rlt import ChunkTransitionAssembler, ExecutedChunk, RLTReplayBuffer


def _chunk(
    start: float,
    *,
    executed_steps: int = 4,
    terminated: bool = False,
    truncated: bool = False,
    intervention_mask: torch.Tensor | None = None,
) -> ExecutedChunk:
    offsets = len(range(0, executed_steps, 2))
    actions = torch.arange(start, start + 4).reshape(4, 1)
    references = torch.zeros(offsets, 4, 1)
    if intervention_mask is None:
        intervention_mask = torch.zeros(4, dtype=torch.bool)
    return ExecutedChunk(
        states=torch.stack([torch.tensor([start + index, 1.0]) for index in range(offsets)]),
        references=references,
        actions=actions,
        rewards=torch.ones(4),
        intervention_mask=intervention_mask,
        executed_steps=executed_steps,
        final_state=torch.tensor([start + executed_steps, 2.0]),
        final_reference=torch.full((4, 1), -1.0),
        terminated=terminated,
        truncated=truncated,
    )


class ChunkTransitionAssemblerTest(unittest.TestCase):
    def test_overlap_preserves_explicit_next_state_and_step_interventions(self) -> None:
        assembler = ChunkTransitionAssembler(chunk_length=4, action_dim=1, discount=0.9, stride=2)
        first = _chunk(0.0, intervention_mask=torch.tensor([False, True, False, False]))
        second = _chunk(10.0, executed_steps=1, terminated=True)

        self.assertEqual(assembler.add_chunk(first), [])
        transitions = assembler.add_chunk(second)

        self.assertEqual(len(transitions), 3)
        full, overlap_tail, final = transitions
        self.assertEqual(int(full.valid_horizon), 4)
        torch.testing.assert_close(full.next_state, second.states[0])
        self.assertFalse(bool(full.terminated))
        self.assertEqual(full.reference[:, 0].tolist(), [0.0, 1.0, 0.0, 0.0])
        self.assertEqual(full.intervention_mask.tolist(), [False, True, False, False])

        self.assertEqual(int(overlap_tail.valid_horizon), 3)
        self.assertTrue(bool(overlap_tail.terminated))
        self.assertEqual(float(overlap_tail.bootstrap_discount), 0.0)
        self.assertEqual(overlap_tail.action[:, 0].tolist(), [2.0, 3.0, 10.0, 0.0])
        self.assertEqual(int(final.valid_horizon), 1)

    def test_timeout_bootstraps_with_real_final_context_and_gamma_n(self) -> None:
        assembler = ChunkTransitionAssembler(chunk_length=4, action_dim=1, discount=0.5, stride=2)
        chunk = _chunk(3.0, executed_steps=3, truncated=True)
        transitions = assembler.add_chunk(chunk)

        self.assertEqual(len(transitions), 2)
        self.assertFalse(bool(transitions[0].terminated))
        self.assertTrue(bool(transitions[0].truncated))
        self.assertAlmostEqual(float(transitions[0].bootstrap_discount), 0.5**3)
        self.assertAlmostEqual(float(transitions[1].bootstrap_discount), 0.5)
        torch.testing.assert_close(transitions[0].next_state, chunk.final_state)

    def test_replay_stores_next_state_instead_of_deriving_it_from_adjacency(self) -> None:
        assembler = ChunkTransitionAssembler(chunk_length=4, action_dim=1, discount=0.99, stride=2)
        transitions = assembler.add_chunk(_chunk(4.0, executed_steps=3, truncated=True))
        replay = RLTReplayBuffer(capacity=8, state_dim=2, chunk_length=4, action_dim=1)
        replay.add_many(list(reversed(transitions)))

        torch.testing.assert_close(replay.next_state[0], transitions[1].next_state)
        torch.testing.assert_close(replay.next_state[1], transitions[0].next_state)
        self.assertEqual(replay.valid_horizon[:2].tolist(), [1, 3])


if __name__ == "__main__":
    unittest.main()
