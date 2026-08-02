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

import tempfile
import unittest
from pathlib import Path

import torch

from lerobot.policies.rl_token import RLTokenConfig, RLTokenModel, RLTokenStage1Trainer


class RLTokenModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = RLTokenConfig(
            vla_dim=12,
            token_dim=8,
            max_tokens=6,
            encoder_layers=1,
            decoder_layers=1,
            num_heads=2,
        )

    def test_reconstruction_uses_mask_and_stops_vla_gradient(self) -> None:
        model = RLTokenModel(self.config)
        embeddings = torch.randn(3, 6, 12, requires_grad=True)
        mask = torch.tensor(
            [
                [True, True, True, True, True, True],
                [True, True, True, False, False, False],
                [True, True, True, True, False, False],
            ]
        )

        loss, token = model.reconstruction_loss(embeddings, mask)
        loss.backward()

        self.assertEqual(token.shape, (3, 8))
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNone(embeddings.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_stage1_step_and_pretrained_round_trip(self) -> None:
        model = RLTokenModel(self.config)
        trainer = RLTokenStage1Trainer(model, lr=1e-3)
        embeddings = torch.randn(2, 5, 12)
        mask = torch.ones(2, 5, dtype=torch.bool)

        metrics = trainer.step(embeddings, mask)
        self.assertEqual(trainer.steps, 1)
        self.assertIn("reconstruction_loss", metrics)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            restored = RLTokenModel.from_pretrained(tmpdir)
            self.assertEqual(restored.config, self.config)
            for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
                torch.testing.assert_close(actual, expected)
            self.assertTrue((Path(tmpdir) / "rl_token_config.json").is_file())

            trainer_state_path = Path(tmpdir) / "trainer.pt"
            torch.save(trainer.state_dict(), trainer_state_path)
            restored_trainer = RLTokenStage1Trainer(restored, lr=1e-3)
            restored_trainer.load_state_dict(torch.load(trainer_state_path, weights_only=True))
            self.assertEqual(restored_trainer.steps, trainer.steps)

    def test_rejects_sequence_over_capacity(self) -> None:
        model = RLTokenModel(self.config)
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            model.encode(torch.randn(1, 7, 12))


if __name__ == "__main__":
    unittest.main()
