from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class PreparationSkeletonTest(unittest.TestCase):
    def test_rerender_identity_is_unbound(self) -> None:
        skeleton = json.loads((HERE / "preparation_skeleton.json").read_text())
        self.assertEqual(skeleton["status"], "waiting_for_final_rerender_identity")
        self.assertFalse(skeleton["final_contract_frozen"])
        self.assertFalse(skeleton["training_started"])
        self.assertFalse(skeleton["hardware_authorized"])
        self.assertTrue(all(value is None for value in skeleton["required_rerender_handoff"].values()))

    def test_binder_fails_without_final_handoff(self) -> None:
        process = subprocess.run(
            [sys.executable, str(HERE / "bind_rerender_and_prepare.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("required", process.stderr)

    def test_eval_plan_fails_without_trained_render_model(self) -> None:
        process = subprocess.run(
            [sys.executable, str(HERE / "build_paired_eval24_plan.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("required", process.stderr)


if __name__ == "__main__":
    unittest.main()
