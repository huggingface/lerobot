from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load_binder():
    spec = importlib.util.spec_from_file_location("task1_caligned_v2_binder", HERE / "bind_and_prepare.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PreparationContractTest(unittest.TestCase):
    def test_new_trajectory_and_training_identity(self) -> None:
        contract = json.loads((HERE / "preparation_contract.json").read_text())
        self.assertEqual(contract["status"], "implementation_ready_training_not_started")
        self.assertEqual(contract["sim24"]["episodes"], 24)
        self.assertEqual(contract["sim24"]["frames"], 3023)
        self.assertFalse(contract["training"]["old_sim_stream_equality_required"])
        self.assertEqual(
            contract["training"]["old_c_real_stream_sha256_int64le"],
            "f392d7b148905d90467a2565229df92d33e7805e8037a48eeca02c6d31730c53",
        )
        self.assertEqual(contract["training"]["sample_slots"], {"real24": 800000, "sim24": 800000})
        self.assertFalse(contract["hardware_authorized"])

    def test_eval_scope_is_only_simseen6_paired12(self) -> None:
        evaluation = json.loads((HERE / "preparation_contract.json").read_text())["evaluation"]
        self.assertEqual(evaluation["poses"], 6)
        self.assertEqual(evaluation["planned_rollouts"], 12)
        self.assertEqual(evaluation["first_model_balance"], {"C_source": 3, "C_aligned_v2": 3})
        self.assertFalse(evaluation["full_eval24_fallback"])
        self.assertFalse(evaluation["hardware_authorized"])

    def test_old_c_template_has_frozen_act_recipe(self) -> None:
        config = json.loads(
            (REPO / "experiments/task1_picklift_act_additive_three_model_200k_v1/configs/r24_localsim24gap_full.json").read_text()
        )
        self.assertEqual(config["steps"], 200000)
        self.assertEqual(config["seed"], 1000)
        self.assertEqual(config["batch_size"], 8)
        self.assertEqual(config["dataset"]["matched_two_stream_episode_groups"], {
            "real24": list(range(24)), "source_b": list(range(24, 48))
        })
        self.assertTrue(config["dataset"]["use_imagenet_stats"])
        self.assertEqual(config["policy"]["chunk_size"], 67)
        self.assertEqual(config["policy"]["n_action_steps"], 67)
        self.assertIsNone(config["policy"]["pretrained_path"])
        self.assertFalse(config["resume"])

    def test_source_eval24_stride_covers_exactly_24_pose_groups(self) -> None:
        plan = json.loads(
            (REPO / "experiments/task1_picklift_act_additive_three_model_200k_v1/evaluation_plan.json").read_text()
        )
        self.assertEqual(len(plan["trials"]), 72)
        groups = [plan["trials"][offset : offset + 3] for offset in range(0, 72, 3)]
        self.assertEqual(len(groups), 24)
        self.assertTrue(all(len({row["eval_pose_id"] for row in group}) == 1 for group in groups))
        self.assertEqual(len({group[0]["eval_pose_id"] for group in groups}), 24)

    def test_config_diff_helper_rejects_hidden_changes(self) -> None:
        binder = load_binder()
        left = {"dataset": {"root": "a", "repo_id": "a"}, "steps": 200000}
        right = {"dataset": {"root": "b", "repo_id": "b"}, "steps": 199999}
        self.assertEqual(binder.changed_paths(left, right), {"dataset.root", "dataset.repo_id", "steps"})

    def test_binder_requires_frozen_handoff(self) -> None:
        process = subprocess.run(
            [sys.executable, str(HERE / "bind_and_prepare.py")],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("required", process.stderr)

    def test_final_metric_accepts_actual_ot_train_log_prefix(self) -> None:
        spec = importlib.util.spec_from_file_location("task1_caligned_v2_freezer", HERE / "validate_and_freeze.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        line = "INFO 2026-08-10 ot_train.py:677 step:200000 loss:1.0 l1_loss:0.9 kld_loss:0.1 grdn:2.0 smp/s:3.0"
        parsed = module.final_metric_from_text(line)
        self.assertEqual(parsed["step"], "200000")
        self.assertEqual(parsed["smp/s"], "3.0")
        rounded = line.replace("step:200000", "step:200K")
        parsed_rounded = module.final_metric_from_text(rounded)
        self.assertEqual(parsed_rounded["step"], "200000")
        self.assertEqual(parsed_rounded["step_logged"], "200K")


if __name__ == "__main__":
    unittest.main()
