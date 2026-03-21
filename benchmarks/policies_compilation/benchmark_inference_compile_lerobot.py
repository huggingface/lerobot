#!/usr/bin/env python
"""
Comprehensive torch.compile benchmark for LeRobot policies.

Compares main branch performance with torch.compile optimized version.

Usage:
    python benchmark_inference_compile_lerobot.py --policy act --device cuda
    python benchmark_inference_compile_lerobot.py --policy diffusion --device cpu
"""

import argparse
import copy
import json
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

# Safe imports for all processor functions
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors
from lerobot.policies.factory import make_policy, make_policy_config
from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.policies.tdmpc.processor_tdmpc import make_tdmpc_pre_post_processors
from lerobot.policies.vqbet.processor_vqbet import make_vqbet_pre_post_processors


class PolicyConfig:
    """Type-safe policy configuration"""

    def __init__(
        self,
        processor_func: Callable,
        config_kwargs: dict[str, Any],
        delta_timestamps_func: Callable[[Any, float], dict[str, list[float]]],
    ):
        self.processor_func = processor_func
        self.config_kwargs = config_kwargs
        self.delta_timestamps_func = delta_timestamps_func


# Safe policy configuration mapping
POLICY_CONFIGS: dict[str, PolicyConfig] = {
    "act": PolicyConfig(
        processor_func=make_act_pre_post_processors,
        config_kwargs={"use_vae": False, "chunk_size": 100, "n_action_steps": 10},
        delta_timestamps_func=lambda cfg, fps: {"action": [i / fps for i in range(cfg.chunk_size)]},
    ),
    "diffusion": PolicyConfig(
        processor_func=make_diffusion_pre_post_processors,
        config_kwargs={"n_obs_steps": 2, "horizon": 16, "n_action_steps": 8},
        delta_timestamps_func=lambda cfg, fps: {"action": [i / fps for i in range(cfg.n_action_steps)]},
    ),
    "pi0": PolicyConfig(
        processor_func=make_pi0_pre_post_processors,
        config_kwargs={"n_obs_steps": 1, "chunk_size": 50, "n_action_steps": 50},
        delta_timestamps_func=lambda cfg, fps: {"action": [i / fps for i in range(cfg.chunk_size)]},
    ),
    "tdmpc": PolicyConfig(
        processor_func=make_tdmpc_pre_post_processors,
        config_kwargs={"horizon": 5, "n_action_steps": 1},
        delta_timestamps_func=lambda cfg, fps: {"action": [0]},
    ),
    "vqbet": PolicyConfig(
        processor_func=make_vqbet_pre_post_processors,
        config_kwargs={"n_obs_steps": 1, "chunk_size": 8, "n_action_steps": 8},
        delta_timestamps_func=lambda cfg, fps: {"action": [i / fps for i in range(cfg.chunk_size)]},
    ),
    "smolvla": PolicyConfig(
        processor_func=make_smolvla_pre_post_processors,
        config_kwargs={"n_obs_steps": 1, "chunk_size": 50, "n_action_steps": 50},
        delta_timestamps_func=lambda cfg, fps: {"action": [i / fps for i in range(cfg.chunk_size)]},
    ),
}


class TorchCompileBenchmark:
    """Safe torch.compile benchmark for LeRobot policies"""

    def __init__(self, policy_name: str, device: str, repo_id: str = "AdilZtn/grab_red_cube_test_25"):
        self.policy_name = policy_name
        self.device = torch.device(device)
        self.repo_id = repo_id

        # Benchmark parameters
        self.n_inference = 100
        self.n_training = 50
        self.batch_size = 8
        self.warmup_steps = 10
        self.tolerance = 1e-5
        self.compile_mode = "default"
        self.fullgraph = False
        self.disable_dropout = False

        print(f"🤖 Torch.compile Benchmark for {policy_name.upper()} Policy")
        print(f"Device: {device}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Dataset: {repo_id}")
        print("=" * 60)

    def setup_policy_and_data(self) -> tuple[Any, dict, Any, Any]:
        """Setup policy, data, and processors"""
        torch.manual_seed(42)
        np.random.seed(42)

        # Load dataset metadata
        ds_meta = LeRobotDatasetMetadata(self.repo_id)

        # Get policy configuration safely
        if self.policy_name not in POLICY_CONFIGS:
            available_policies = list(POLICY_CONFIGS.keys())
            raise ValueError(f"Policy '{self.policy_name}' not supported. Available: {available_policies}")

        policy_config = POLICY_CONFIGS[self.policy_name]

        # Create policy configuration
        cfg = make_policy_config(self.policy_name, device=str(self.device), **policy_config.config_kwargs)

        # Disable dropout if dropout is present in the policy config and the flag is set
        if self.disable_dropout and hasattr(cfg, "dropout"):
            cfg.dropout = 0.0

        # Create policy
        policy = make_policy(cfg, ds_meta=ds_meta)
        policy.to(self.device)

        # Create processors using the safe function reference
        preprocessor, postprocessor = policy_config.processor_func(cfg, dataset_stats=ds_meta.stats)

        # Setup dataset with appropriate delta_timestamps
        delta_timestamps = policy_config.delta_timestamps_func(cfg, ds_meta.fps)
        dataset = LeRobotDataset(self.repo_id, episodes=[0], delta_timestamps=delta_timestamps)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

        # Get sample batch
        sample_batch = next(iter(dataloader))

        # Move to device
        for key in sample_batch:
            if isinstance(sample_batch[key], torch.Tensor):
                sample_batch[key] = sample_batch[key].to(self.device)

        # Preprocess
        sample_batch = preprocessor(sample_batch)

        return policy, sample_batch, preprocessor, postprocessor

    def benchmark_inference(self, policy, batch) -> tuple[float, torch.Tensor]:
        """Benchmark inference performance"""
        policy.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_steps):
                _ = policy.select_action(batch)
                if hasattr(policy, "reset"):
                    policy.reset()

        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(self.n_inference):
                action = policy.select_action(batch)
                if hasattr(policy, "reset"):
                    policy.reset()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / self.n_inference * 1000  # ms

        return avg_time, action

    def benchmark_training(self, policy, batch) -> tuple[float, list[float], list[float]]:
        """Benchmark training performance"""
        policy.train()
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

        # Warmup
        for _ in range(self.warmup_steps):
            optimizer.zero_grad()
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            if hasattr(policy, "reset"):
                policy.reset()

        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        losses = []
        grad_norms = []

        for _ in range(self.n_training):
            optimizer.zero_grad()
            loss, loss_dict = policy.forward(batch)
            loss.backward()

            # Calculate gradient norm safely
            grad_tensors = [p.grad.detach() for p in policy.parameters() if p.grad is not None]
            if grad_tensors:
                total_norm = torch.norm(torch.stack([torch.norm(g) for g in grad_tensors]))
            else:
                total_norm = torch.tensor(0.0)

            optimizer.step()

            losses.append(loss.item())
            grad_norms.append(total_norm.item())

            if hasattr(policy, "reset"):
                policy.reset()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / self.n_training * 1000  # ms

        return avg_time, losses, grad_norms

    def test_correctness(self, policy_original, policy_compiled, batch) -> dict[str, Any]:
        """Test numerical correctness between original and compiled versions"""

        # Test inference
        policy_original.eval()
        policy_compiled.eval()

        with torch.no_grad():
            # Reset both policies to same state
            if hasattr(policy_original, "reset"):
                policy_original.reset()

            action_original = policy_original.select_action(batch)

            if hasattr(policy_compiled, "reset"):
                policy_compiled.reset()

            action_compiled = policy_compiled.select_action(batch)

        action_diff = torch.abs(action_original - action_compiled).max().item()
        inference_correct = action_diff < self.tolerance

        # Test training forward pass
        policy_original.train()
        policy_compiled.train()

        # Use same random state
        torch.manual_seed(42)
        loss_original, _ = policy_original.forward(batch)

        torch.manual_seed(42)
        loss_compiled, _ = policy_compiled.forward(batch)

        loss_diff = torch.abs(loss_original - loss_compiled).item()
        training_correct = loss_diff < self.tolerance

        return {
            "inference_correct": inference_correct,
            "training_correct": training_correct,
            "action_diff": action_diff,
            "loss_diff": loss_diff,
        }

    def generate_report(self, results: dict[str, Any]) -> str:
        """Generate comprehensive markdown report including failures"""

        # Handle cases where benchmarks might have failed
        inference_time_orig = results.get("time_original_inference", float("inf"))
        inference_time_comp = results.get("time_compiled_inference", float("inf"))
        training_time_orig = results.get("time_original_training", float("inf"))
        training_time_comp = results.get("time_compiled_training", float("inf"))

        speedup_inference = results.get("speedup_inference", 0.0)
        speedup_training = results.get("speedup_training", 0.0)

        compilation_successful = results.get("compilation_successful", False)
        correctness_passed = results.get("correctness_passed", False)
        inference_benchmarked = results.get("inference_benchmarked", False)
        training_benchmarked = results.get("training_benchmarked", False)

        report = f"""# Torch.compile Benchmark Report: {self.policy_name.upper()}

## Environment
- **Policy**: {self.policy_name}
- **Device**: {self.device}
- **PyTorch**: {torch.__version__}
- **Dataset**: {self.repo_id}
- **Batch Size**: {self.batch_size}
- **Benchmark Parameters**: {self.n_inference} inference runs, {self.n_training} training runs

## 🔧 Compilation Results
- **Status**: {"✅ SUCCESS" if compilation_successful else "❌ FAILED"}"""

        if not compilation_successful:
            compilation_error = results.get("compilation_error", "Unknown error")
            report += f"""
- **Error**: `{compilation_error}`
- **Impact**: Policy will fall back to eager execution"""

        report += f"""

## 🎯 Correctness Results
- **Status**: {"✅ PASSED" if correctness_passed else "❌ FAILED"}
- **Inference**: {"PASSED" if results["correctness"]["inference_correct"] else "FAILED"}
- **Training**: {"PASSED" if results["correctness"]["training_correct"] else "FAILED"}

### Detailed Differences
- **Max Action Difference**: {results["correctness"]["action_diff"]:.2e} (threshold: {self.tolerance:.2e})
- **Loss Difference**: {results["correctness"]["loss_diff"]:.2e} (threshold: {self.tolerance:.2e})"""

        if not correctness_passed:
            action_diff = results["correctness"]["action_diff"]
            loss_diff = results["correctness"]["loss_diff"]

            report += f"""

### ⚠️ Correctness Analysis
- **Action diff magnitude**: {action_diff:.2e} ({"SEVERE" if action_diff > 1e-3 else "MODERATE" if action_diff > 1e-4 else "MINOR"})
- **Loss diff magnitude**: {loss_diff:.2e} ({"SEVERE" if loss_diff > 1e-3 else "MODERATE" if loss_diff > 1e-4 else "MINOR"})
- **Likely causes**: Graph breaks, dynamic shapes, numerical precision issues"""

        report += """

## ⚡ Performance Results

### Inference Performance"""

        if inference_benchmarked:
            report += f"""
- **Original**: {inference_time_orig:.2f} ms/iter
- **Compiled**: {inference_time_comp:.2f} ms/iter
- **🚀 Speedup**: {speedup_inference:.2f}x"""

            if speedup_inference < 1.0:
                report += " (⚠️ SLOWDOWN)"
            elif speedup_inference < 1.1:
                report += " (⚠️ INSUFFICIENT)"
        else:
            report += """
- **Status**: ❌ Benchmark failed
- **Reason**: Could not complete inference timing"""

        report += """

### Training Performance"""

        if training_benchmarked:
            report += f"""
- **Original**: {training_time_orig:.2f} ms/iter
- **Compiled**: {training_time_comp:.2f} ms/iter
- **🚀 Speedup**: {speedup_training:.2f}x"""

            if speedup_training < 1.0:
                report += " (⚠️ SLOWDOWN)"
            elif speedup_training < 1.1:
                report += " (⚠️ INSUFFICIENT)"
        else:
            report += """
- **Status**: ❌ Benchmark failed
- **Reason**: Could not complete training timing"""

        # Consistency metrics if available
        loss_consistency = results.get("loss_consistency", float("inf"))
        grad_norm_consistency = results.get("grad_norm_consistency", float("inf"))

        if loss_consistency != float("inf") and grad_norm_consistency != float("inf"):
            report += f"""

### Consistency Metrics
- **Average Loss Difference**: {loss_consistency:.2e}
- **Average Grad Norm Difference**: {grad_norm_consistency:.2e}"""

        report += f"""

## 📋 Success Criteria Analysis
- **✅ Compilation**: {"PASSED" if compilation_successful else "FAILED"}
- **✅ Correctness**: {"PASSED" if correctness_passed else "FAILED"}
- **✅ Performance**: {"PASSED" if speedup_inference > 1.1 and speedup_training > 1.1 else "FAILED"}
- **✅ Benchmarking**: {"PASSED" if inference_benchmarked and training_benchmarked else "FAILED"}

## 🎯 Overall Result
{"🎯 SUCCESS: Policy is torch.compile compatible!" if results["success"] else "❌ NEEDS WORK: torch.compile not yet functional"}

## 🛠️ Next Steps"""

        if not compilation_successful:
            report += """
1. **Fix compilation errors** - Enable torch._dynamo verbose mode for details
2. **Identify graph breaks** - Look for .item(), dynamic shapes, control flow
3. **Test incrementally** - Fix one issue at a time"""
        elif not correctness_passed:
            report += """
1. **Debug numerical differences** - Check for precision issues
2. **Verify tensor operations** - Ensure deterministic behavior
3. **Test with smaller tolerance** - May be acceptable for some use cases"""
        elif not (speedup_inference > 1.1 and speedup_training > 1.1):
            report += """
1. **Profile bottlenecks** - Use torch.profiler to identify slow ops
2. **Optimize compilation mode** - Try 'reduce-overhead' or 'max-autotune' modes
3. **Check graph breaks** - Even successful compilation can have breaks"""
        else:
            report += """
1. **Document changes** - Create clear PR with benchmark results
2. **Add tests** - Ensure torch.compile compatibility is maintained
3. **Consider edge cases** - Test with different batch sizes, VAE modes, etc."""

        report += f"""

## 🔍 Raw Data
```json
{json.dumps(results, indent=2)}
```
"""
        return report

    def run_benchmark(self) -> dict[str, Any]:
        """Run complete benchmark suite"""

        # Setup
        print("📦 Setting up policy and data...")
        try:
            policy, batch, preprocessor, postprocessor = self.setup_policy_and_data()
        except Exception as e:
            return {"success": False, "error": f"Setup failed: {str(e)}"}

        # Test compilation
        print("🔧 Testing torch.compile...")
        compilation_error = None
        policy_compiled = None

        try:
            # Enable verbose compilation debugging
            import torch._dynamo

            torch._dynamo.config.verbose = True
            torch._dynamo.config.suppress_errors = False

            policy_compiled = copy.deepcopy(policy)
            policy_compiled.forward = torch.compile(
                policy_compiled.forward, mode=self.compile_mode, fullgraph=self.fullgraph
            )
            policy_compiled.select_action = torch.compile(
                policy_compiled.select_action, mode="default", fullgraph=self.fullgraph
            )

            # Force compilation by running once
            policy_compiled.eval()
            with torch.no_grad():
                _ = policy_compiled.select_action(batch)

            print("✅ Compilation successful!")

        except Exception as e:
            compilation_error = str(e)
            print(f"❌ Compilation failed: {e}")
            # Continue with analysis even if compilation fails
            policy_compiled = policy  # Use original policy for comparison

        # Test correctness (always run this for analysis)
        print("🔍 Testing correctness...")
        correctness = self.test_correctness(policy, policy_compiled, batch)

        # Print detailed correctness results
        print(f"   Inference correct: {correctness['inference_correct']}")
        print(f"   Training correct: {correctness['training_correct']}")
        print(f"   Max action difference: {correctness['action_diff']:.2e}")
        print(f"   Loss difference: {correctness['loss_diff']:.2e}")

        correctness_passed = correctness["inference_correct"] and correctness["training_correct"]
        if correctness_passed:
            print("✅ Correctness tests passed!")
        else:
            print("⚠️  Correctness test failed - continuing with timing analysis...")

        # Always run timing benchmarks for analysis
        print(f"🚀 Benchmarking inference ({self.n_inference} runs)...")
        try:
            time_orig_inf, _ = self.benchmark_inference(policy, batch)
            time_comp_inf, _ = self.benchmark_inference(policy_compiled, batch)
            inference_benchmarked = True
        except Exception as e:
            print(f"❌ Inference benchmark failed: {e}")
            time_orig_inf = time_comp_inf = float("inf")
            inference_benchmarked = False

        print(f"🚀 Benchmarking training ({self.n_training} runs)...")
        try:
            time_orig_train, losses_orig, grad_norms_orig = self.benchmark_training(policy, batch)
            time_comp_train, losses_comp, grad_norms_comp = self.benchmark_training(policy_compiled, batch)
            training_benchmarked = True

            loss_consistency = (
                np.mean(np.abs(np.array(losses_orig) - np.array(losses_comp)))
                if losses_orig and losses_comp
                else float("inf")
            )
            grad_norm_consistency = (
                np.mean(np.abs(np.array(grad_norms_orig) - np.array(grad_norms_comp)))
                if grad_norms_orig and grad_norms_comp
                else float("inf")
            )
        except Exception as e:
            print(f"❌ Training benchmark failed: {e}")
            time_orig_train = time_comp_train = float("inf")
            loss_consistency = grad_norm_consistency = float("inf")
            training_benchmarked = False

        # Calculate speedups (handle inf cases)
        speedup_inference = (
            time_orig_inf / time_comp_inf if inference_benchmarked and time_comp_inf > 0 else 0.0
        )
        speedup_training = (
            time_orig_train / time_comp_train if training_benchmarked and time_comp_train > 0 else 0.0
        )

        # Success criteria
        compilation_successful = compilation_error is None
        performance_good = speedup_inference > 1.1 and speedup_training > 1.1

        success = (
            compilation_successful
            and correctness_passed
            and performance_good
            and inference_benchmarked
            and training_benchmarked
        )

        results = {
            "success": success,
            "policy": self.policy_name,
            "device": str(self.device),
            "pytorch_version": torch.__version__,
            # Compilation results
            "compilation_successful": compilation_successful,
            "compilation_error": compilation_error,
            # Correctness results
            "correctness": correctness,
            "correctness_passed": correctness_passed,
            # Timing results
            "inference_benchmarked": inference_benchmarked,
            "training_benchmarked": training_benchmarked,
            "time_original_inference": time_orig_inf,
            "time_compiled_inference": time_comp_inf,
            "speedup_inference": speedup_inference,
            "time_original_training": time_orig_train,
            "time_compiled_training": time_comp_train,
            "speedup_training": speedup_training,
            # Consistency metrics
            "loss_consistency": loss_consistency,
            "grad_norm_consistency": grad_norm_consistency,
        }

        # Print detailed summary
        print("\n📊 DETAILED RESULTS SUMMARY")
        print("=" * 60)
        print(f"🔧 Compilation: {'✅ SUCCESS' if compilation_successful else '❌ FAILED'}")
        if compilation_error:
            print(f"   Error: {compilation_error}")

        print(f"🎯 Correctness: {'✅ PASSED' if correctness_passed else '❌ FAILED'}")
        print(f"   Inference diff: {correctness['action_diff']:.2e} (threshold: {self.tolerance:.2e})")
        print(f"   Training diff:  {correctness['loss_diff']:.2e} (threshold: {self.tolerance:.2e})")

        print("⚡ Performance:")
        if inference_benchmarked:
            print(f"   Inference: {time_orig_inf:.2f}ms → {time_comp_inf:.2f}ms ({speedup_inference:.2f}x)")
        else:
            print("   Inference: ❌ Benchmark failed")

        if training_benchmarked:
            print(
                f"   Training:  {time_orig_train:.2f}ms → {time_comp_train:.2f}ms ({speedup_training:.2f}x)"
            )
        else:
            print("   Training:  ❌ Benchmark failed")

        print(f"🏆 Overall: {'✅ SUCCESS' if success else '❌ NEEDS WORK'}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Torch.compile benchmark for LeRobot policies")
    parser.add_argument(
        "--policy", required=True, choices=list(POLICY_CONFIGS.keys()), help="Policy to benchmark"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on"
    )
    parser.add_argument("--output", help="Output file for results (optional)")
    parser.add_argument("--n-inference", type=int, default=100, help="Number of inference runs")
    parser.add_argument("--n-training", type=int, default=50, help="Number of training runs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size to use")
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead"],
        default="default",
        help="Torch compile mode to use.",
    )
    parser.add_argument(
        "--fullgraph",
        action="store_true",
        help="If set, compile the entire model as a single graph and raise an error if graph breaks.",
    )
    parser.add_argument(
        "--disable-dropout",
        action="store_true",
        help="If set, disable dropout layers by setting their dropout rate to 0.",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=["highest", "high", "medium"],
        default=None,
        help="Set float32 matmul precision (only applies when device is cuda)",
    )
    parser.add_argument(
        "--disable-cudnn-tf32",
        action="store_true",
        help="Disallow the use of TensorFloat-32 tensor cores in cuDNN convolutions (only applies when device is CUDA)",
    )

    args = parser.parse_args()

    if args.device == "cuda":
        # Set matmul precision if the argument is specified
        if args.matmul_precision:
            torch.set_float32_matmul_precision(args.matmul_precision)
        # Disable cuDNN TF32 when the argument is provided
        if args.disable_cudnn_tf32:
            torch.backends.cudnn.allow_tf32 = False

    # Run benchmark
    benchmark = TorchCompileBenchmark(args.policy, args.device)

    # Override default parameters if provided
    if args.n_inference:
        benchmark.n_inference = args.n_inference
    if args.n_training:
        benchmark.n_training = args.n_training
    if args.batch_size:
        benchmark.batch_size = args.batch_size

    benchmark.compile_mode = args.compile_mode
    benchmark.fullgraph = args.fullgraph
    benchmark.disable_dropout = args.disable_dropout

    results = benchmark.run_benchmark()

    # Always generate a report (even for failures)
    if "error" in results and len(results) == 2:  # Only basic error info
        # Simple failure case - create basic report
        simple_report = f"""# Benchmark Failed: {benchmark.policy_name.upper()}

## Error
{results.get("error", "Unknown error")}

## Environment
- **Policy**: {benchmark.policy_name}
- **Device**: {benchmark.device}
- **PyTorch**: {torch.__version__}

## Next Steps
1. Check the error message above
2. Ensure the policy and dataset are properly configured
3. Try with a simpler configuration first
"""
        if args.output:
            with open(args.output, "w") as f:
                f.write(simple_report)
            print(f"\n💾 Basic error report saved to {args.output}")
        else:
            print("\n📝 BASIC ERROR REPORT:")
            print(simple_report)
    else:
        # Full benchmark was attempted - generate comprehensive report
        report = benchmark.generate_report(results)

        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            print(f"\n💾 Comprehensive report saved to {args.output}")
        else:
            print("\n📝 COMPREHENSIVE REPORT:")
            print(report)


if __name__ == "__main__":
    main()
