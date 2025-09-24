"""
Benchmark memory footprint and inference latency of a policy on arbitrary devices.

This script loads a pretrained policy directly (similar to the async inference server)
and generates dummy input data based on the policy's input_features to perform
accurate benchmarking without requiring datasets.
"""

import argparse
import os
import signal
import statistics
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import psutil
import torch
from tqdm import tqdm

from lerobot.configs.types import FeatureType
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy


class TimeoutExceptionError(Exception):
    pass


@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutExceptionError(f"Timed out after {seconds} seconds")

    # On Windows, signal is not available, so we can't use this timeout mechanism
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    try:
        # signal.alarm expects integer seconds
        # for float seconds, we can use setitimer
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def bytes_to_human(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def generate_dummy_observation(input_features: dict, device: str = "cpu") -> dict:
    """Generate dummy observation data based on policy input features."""
    dummy_obs = {}

    for key, feature in input_features.items():
        shape = feature.shape

        if feature.type == FeatureType.VISUAL:
            # Images: random values in [0, 1] range (already normalized)
            dummy_obs[key] = torch.rand(shape, dtype=torch.float32, device=device)
        elif feature.type in [FeatureType.STATE, FeatureType.ACTION, FeatureType.ENV]:
            # State/action/env: random normal distribution
            dummy_obs[key] = torch.randn(shape, dtype=torch.float32, device=device)
        else:
            # Default: random normal for unknown types
            dummy_obs[key] = torch.randn(shape, dtype=torch.float32, device=device)

    # # Add batch dimension
    # for key in dummy_obs:
    #     dummy_obs[key] = dummy_obs[key].unsqueeze(0)

    # Add task string for language-conditioned policies
    dummy_obs["task"] = " this is a dummy task"

    return dummy_obs


def main():
    parser = argparse.ArgumentParser(description="Policy inference benchmark")
    parser.add_argument(
        "--policy-id", type=str, required=True, help="Model ID or local path to pretrained policy"
    )
    parser.add_argument(
        "--policy-type", type=str, required=True, help="Type of policy (smolvla, act, diffusion, etc.)"
    )
    parser.add_argument(
        "--device", type=str, default="mps", choices=["cuda", "cpu", "mps"], help="Device to run on"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num-samples", type=int, default=100, help="Number of inference samples to benchmark"
    )
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup samples (not timed)")
    parser.add_argument(
        "--output-dir", type=str, default="outputs/benchmarks", help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.3,
        help="Timeout for each inference pass in seconds (default: 0.3s = 300ms)",
    )
    args = parser.parse_args()

    # Seed & deterministic-ish setup
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False  # leave False to avoid perf cliffs

    # Resolve device availability
    device = args.device.lower()
    if device == "cuda" and not torch.cuda.is_available():
        print("[!] CUDA requested but unavailable. Falling back to CPU.")
        device = "cpu"
    elif device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        print("[!] MPS requested but unavailable. Falling back to CPU.")
        device = "cpu"

    use_cuda = device == "cuda"

    # Create output directory and log file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    policy_name = args.policy_id.replace("/", "_").replace("\\", "_")
    log_file = output_dir / f"benchmark_{args.policy_type}_{policy_name}_{device}_{timestamp}.txt"

    # Load policy directly from pretrained (similar to async inference server)
    print(f"Loading policy {args.policy_type} from {args.policy_id}...")
    policy_class = get_policy_class(args.policy_type)
    policy: PreTrainedPolicy = policy_class.from_pretrained(args.policy_id)
    policy.eval()
    policy.to(device, torch.float32)
    policy.config.device = device
    preprocessor, postprocessor = make_pre_post_processors(policy.config)

    print(f"Policy loaded on {device}")
    print(f"Input features: {list(policy.config.input_features.keys())}")
    print(f"Output features: {list(policy.config.output_features.keys())}")

    # Generate dummy observation based on policy input features
    dummy_observation = generate_dummy_observation(policy.config.input_features, device)
    dummy_observation["task"] = "this is a dummy task"

    # Helper to sync for fair timings
    def _sync(dev_=device):
        if dev_ == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif dev_ == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.synchronize()
            except AttributeError:
                pass  # MPS sync not available in this PyTorch version

    # Warmup (to stabilize kernels/caches)
    print("Warming up...")
    with torch.no_grad():
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        orginal_dummy_observation = deepcopy(dummy_observation)
        for _ in range(args.warmup):
            dummy_observation_model = deepcopy(orginal_dummy_observation)
            dummy_observation_model = preprocessor(dummy_observation_model)
            action_model = policy.select_action(dummy_observation_model)
            _ = postprocessor(action_model)
            policy.reset()
        _sync()

    # Memory footprint before timing
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    # PyTorch timing with Event objects for more accurate GPU timing
    print(f"Running benchmark: {args.num_samples} samples...")

    if use_cuda:
        # Use CUDA Events for precise GPU timing
        start_events = []
        end_events = []
        timeout_count = 0
        orginal_dummy_observation = deepcopy(dummy_observation)

        with torch.no_grad():
            for forward in tqdm(range(args.num_samples), desc="Trials"):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                try:
                    dummy_observation_model = deepcopy(orginal_dummy_observation)
                    dummy_observation_model = preprocessor(dummy_observation)
                    with timeout(args.timeout):
                        start_event.record()
                        action_model = policy.select_action(dummy_observation_model)
                        end_event.record()
                    _ = postprocessor(action_model)
                    policy.reset()

                    start_events.append(start_event)
                    end_events.append(end_event)
                except TimeoutExceptionError:
                    timeout_count += 1
                    # Add placeholder for timeout
                    start_events.append(None)
                    end_events.append(None)
                    print(f"\n[!] Timeout on forward {forward + 1}")
                    continue

        # Synchronize and collect timing results
        torch.cuda.synchronize()
        per_forward_ms = []
        for start_event, end_event in zip(start_events, end_events, strict=True):
            if start_event is None:
                per_forward_ms.append(args.timeout * 1000)
            else:
                per_forward_ms.append(start_event.elapsed_time(end_event))

        if timeout_count > 0:
            print(f"[!] {timeout_count} inference passes timed out (>{args.timeout * 1000:.1f}ms)")

    else:
        # Use simple time.perf_counter for CPU/MPS timing with timeout
        import time

        per_forward_ms = []
        timeout_count = 0

        with torch.no_grad():
            for sample in tqdm(range(args.num_samples), desc="Samples"):
                try:
                    dummy_observation_model = deepcopy(orginal_dummy_observation)
                    dummy_observation_model = preprocessor(dummy_observation_model)
                    with timeout(args.timeout):
                        start_time = time.perf_counter()
                        action_model = policy.select_action(dummy_observation_model)
                        end_time = time.perf_counter()
                    policy.reset()

                    per_forward_ms.append((end_time - start_time) * 1000)  # Convert to ms
                except TimeoutExceptionError:
                    timeout_count += 1
                    per_forward_ms.append(args.timeout * 1000)
                    print(f"\n[!] Timeout on sample {sample + 1}")
                    continue

        if timeout_count > 0:
            print(f"[!] {timeout_count} inference passes timed out (>{args.timeout * 1000:.1f}ms)")
            print(f"Timeout percentage: {timeout_count / args.num_samples * 100:.1f}%")

    # Memory footprint after timing
    rss_after = process.memory_info().rss
    rss_delta = rss_after - rss_before
    cuda_peak = torch.cuda.max_memory_allocated() if use_cuda else 0

    # Sort timing results for percentile calculations
    per_forward_ms_sorted = sorted(per_forward_ms)

    mean_ms = statistics.fmean(per_forward_ms) if per_forward_ms else float("nan")
    std_ms = statistics.pstdev(per_forward_ms) if len(per_forward_ms) > 1 else 0.0
    min_ms = per_forward_ms_sorted[0] if per_forward_ms_sorted else float("nan")
    max_ms = per_forward_ms_sorted[-1] if per_forward_ms_sorted else float("nan")
    p50_ms = percentile(per_forward_ms_sorted, 50)
    p95_ms = percentile(per_forward_ms_sorted, 95)

    # Model size
    num_params = sum(p.numel() for p in policy.parameters())

    # Prepare results for logging
    results = {
        "timestamp": datetime.now().isoformat(),
        "policy_type": args.policy_type,
        "policy_id": args.policy_id,
        "device": device,
        "num_trials": args.num_samples,
        "forwards_per_trial": 1,
        "warmup": args.warmup,
        "timeout_ms": args.timeout * 1000,
        "seed": args.seed,
        "num_params": num_params,
        "timeout_count": timeout_count,
        "latency_mean_ms": mean_ms,
        "latency_std_ms": std_ms,
        "latency_min_ms": min_ms,
        "latency_max_ms": max_ms,
        "latency_p50_ms": p50_ms,
        "latency_p95_ms": p95_ms,
        "cpu_rss_before": rss_before,
        "cpu_rss_after": rss_after,
        "cpu_rss_delta": rss_delta,
        "cuda_peak_alloc": cuda_peak,
        "input_features": list(policy.config.input_features.keys()),
        "output_features": list(policy.config.output_features.keys()),
    }

    # Format and write results to log file
    log_content = f"""
=== LeRobot Policy Inference Benchmark ===
Timestamp: {results["timestamp"]}
Policy: {results["policy_type"]} ({results["policy_id"]})
Device: {results["device"]}
Seed: {results["seed"]}

=== Model Information ===
Parameters: {results["num_params"]:,}
Input Features: {", ".join(results["input_features"])}
Output Features: {", ".join(results["output_features"])}

=== Benchmark Configuration ===
Samples: {results["num_trials"]}
Warmup: {results["warmup"]}
Total Measurements: {len(per_forward_ms)}
Timeout: {results["timeout_ms"]:.1f}ms
Timeouts: {results["timeout_count"]} / {results["num_trials"]}

=== Latency Results (ms) ===
Mean:     {results["latency_mean_ms"]:.3f}
Std Dev:  {results["latency_std_ms"]:.3f}
Min:      {results["latency_min_ms"]:.3f}
Max:      {results["latency_max_ms"]:.3f}
P50:      {results["latency_p50_ms"]:.3f}
P95:      {results["latency_p95_ms"]:.3f}

=== Memory Footprint ===
CPU RSS Before: {bytes_to_human(results["cpu_rss_before"])}
CPU RSS After:  {bytes_to_human(results["cpu_rss_after"])} (Δ {bytes_to_human(results["cpu_rss_delta"])})
"""

    if use_cuda:
        log_content += f"CUDA Peak:      {bytes_to_human(results['cuda_peak_alloc'])} (reset before timing)\n"

    log_content += f"""
=== Raw Timing Data (first 20 measurements, ms) ===
{", ".join(f"{t:.3f}" for t in per_forward_ms[:20])}
{"..." if len(per_forward_ms) > 20 else ""}

=== Summary Statistics ===
Timing Method: {"CUDA Events" if use_cuda else "torch.utils.benchmark.Timer"}
Device Available: {torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available() if device == "mps" else True}
PyTorch Version: {torch.__version__}

Benchmark completed successfully at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    # Write to log file
    with open(log_file, "w") as f:
        f.write(log_content)

    # Print to console (shorter version)
    print("\n=== Inference Benchmark Results ===")
    print(f"Policy: {args.policy_type} ({args.policy_id})")
    print(f"Device: {device}")
    print(f"Samples: {args.num_samples} | Warmup: {args.warmup}")
    print(f"Model params: {num_params:,}")

    print("\nLatency per forward (ms):")
    print(f"  mean: {mean_ms:.3f}  std: {std_ms:.3f}")
    print(f"  min:  {min_ms:.3f}   max: {max_ms:.3f}")
    print(f"  p50:  {p50_ms:.3f}   p95: {p95_ms:.3f}")

    print("\nMemory footprint:")
    print(f"  CPU RSS before: {bytes_to_human(rss_before)}")
    print(f"  CPU RSS after : {bytes_to_human(rss_after)}  (Δ {bytes_to_human(rss_delta)})")
    if use_cuda:
        print(
            f"  CUDA peak allocated: {bytes_to_human(cuda_peak)} "
            f"(reset by reset_peak_memory_stats before timing)"
        )

    print(f"\nResults saved to: {log_file}")
    print("Benchmark completed successfully!")


if __name__ == "__main__":
    main()
