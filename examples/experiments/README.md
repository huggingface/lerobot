# Latency-Adaptive Async Inference Experiments

This directory contains experiment infrastructure for validating and characterizing the latency-adaptive async inference algorithm described in the paper.

**Important**: These experiments run on a REAL ROBOT with a REAL POLICY. The policy server must be started separately before running experiments.

## Overview

The experiment framework supports:
- **Latency estimator comparison**: JK (Jacobson-Karels) vs max-of-last-10
- **Cooldown on/off comparison**: Testing the cooldown mechanism
- **Parameter sweeps**: K (deviation scaling), epsilon (safety margin)
- **Drop recovery testing**: Simulated observation and action drops to test robustness

## Quick Start

```bash
# 1. Start the policy server (in a separate terminal)
uv run python examples/tutorial/async-inf/policy_server_improved.py

# 2. Run experiments (in another terminal)
uv run python examples/experiments/latency_adaptive_sweep.py \
    --sweep estimator_comparison \
    --duration_s 60 \
    --output_dir results/estimator_sweep/

# 3. Plot the results
uv run python examples/experiments/plot_results.py \
    --input results/sweep/ \
    --mode estimator_comparison \
    --output results/estimator_comparison.png
```

## Running Experiments

### Prerequisites

1. **Policy server running**: Start the server in a separate terminal:
   ```bash
   uv run python examples/tutorial/async-inf/policy_server_improved.py
   ```

2. **Robot connected**: The SO101 follower robot must be connected and calibrated.

3. **Environment ready**: Position the robot and task objects (e.g., orange cube, X marker).

### Predefined Sweeps

| Sweep Name | Description | Experiments |
|------------|-------------|-------------|
| `estimator_comparison` | Compare JK vs max_last_10 with cooldown on/off | 2 configs |
| `k_parameter` | Sweep K (deviation scaling factor) | 5 configs |
| `epsilon` | Sweep epsilon (safety margin) | 5 configs |
| `quick_test` | Quick test (JK with cooldown on/off, 30s each) | 2 configs |
| `obs_drop` | Test observation drop recovery (random and burst) | 4 configs |
| `action_drop` | Test action chunk drop recovery | 3 configs |
| `drop_recovery` | Compare cooldown vs no-cooldown under drops | 2 configs |
| `spike` | Test JK estimator adaptation to latency spikes | 4 configs |
| `spike_estimator` | Compare JK vs max_last_10 under latency spikes | 2 configs |

Run a sweep:

```bash
uv run python examples/experiments/latency_adaptive_sweep.py \
    --sweep estimator_comparison \
    --duration_s 60 \
    --pause_between_s 10 \
    --output_dir results/sweep/
```

The script pauses between experiments to allow robot/environment reset.

### Single Experiment

Run a custom single experiment:

```bash
uv run python examples/experiments/latency_adaptive_sweep.py \
    --estimator jk \
    --cooldown on \
    --latency_k 1.5 \
    --epsilon 1 \
    --duration_s 60 \
    --output_dir results/
```

### Drop Testing

The experiment runner supports simulated observation and action drops to test the robustness of the latency-adaptive algorithm. This is key for validating the cooldown mechanism's ability to recover from network issues.

#### Random Drops

Simulate random packet loss (e.g., 5% of observations dropped):

```bash
uv run python examples/experiments/latency_adaptive_sweep.py \
    --drop_obs_random_p 0.05 \
    --duration_s 60 \
    --output_dir results/drop_test/
```

#### Burst Drops

Simulate periodic network outages (e.g., 1 second drop every 20 seconds):

```bash
uv run python examples/experiments/latency_adaptive_sweep.py \
    --drop_obs_burst_period_s 20 \
    --drop_obs_burst_duration_s 1 \
    --duration_s 60 \
    --output_dir results/drop_test/
```

#### Action Chunk Drops

Simulate dropped action chunks from the server:

```bash
uv run python examples/experiments/latency_adaptive_sweep.py \
    --drop_action_random_p 0.05 \
    --duration_s 60 \
    --output_dir results/drop_test/
```

#### Drop Recovery Sweep

Compare cooldown vs no-cooldown behavior under drops:

```bash
uv run python examples/experiments/latency_adaptive_sweep.py \
    --sweep drop_recovery \
    --duration_s 60 \
    --output_dir results/drop_recovery/
```

#### Drop Parameters

| Parameter | Description |
|-----------|-------------|
| `--drop_obs_random_p` | Random observation drop probability (0.0-1.0) |
| `--drop_obs_burst_period_s` | Time between observation drop bursts (0 = disabled) |
| `--drop_obs_burst_duration_s` | Duration of each observation drop burst |
| `--drop_action_random_p` | Random action chunk drop probability (0.0-1.0) |
| `--drop_action_burst_period_s` | Time between action chunk drop bursts (0 = disabled) |
| `--drop_action_burst_duration_s` | Duration of each action chunk drop burst |

### Latency Spike Testing

Test how the latency-adaptive algorithm handles sudden increases in inference time (e.g., GPU throttling, model loading). Spike configuration is passed from the experiment runner to the server, so all parameters are tracked together.

Spikes are defined as explicit events: each spike fires once at a specific time, adding a specified delay.

#### Single Experiment with Spikes

```bash
# Add a 2s spike at 5s into the experiment
uv run python examples/experiments/latency_adaptive_sweep.py \
    --spikes '[{"start_s": 5, "delay_ms": 2000}]' \
    --duration_s 30 \
    --output_dir results/spike_test/

# Multiple spikes at 5s and 15s
uv run python examples/experiments/latency_adaptive_sweep.py \
    --spikes '[{"start_s": 5, "delay_ms": 2000}, {"start_s": 15, "delay_ms": 1000}]' \
    --duration_s 30 \
    --output_dir results/spike_test/
```

#### Spike Sweep

Run predefined spike experiments:

```bash
uv run python examples/experiments/latency_adaptive_sweep.py \
    --sweep spike \
    --output_dir results/spike_sweep/
```

#### Compare Estimators Under Spikes

```bash
uv run python examples/experiments/latency_adaptive_sweep.py \
    --sweep spike_estimator \
    --output_dir results/spike_estimator/
```

#### Spike Format

Spikes are passed as a JSON array via `--spikes`:

```json
[
  {"start_s": 5, "delay_ms": 2000},
  {"start_s": 15, "delay_ms": 1000}
]
```

| Field | Description |
|-------|-------------|
| `start_s` | When to trigger the spike (seconds from experiment start) |
| `delay_ms` | How much delay to add when triggered (milliseconds) |

Each spike fires exactly once when the elapsed time crosses its `start_s` threshold.

## Plotting Results

### Basic Plot

Plot metrics from one or more CSV files:

```bash
uv run python examples/experiments/plot_results.py \
    --input results/experiment.csv \
    --output results/plot.png
```

### Estimator Comparison

Compare JK vs max-of-last-10 estimators:

```bash
uv run python examples/experiments/plot_results.py \
    --input results/estimator_sweep/ \
    --mode estimator_comparison \
    --output results/estimator_comparison.png
```

### Detailed Single-Experiment Analysis

Get detailed plots including schedule size and L2 metrics:

```bash
uv run python examples/experiments/plot_results.py \
    --input results/single_experiment.csv \
    --mode detailed \
    --output results/detailed_analysis.png
```

### Filtering Files

Plot only files matching a pattern:

```bash
uv run python examples/experiments/plot_results.py \
    --input results/ \
    --filter "jk" \
    --output results/jk_only.png
```

## Configuration Reference

### Latency Estimator Types

| Type | Description | Use Case |
|------|-------------|----------|
| `jk` | Jacobson-Karels with exponential smoothing | Default, fast recovery from spikes |
| `max_last_10` | Max of last 10 measurements | Conservative, RTC-style |

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latency_k` | 1.5 | JK scaling factor for deviation (K). Higher = more conservative |
| `epsilon` | 1 | Safety margin in action steps. Triggers inference earlier |
| `cooldown` | on | Enable cooldown mechanism for drop recovery |

### Hardware Configuration

The experiment runner uses these defaults (matching the tutorial):

```python
DEFAULT_SERVER_ADDRESS = "192.168.4.37:8080"
DEFAULT_ROBOT_PORT = "/dev/ttyACM0"
DEFAULT_ROBOT_ID = "so101_follower_2026_01_03"
DEFAULT_MODEL_PATH = "/home/jack/code/self-driving-screwdriver-robot/..."
```

Override the server address via CLI:

```bash
uv run python examples/experiments/latency_adaptive_sweep.py \
    --server_address 192.168.1.100:8080 \
    ...
```

## Metrics Collected

Each experiment generates a CSV file with per-tick metrics:

| Column | Description |
|--------|-------------|
| `t` | Timestamp (seconds since epoch) |
| `step` | Action step counter |
| `schedule_size` | Number of actions in schedule |
| `latency_estimate_steps` | Current latency estimate (in action steps) |
| `cooldown` | Cooldown counter value |
| `stall` | 1 if no actions available, 0 otherwise |
| `obs_sent` | 1 if observation was sent this tick |
| `action_received` | 1 if actions were received this tick |
| `measured_latency_ms` | Measured RTT when actions received |
| `chunk_mean_l2` | Mean L2 distance of overlapping actions |
| `chunk_max_l2` | Max L2 distance of overlapping actions |

## Experiment Workflow

### Typical Workflow

1. **Start server** in one terminal:
   ```bash
   uv run python examples/tutorial/async-inf/policy_server_improved.py
   ```

2. **Run experiments** in another terminal:
   ```bash
   uv run python examples/experiments/latency_adaptive_sweep.py \
       --sweep estimator_comparison \
       --duration_s 60 \
       --output_dir results/
   ```

3. **Between experiments**: The script pauses to let you reset the robot/environment.

4. **Analyze results**:
   ```bash
   uv run python examples/experiments/plot_results.py \
       --input results/ \
       --mode estimator_comparison \
       --output results/analysis.png
   ```

### Cooldown Mechanism

The cooldown mechanism (`--cooldown on`) is the paper's key contribution:
- Cooldown counter decrements each tick
- New inference triggers when cooldown reaches 0 AND schedule is low
- **Enables recovery from network drops** (unlike RTC-style merge-reset)

## File Structure

```
examples/experiments/
├── README.md                    # This file
├── __init__.py
├── latency_adaptive_sweep.py    # Main experiment runner
├── plot_results.py              # Plotting utilities
└── archived/                    # RTC sweep code (not for paper)
    ├── rtc_sweep.py
    └── plot_rtc_sweep.py
```

## Tips

1. **Start with short durations** (30-60s) to verify setup before long runs
2. **Use simulation mode** (`simulation_mode=True`) for quick iteration
3. **Check server is running** before starting client experiments
4. **Monitor GPU memory** when running with real policies
5. **Plot incrementally** to catch issues early

## Troubleshooting

### Client failed to connect
- Ensure the policy server is running (`python examples/tutorial/async-inf/policy_server_improved.py`)
- Check the server address matches (`--server_address`)
- Verify network connectivity between client and server machines

### Robot connection failed
- Check robot is connected (`/dev/ttyACM0`)
- Verify calibration files exist for robot ID

### No metrics file created
- Check write permissions on output directory
- Look for errors in the experiment output

### Experiment interrupted
- Press Ctrl+C to gracefully stop
- Metrics are flushed on stop, partial data is saved
