# Action tokenizer benchmark

## Questions

What is the trade-off between:

- **Compression**: how many tokens are needed to represent an action chunk (e.g. horizon × action_dim floats)?
- **Reconstruction quality**: how well does encode-then-decode preserve the original actions?
- **Speed**: how long does encoding and decoding take per chunk?

How to choose an action tokenizer?

- Which tokenizer architecture (e.g. dct + BPE, DCT + BPE)?
- Which **action horizon** and **encoded dimensions** to use?
- Which **normalization** (QUANTILES, MEAN_STD, MIN_MAX) and **delta transform** (relative vs absolute actions)?
- How do reconstruction error and compression ratio vary across datasets and tokenizer settings?

This benchmark loads action chunks from a LeRobot dataset using the same pipeline as `lerobot-train-tokenizer`, runs a trained action tokenizer in encode/decode mode, and reports reconstruction error, compression stats, and timing. Results are saved as JSON under `outputs/` for comparison and analysis.

## Variables

**Dataset & chunking**

- **repo_id**: LeRobot dataset (e.g. `lerobot/pusht`). Action statistics and normalization are taken from the dataset metadata when available.
- **action_horizon**: Number of future steps per action chunk (must match the tokenizer’s training).
- **encoded_dims**: Dimension ranges to encode (e.g. `0:6` or `0:6,7:14`). Must match the tokenizer.
- **max_episodes**: Cap on episodes to load (default: all).
- **sample_fraction**: Fraction of chunks to sample per episode (default `0.2`) to keep runtime manageable.

**Transform & normalization**

- **normalization_mode**: `IDENTITY`, `MEAN_STD`, `MIN_MAX`, `QUANTILES`, `QUANTILE10`. Should match the tokenizer’s training.
- **delta_dims**: Comma-separated dimension indices for delta (relative) transform.
- **use_delta_transform**: Whether to convert actions to relative to current state for those dimensions.
- **state_key**: Dataset key for state (e.g. `observation.state`) used when applying delta transform.

**Tokenizer & evaluation**

- **action_tokenizer_path**: Path or HuggingFace repo id of the trained tokenizer (e.g. `outputs/wavetoken`).
- **max_chunks_for_reconstruction**: Max number of chunks to use for reconstruction and timing (default `500`) to limit runtime.

### Main parameters

| parameter                        | default                      | description                                      |
| -------------------------------- | ---------------------------- | ------------------------------------------------ |
| **action_tokenizer_path**        | (required)                   | Path or Hub id of the trained action tokenizer.  |
| **repo_id**                      | (required)                   | LeRobot dataset repo id.                         |
| **action_horizon**               | `10`                         | Future steps per chunk.                          |
| **encoded_dims**                 | `0:6`                        | Dimension ranges to encode (e.g. `0:6,7:14`).   |
| **normalization_mode**           | `QUANTILES`                  | Normalization mode for actions.                  |
| **max_episodes**                 | all                          | Max episodes to load.                            |
| **sample_fraction**              | `0.2`                        | Fraction of chunks sampled per episode.          |
| **max_chunks_for_reconstruction**| `500`                        | Chunks used for reconstruction and timing.       |
| **output_dir**                   | `outputs/action_tokenizer_benchmark` | Directory for results JSON.              |

## Metrics

**Reconstruction (lower is better)**

- **reconstruction_mae**: Mean absolute error between original and decoded action chunks.
- **reconstruction_mse**: Mean squared error.
- **reconstruction_rmse**: Root mean squared error.
- **reconstruction_max_abs_error**: Maximum absolute error over all dimensions and samples.
- **per_dimension_mae**: MAE per action dimension (list of length `action_dim`).

**Compression**

- **compression_ratio**: Ratio (action_horizon × action_dim) / mean number of tokens. Higher means more compression.
- **mean_token_length**, **std_token_length**: Mean and standard deviation of token count per chunk.
- **min_token_length**, **max_token_length**: Min and max token count.
- **p50_token_length**, **p99_token_length**: 50th and 99th percentile token counts.

**Timing (seconds per chunk)**

- **mean_encode_time_sec**: Mean time to encode one chunk.
- **mean_decode_time_sec**: Mean time to decode one chunk.

The JSON output also includes **num_chunks_evaluated** and **total_chunks_available** for context.

## How the benchmark works

1. **Load dataset**: LeRobot dataset is loaded for the given `repo_id` and `root`.
2. **Build action chunks**: For each episode (up to `max_episodes`), action chunks are built with the same logic as `lerobot-train-tokenizer`: sliding window of length `action_horizon`, optional delta transform, and per-episode sampling with `sample_fraction`.
3. **Extract and normalize**: Only `encoded_dims` are kept. Normalization is applied using the dataset’s action stats when available, according to `normalization_mode`.
4. **Encode / decode**: A random sample of chunks (size `max_chunks_for_reconstruction`) is encoded and then decoded with the tokenizer. Encode and decode times are recorded per chunk.
5. **Compute metrics**: Reconstruction metrics are computed between original and decoded chunks; compression and timing stats are aggregated.
6. **Save results**: A JSON file is written to `output_dir` with name `{timestamp}_{repo_id}_action_tokenizer_results.json`, containing the full config and all metrics.

The pipeline (chunking, dimensions, normalization, delta) must match how the tokenizer was trained; otherwise reconstruction error can be large or the tokenizer may raise.

## Caveats

- The tokenizer’s **action_horizon** and **action_dim** (and optionally DCT settings) are fixed at training time. The benchmark infers dimensions from the dataset and encoded dims; the tokenizer path must correspond to a model trained with the same horizon and encoded dimensions.
- Reconstruction is evaluated in **normalized space** (the same space the tokenizer sees). For interpretation in raw action space, you would need to invert normalization outside this script.
- Only one tokenizer and one dataset are evaluated per run. To compare tokenizers or datasets, run the script multiple times and compare the saved JSON files.

## Example

Quick run with a local tokenizer and a small number of episodes:

```bash
python benchmarks/tokens/run_action_tokenizer_benchmark.py \
    --action-tokenizer-path=outputs/wavetoken \
    --repo-id=lerobot/pusht \
    --action-horizon=10 \
    --max-episodes=50 \
    --output-dir=outputs/action_tokenizer_benchmark
```

With delta transform and custom encoded dimensions:

```bash
python benchmarks/tokens/run_action_tokenizer_benchmark.py \
    --action-tokenizer-path=outputs/wavetoken \
    --repo-id=lerobot/pusht \
    --action-horizon=10 \
    --encoded-dims=0:6,7:14 \
    --delta-dims=0,1,2,3,4,5 \
    --use-delta-transform \
    --normalization-mode=QUANTILES \
    --max-chunks-for-reconstruction=500 \
    --output-dir=outputs/action_tokenizer_benchmark
```

Results are written to e.g. `outputs/action_tokenizer_benchmark/2026-02-12_14-30-00_lerobot_pusht_action_tokenizer_results.json`.

## Results

Results are stored as JSON in the directory given by `--output-dir` (default: `outputs/action_tokenizer_benchmark`). Each file contains:

- **config**: All script arguments (tokenizer path, repo_id, action_horizon, encoded_dims, normalization_mode, etc.) for reproducibility.
- **metrics**: All reconstruction, compression, and timing metrics described above.

To compare runs, load and diff or aggregate these JSON files with your own scripts or notebooks.
