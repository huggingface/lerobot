# Streaming dataloading benchmark

Measures **dataloading only** (no model) for `StreamingLeRobotDataset`: parquet read + video decode +
delta windowing + shuffle. A dummy consumer pulls batches and moves them to the device, so the numbers
isolate the data pipeline. Use it to compare sources (Hub vs. storage bucket vs. prewarmed bucket),
frame modes, and node counts, and to catch p95/p99 video-decode regressions.

## Run

```bash
python benchmarks/streaming/benchmark_streaming.py \
    --repo_id pepijn223/robocasa_pretrain_human300_v4 \
    --mode sarm --batch_size 64 --num_workers 12 --num_batches 200 \
    --source hub --out_dir benchmarks/streaming/results
```

Multinode (per-node throughput) goes through Accelerate under SLURM:

```bash
sbatch slurm/benchmark_streaming_robocasa.sh
```

## Matrix

| Axis       | Values                                                                                                               |
| ---------- | -------------------------------------------------------------------------------------------------------------------- |
| Source     | `hub` (verify now), `bucket`, `warmed_bucket` (bucket + prewarming; with user's help later)                          |
| Baseline   | current `main` `StreamingLeRobotDataset` on Hub streaming                                                            |
| Nodes      | 1 and 2 (per-node throughput should be independent)                                                                  |
| Frame mode | `single` (1 frame, all cameras; target ≥ 120 frames/s/node) · `sarm` (8 steps spaced 1s; target ≥ 320 frames/s/node) |

`--source` is a label only; the actual source is whatever `--repo_id` / `--root` point at.

## Metrics emitted (JSON + CSV)

`frames_per_s_node`, `samples_per_s`, `first_batch_latency_s`, `p50/p95/p99_sample_latency_ms`,
`wallclock_s`, and `video_decoder_cache` (`hits`, `misses`, `evictions`, `hit_rate`, `size`). A low
cache `hit_rate` with high `p99` is the decoder-thrash signature — raise `--video_decoder_cache_size`
or `--buffer_size`, or reduce `num_workers`.

## Bucket sources & prewarming (manual)

Prewarming is a **server-side** Hugging Face storage-bucket feature — there is no client script. To
benchmark the `warmed_bucket` source:

1. Attach a storage bucket to the dataset and enable it (see
   <https://huggingface.co/docs/hub/storage-buckets>). Buckets resolve through `fsspec`, the same as
   `hf://`, so no code change is needed — point `--repo_id`/`--revision` (or `--root`) at the bucket.
2. Enable **prewarming** in the bucket settings and wait for warm-up to complete.
3. Run the benchmark with `--source warmed_bucket`. Compare against the cold `--source bucket` and the
   `--source hub` baseline.

Manual only — not run in CI.
