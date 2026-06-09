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

`--source` is a label only; the actual source is whatever `--repo_id` / `--root` / `--data_files_root`
point at.

### GPU (NVDEC) decoding

By default video is decoded on the **CPU** in each DataLoader worker, so throughput is CPU-decode-bound and
scales with `--num_workers` (capped by the dataset's `num_shards`). Pass `--video_decode_device cuda` to
offload H.264/H.265 decode to the GPU's dedicated **NVDEC** engine, which runs independently of the SMs used
for training (see <https://developer.nvidia.com/video-codec-sdk>). This requires a CUDA-enabled torchcodec
build, and because CUDA cannot initialize in forked workers the benchmark switches to the `spawn` start
method automatically when `--num_workers > 0`.

```bash
# GPU/NVDEC decode, 6 workers, bucket source
python benchmarks/streaming/benchmark_streaming.py \
    --repo_id pepijn223/robocasa_pretrain_human300_v4 \
    --data_files_root hf://buckets/pepijn223/robocasa-stream \
    --mode sarm --batch_size 64 --num_workers 6 --num_batches 200 \
    --video_decode_device cuda --source bucket
```

Caveats with `cuda` + many workers: each worker creates its own CUDA context (VRAM overhead) and NVDEC has a
limited number of concurrent decode sessions per GPU; if you hit session/IPC limits, reduce `--num_workers`
or compare against `--num_workers 0` (single-process NVDEC, which often saturates the decode engine on its
own). Result files include the decode device in their name (`..._w6_cuda.json`).

> **Codec ⇄ NVDEC compatibility (important).** NVDEC can only decode codecs its hardware supports. LeRobot
> v3 datasets are often **AV1**-encoded, and the **A100 and H100 compute GPUs have no AV1 NVDEC decoder**
> (per NVIDIA's [decode support matrix](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new));
> only Ada (L4/L40/RTX40) and a few Ampere cards (A10/A40/A16) do. On A100/H100, AV1 must be decoded on
> **CPU**, or the dataset re-encoded to H.265/H.264 (which those GPUs' NVDEC do support). Run
> `diagnose_decode.py --video_decode_device cuda` to check your exact node before relying on `cuda` decode.
> A `cuda` torchcodec build also needs an FFmpeg with NVDEC; see
> <https://github.com/meta-pytorch/torchcodec#installing-cuda-enabled-torchcodec>.

Reference data root: bucket sources resolve through `--data_files_root hf://buckets/<owner>/<name>` (metadata
still loads from `--repo_id`). The local `single`/`sarm` CPU baselines on this dataset were ~176 / ~212
frames/s/node at `--num_workers 3` (3 cameras, fps 20).

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
