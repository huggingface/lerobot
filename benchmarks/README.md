# Policy profiling

Profile both fresh-action inference and training updates without changing the normal training loop. The weekly and manually dispatched [Model Profiling workflow](../.github/workflows/model_profiling.yml) discovers every registered `PreTrainedConfig` policy automatically. Each policy runs in its own dependency environment, and a failure does not stop the others. Reward models have a separate registry and are outside this policy benchmark.

Automatic profiling runs once a week at **01:00 Monday, Europe/Paris** (Sunday night into Monday). The timezone follows daylight-saving changes. The schedule takes effect on the default branch after merge; manual runs remain available for debugging.

Run one policy locally with its usual training configuration:

```bash
uv sync --locked --extra training
uv run python benchmarks/profile_policies.py \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/pusht --dataset.episodes='[0]' \
  --batch_size=1 --output_dir=profiling-results/act
```

Use `--profile_mode=inference` or `--profile_mode=training` to select one operation. Defaults are both operations, three warmup calls, and ten measured calls. Override with `--warmup` and `--iterations`. Add `--trace=true` to capture a separate Torch trace for each operation after timing. Open the trace in [Perfetto](https://ui.perfetto.dev/).

All ordinary policy, processor, dataset and optimizer CLI arguments are accepted, including `--policy.path` for a trained checkpoint and `--accelerator.mixed_precision=bf16`. This is a single-device benchmark using one fixed, preprocessed dataset batch. It does not support streaming, distributed training, PEFT or checkpoint resume.

## Measurements

`result.json` contains raw latency samples, median and p95 latency in milliseconds, peak allocated and reserved CUDA memory in bytes, policy and dataset identity, commit, device, PyTorch/CUDA versions, dtype, warmup and batch size. It also stores policy, optimizer and accelerator settings with a configuration fingerprint for history comparisons. CPU/MPS CUDA-memory fields are null. The effective configuration and preprocessing state are saved beside the result.

- **Inference:** reset policy state and call `select_action` for every sample. This includes computing a fresh action chunk where applicable; queued-action cache hits cannot make the benchmark appear faster. Observations use timestamp zero from the dataset windows. Inference runs before training changes the weights.
- **Training:** reuse `lerobot_train.update_policy`, including backward, gradient clipping, optimizer and scheduler updates. Gaussian actor uses the existing SAC update with real dataset rewards and adjacent next observations, including actor, critics, temperature and target updates.
- **Timing:** exclude downloads, model setup, video decoding, preprocessing, batch copying, warmup and optional Torch tracing. Synchronize the accelerator around each timed call. These measurements describe model calls on a fixed batch, rather than end-to-end data throughput or task success.

The default CI recipes use policy defaults, PushT for the smaller policies and single-camera Flux3, and LIBERO Plus for the others. VQ-BeT completes its codebook stage after one warmup update so measured training exercises BeT. These are profiling fixtures, not a comparison of trained policy quality. Default initialization may load pretrained backbones while leaving other weights randomly initialized. Use explicit checkpoint and dataset revisions for controlled comparisons. Model-specific assets, permissions, GPU capacity and dependency requirements still apply; setup failures and OOMs appear as failures with logs, not successful coverage.

The recipes compute missing vector quantiles for DM05, Pi0.5 and MolmoAct2 from the selected dataset rows (`--compute_missing_quantiles=true`), using the existing histogram estimator without modifying the Hub dataset. Flux3 reuses its base-export helper for command-delta statistics. FineART-VLA uses ten episodes for its default tokenizer fit. PI0-FAST uses a pinned trained checkpoint so autoregressive output can be decoded, while retaining its default training configuration. XVLA reads the required Florence backbone architecture from a pinned base configuration. Camera names follow the fixture dataset. All preparation is outside timing, and effective settings and preprocessing state are preserved in the artifacts.

## Publishing

Every workflow run produces a GitHub job summary, a `Profiling results` check linking directly to the public dashboard, and a downloadable `model-profiling-report-<attempt>` artifact containing `results.json`, a Markdown table, `comparison.json` with baseline provenance, and the exact Slack payload (`slack.json`). Separate `model-profiling-<policy>-<attempt>` artifacts contain the effective configurations, individual results and optional traces. A filtered manual run reports only the requested policies. Failed and missing results remain visible.

The public [LeRobot health dashboard](https://huggingface.co/spaces/lerobot/health-dashboard) provides a sortable results table, error details and report downloads. Its History view charts inference or training latency by policy over the last month, three months, or all recorded runs. Matching setup comparisons use hardware, batch size, dtype, dataset and configuration fingerprint; source commits can differ.

The workflow preserves each run and attempt under `data/<run-id>-<attempt>/` in the Space, using the `MODEL_PROFILING_HF_TOKEN` (or the existing `LEROBOT_HF_USER` fallback) secret with write access. Repository variable `MODEL_PROFILING_SPACE` overrides the destination. Archived JSON, report ZIP and source metadata remain available without GitHub authentication or artifact expiration. A valid server-side GitHub token also enables live results.

Weekly runs post one compact table to the existing `CI_SLACK_CHANNEL`, using `SLACK_CIFEEDBACK_BOT_TOKEN`. Manual runs post only when `notify_slack` is enabled. Use `publish_latest` to regenerate and optionally post the latest archived table without allocating GPUs; each row retains its source run, and this creates no duplicate history measurements. Policies marked `*` use native gradient checkpointing. Each policy shows current inference/training median milliseconds and a separate four-week latency change (negative is faster, positive is slower). The “Check Action results” button opens the public Space; a separate link opens the GitHub run. Slack delivery and dashboard upload run independently, so an upload failure does not suppress the message.

The baseline is the nearest archived measurement to 28 days before each measurement run started, within three days either side. It must match GPU model/capacity, batch size, dtype, mixed precision, warmup/iteration counts, dataset/revision/episodes, checkpoint/revision, configuration fingerprint and checkpointing state. Inference and training choose successful baselines independently; failed measurements, cancelled runs and unrecorded setups are excluded. If no matching baseline exists, the change is `—`. If history cannot be fetched, current results are still published with an explicit note. History starts with the first retained measurement; older numbers are never inferred.

GitHub's raw per-policy artifacts follow the repository's retention policy. Download configurations and optional traces you need to retain separately.

PI0-FAST uses the pinned LIBERO checkpoint with its matching action tokenizer, horizon and camera layout, because the base checkpoint does not reliably emit a valid action prefix. Its optimizer, float32 dtype and training memory settings retain the policy defaults.

LingBot-VA uses `attn_mode=flex`, which its block-causal flow-matching training requires. The attention backend is recorded in the configuration so its measurements are not compared with the default inference-only backend.

Manual workflow inputs can select `runner_group` and enable `gradient_checkpointing` where the policy exposes a native switch. The default runner is `aws-g6-4xlarge-plus` (L4); repository variable `MODEL_PROFILING_RUNNER_GROUP` can change it. Hugging Face's `aws-g6e-4xlarge` group is an L40S candidate, subject to repository access. Results record the actual GPU, its memory capacity, the checkpointing state and the full configuration; history matches setups separately.

Checkpointing trades extra forward computation for lower activation memory. It leaves the optimizer unchanged and cannot remove parameter or optimizer-state memory. DM05 already enables its native checkpointing by default. Policies without a native checkpointing switch keep their existing setup. Multi-GPU sharding is outside this single-device benchmark; adding GPUs with DDP does not pool their memory.
