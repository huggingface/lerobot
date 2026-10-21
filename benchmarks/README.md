# Policy profiling

Profile both fresh-action inference and training updates without changing the normal training loop. The weekly and manually dispatched [Model Profiling workflow](../.github/workflows/model_profiling.yml) discovers every registered `PreTrainedConfig` policy automatically. Each policy runs in its own dependency environment, and a failure does not stop the others. Reward models have a separate registry and are outside this policy benchmark.

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

`result.json` contains raw latency samples, median and p95 latency in milliseconds, peak allocated and reserved CUDA memory in bytes, policy and dataset identity, commit, device, PyTorch/CUDA versions, dtype, warmup and batch size. CPU/MPS CUDA-memory fields are null. The effective configuration is saved beside the result.

- **Inference:** reset policy state and call `select_action` for every sample. This includes computing a fresh action chunk where applicable; queued-action cache hits cannot make the benchmark appear faster. Observations use timestamp zero from the dataset windows. Inference runs before training changes the weights.
- **Training:** reuse `lerobot_train.update_policy`, including backward, gradient clipping, optimizer and scheduler updates. Gaussian actor uses the existing SAC update with real dataset rewards and adjacent next observations, including actor, critics, temperature and target updates.
- **Timing:** exclude downloads, model setup, video decoding, preprocessing, batch copying, warmup and optional Torch tracing. Synchronize the accelerator around each timed call. These measurements describe model calls on a fixed batch, rather than end-to-end data throughput or task success.

The default CI recipes use policy defaults, PushT for the smaller policies and single-camera Flux3, and LIBERO Plus for the others. VQ-BeT completes its codebook stage after one warmup update so measured training exercises BeT. These are profiling fixtures, not a comparison of trained policy quality. Default initialization may load pretrained backbones while leaving other weights randomly initialized. Use explicit checkpoint and dataset revisions for controlled comparisons. Model-specific assets, permissions, GPU capacity and dependency requirements still apply; setup failures and OOMs appear as failures with logs, not successful coverage.

## Publishing

Every workflow run produces a GitHub job summary and downloadable `model-profiling-report` artifact with `results.json`, a Markdown table, per-policy configurations, results and optional traces. A filtered manual run reports only the requested policies. Failed and missing results remain visible.

For a minimal rollout, use this summary and artifact first. For durable history, create a Hugging Face dataset and configure:

| Repository setting                  | Purpose                                                                |
| ----------------------------------- | ---------------------------------------------------------------------- |
| Variable `MODEL_PROFILING_REPO`     | Existing dataset ID, for example `lerobot/model-profiling-history`     |
| Secret `LEROBOT_HF_USER`            | Read access to required assets and write access to the results dataset |
| Secret `CI_SLACK_CHANNEL`           | Optional destination channel, shared with dependency CI notifications  |
| Secret `SLACK_CIFEEDBACK_BOT_TOKEN` | Existing CI Slack bot token, with access to that channel               |

The workflow uploads each run under `runs/<run-id>-<attempt>/` with [Hub folder uploads](https://huggingface.co/docs/huggingface_hub/guides/upload). It reuses the repository's existing Hugging Face Slack notification action to send the pass count and run link when a CI channel is configured. Neither destination is required to run the benchmark.

For a browsable public dashboard, add a small [Hugging Face Space](https://huggingface.co/docs/hub/spaces-overview) reading the results dataset. Keep the dataset as the durable record, the Space as a view, and Slack as the notification channel. Hosting a dashboard is separate from this minimal profiling implementation.
