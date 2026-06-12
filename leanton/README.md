# LeAnton Patches — Inventory

> **Repo:** [anikita/lerobot](https://github.com/anikita/lerobot), branch `leanton`
> **SOP:** [PATCH_WORKFLOW.md](PATCH_WORKFLOW.md)
> **Issue tracker:** [UPSTREAM_ISSUES.md](UPSTREAM_ISSUES.md)
> **Upstream:** [huggingface/lerobot](https://github.com/huggingface/lerobot)
> **Base commit:** `49755a3d` (main, 2026-06)

---

## Usage (Colab / Any Machine)

```bash
git clone https://github.com/anikita/lerobot.git -b leanton
cd lerobot
pip install -e .
# Ready — all patches already applied in source
```

The `leanton/` directory ships alongside the code for reference (diffs, READMEs, issue tracker). No `git apply` step needed.

---

## Active Patches (Applied in Source)

| Patch | Diff | Target | Issue | What it does |
|:------|:-----|:-------|:------|:-------------|
| `action-smoothing-ema-clip` | [diff](action-smoothing-ema-clip/action-smoothing-ema-clip.diff) | `strategies/core.py` | — | EMA (alpha=0.3) + velocity clipping (2.0°/frame) on absolute goal positions. Auto-resets on >10° gaps (DAGGER boundaries). |
| `dagger-rtc-fresh-obs-on-resume` | [diff](dagger-rtc-fresh-obs-on-resume/dagger-rtc-fresh-obs-on-resume.diff) | `strategies/dagger.py` | [#3747](https://github.com/huggingface/lerobot/issues/3747) | Feeds fresh robot observation to RTC engine after reset, before resume. Prevents snap-back from stale `_obs_holder`. |
| `sync_read_retry-and-safe-disconnect` | [diff](sync_read_retry-and-safe-disconnect/sync_read_retry_and_safe_disconnect.diff) | `motors/motors_bus.py` | — | `sync_read` retries 3× (was 1). `disconnect()` catches `ConnectionError` so port always closes. |
| `no-stamp-repo-id` | [patch](no-stamp-repo-id/no_stamp_repo_id.patch) | `configs/dataset.py` | [#3722](https://github.com/huggingface/lerobot/issues/3722) | `--dataset.no_stamp true` to skip timestamp on `repo_id`. |
| `async-diffusion-crash` | [patch](async-diffusion-crash/async-diffusion-crash.patch) | `policies/diffusion/modeling_diffusion.py` | [#3445](https://github.com/huggingface/lerobot/issues/3445) | Fixes async inference crash when `predict_action_chunk` receives empty tensor. |
| `rollout-policy-revision` | [patch](rollout-policy-revision/rollout-policy-revision.patch) | `policies.py`, `rollout/configs.py`, `rollout/context.py` | [#3549](https://github.com/huggingface/lerobot/issues/3549) | Adds `--policy.revision` flag to `lerobot-rollout`. |
| `policy-server-logging` | [patch](policy-server-logging/policy-server-logging.patch) | `async_inference/policy_server.py` | — | Traceback logging in policy server (minor). |
| `mid-training-hub-push` | [diff](mid-training-hub-push/mid-training-hub-push.diff) | `scripts/lerobot_train.py`, `policies/pretrained.py`, `policies/factory.py` | — (planned) | Push intermediate checkpoints to Hub at each `save_freq`. V1: model weights only. Config flag: `push_checkpoints_to_hub`. |
