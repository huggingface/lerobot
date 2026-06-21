# Upstream Issues — LeRobot

> **Repo:** [huggingface/lerobot](https://github.com/huggingface/lerobot)
> **Author filter:** [anikita issues](https://github.com/huggingface/lerobot/issues?q=is%3Aissue%20author%3Aanikita)
> **Check status live:** [Open](https://github.com/huggingface/lerobot/issues?q=is%3Aissue%20state%3Aopen%20author%3Aanikita) | [Closed](https://github.com/huggingface/lerobot/issues?q=is%3Aissue%20state%3Aclosed%20author%3Aanikita) | [All](https://github.com/huggingface/lerobot/issues?q=is%3Aissue%20author%3Aanikita)

## Open

| Issue | Title | Opened | Labels | Local Patch |
|:------|:------|:-------|:-------|:------------|
| [#3445](https://github.com/huggingface/lerobot/issues/3445) | Async inference crashes with diffusion policy | 2026-04-23 | bug, dataset, evaluation, policies | `async-diffusion-crash` |
| [#3549](https://github.com/huggingface/lerobot/issues/3549) | Missing `--policy.revision` flag in rollout | 2026-05-10 | configuration, policies, training | `rollout-policy-revision` |
| [#3722](https://github.com/huggingface/lerobot/issues/3722) | No way to opt out of `stamp_repo_id` timestamp on rollout datasets | 2026-06-05 | bug, dataset, configuration | `no-stamp-repo-id` |
| [#3747](https://github.com/huggingface/lerobot/issues/3747) | DAGGER resume: RTC engine uses stale observation after correction, causing snap-back | 2026-06-09 | bug, policies | `dagger-rtc-fresh-obs-on-resume` |
| [#3793](https://github.com/huggingface/lerobot/issues/3793) | Push intermediate training checkpoints to Hub | 2026-06-13 | enhancement, training | `mid-training-hub-push` |
| [#3794](https://github.com/huggingface/lerobot/issues/3794) | Post-hoc EMA smoothing and velocity clipping for rollout actions | 2026-06-13 | enhancement, policies | `action-smoothing-ema-clip` |
| [#3845](https://github.com/huggingface/lerobot/issues/3845) | pi0_fast: make_policy retains stale input_features from base model, breaking custom datasets | 2026-06-21 | bug, policies | `pi0fast-input-features-from-dataset` |
| [#3846](https://github.com/huggingface/lerobot/issues/3846) | pi0_fast: --policy.action_tokenizer_name has no effect | 2026-06-21 | bug, policies, training | `pi0fast-action-tokenizer-override` |
| [#3847](https://github.com/huggingface/lerobot/issues/3847) | pi0fast.mdx: wrong model ID in training/eval code blocks | 2026-06-21 | bug, documentation | `pi0fast-docs-wrong-model-id` |

## Closed (Merged / Fixed Upstream)

| Issue | Title | Closed | Labels | Upstream Fix |
|:------|:------|:-------|:-------|:-------------|
| [#3458](https://github.com/huggingface/lerobot/issues/3458) | PEFT LoRA resume fails: PosixPath rejected by PeftConfig | 2026-05-04 | bug, configuration, policies, training | `fdbfc015` — fix(peft): fix LoRA resume from Hub |
| [#3459](https://github.com/huggingface/lerobot/issues/3459) | PEFT resume silently discards loaded adapter (double wrap) | 2026-05-04 | bug, configuration, dataset, policies, training | `fdbfc015` — same commit, same root cause |
| [#3551](https://github.com/huggingface/lerobot/issues/3551) | PeftConfig missing `lora_alpha` field — defaults to 8, dampens high-rank LoRA | 2026-05-13 | bug, configuration, policies, training | `9db9c35c` — fix(config): add lora_alpha to PeftConfig |
| [#3723](https://github.com/huggingface/lerobot/issues/3723) | SO-101 DAGGER: bumpless transfer on pause (sync leader to follower) | 2026-06-09 | enhancement, policies | [#3506](https://github.com/huggingface/lerobot/pull/3506) — feat(dagger): adding smooth handover |
