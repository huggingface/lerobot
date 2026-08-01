# Dataset artifact download path drops the resolved version

`lerobot_train.py` already runs dataset construction twice under `accelerate`:
once gated by `is_main_process` (which downloads/populates the shared root),
then once for every other rank after an `accelerator.wait_for_everyone()`
barrier — relying on every rank computing the _same_ `cfg.dataset.root` path
independently, since each rank parses its own copy of `cfg` from argv (no
shared Python memory across ranks).

We materialize a W&B dataset Artifact into a fixed path,
`<output_dir>/artifacts/dataset/`, derived only from `cfg.output_dir` —
_not_ `<output_dir>/artifacts/datasets/<resolved-name-and-version>/` as
originally proposed. The resolved version is only known after rank 0 actually
queries W&B, so embedding it in the path would mean non-main ranks can't
recompute the same path without also calling W&B, breaking the existing
single-download-then-barrier pattern (and adding rank-to-rank W&B chatter we
don't want). The resolved ref is still recorded as metadata (run config/
summary, model artifact metadata) — it's just not part of the directory name.
