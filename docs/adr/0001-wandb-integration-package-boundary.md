# W&B integration package boundary

`WandBLogger` (`src/lerobot/common/wandb_utils.py`) already exists and already
uploads periodic checkpoints to W&B as `model`-type Artifacts via
`log_policy()`. Rather than migrating it into the new
`src/lerobot/integrations/wandb_artifacts/` package, we leave it in place and
only grow it in-line (new `use_dataset_artifact()` and `log_final_model()`
methods, refactored to retain `self.run`). The new package holds only
genuinely new surface: `refs.py`, `store.py`, `inspect.py`, and the
`lerobot-wandb` sidecar CLI. `wandb_utils.py` imports from `store.py` to do
its uploads/downloads.

We picked this over consolidating everything into the new package because
`WandBLogger` is wired into the existing training loop (accelerate,
multi-rank barriers, checkpoint cadence) and moving it would be a much larger
diff for no functional gain — it's existing code being extended, not new
W&B-specific code that constraint 6 asks us to isolate.
