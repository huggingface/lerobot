# Rollout results are their own Artifact type, not a dataset variant

A rollout produced by `lerobot-rollout --strategy episodic` is, on disk, an
ordinary `LeRobotDataset`: same `meta/info.json`, same `data/*.parquet`, same
`videos/`. Nothing in the directory distinguishes it from a human-recorded
training dataset — the only rollout-specific rule (`repo_id` must start with
`rollout_`) lives in the rollout config, never on disk.

We still upload it as `type="rollout"` rather than `type="dataset"`.

The type does not describe the schema, it describes the *lineage claim*. A
training dataset was recorded by a human and is an input to training. A rollout
dataset was produced by a specific model and is an output of evaluating it —
its run declares that model as an input via `use_artifact`, so the two are
connected in W&B's lineage graph. Collapsing both into `type="dataset"` would
put them in one undifferentiated pool where the only way to tell a policy's
output from its training input is to read the collection name and hope the
operator followed a naming convention.

Consequences we accept:

- Validation is shared, not forked. `rollout upload` runs the same
  `inspect_dataset_directory` as `dataset upload`; a rollout that isn't a
  readable LeRobot dataset is rejected by exactly the same code.
- `dataset download` will refuse a rollout artifact, because
  `download_artifact` enforces `expected_type`. That is the intended behavior:
  training must not silently consume a policy's own output as if it were
  ground truth. There is deliberately no `rollout download` yet — nothing
  consumes one. Add it when something does.
- A future `rollout` type that *does* diverge on disk (per-episode success
  flags, for instance) needs no migration of the type name.
