# LeRobot × W&B integration

Adds a W&B-backed remote lifecycle (dataset, model, rollout artifacts) on top of LeRobot's local-first recording/training/rollout pipeline. W&B is a durable store for finished artifacts, not a streaming filesystem the robot control loop touches.

## Language

**Dataset schema version**:
The `codebase_version` field inside a dataset's `meta/info.json` (e.g. `"v3.0"`), recording which on-disk layout the dataset was written in. Belongs to the dataset directory, independent of any installed package.
_Avoid_: "lerobot codebase version" (ambiguous — see LeRobot package version)

**LeRobot package version**:
`lerobot.__version__`, the installed pip package release running the current process. Distinct from Dataset schema version — a dataset written by an old package version can still carry the current schema version.
_Avoid_: "codebase version" alone, "lerobot version"

**Requested ref**:
The artifact reference string a caller passes in (e.g. `entity/project/name:raw`), which may point at a mutable alias.

**Resolved ref**:
The immutable `entity/project/name:vN` reference W&B returns after resolving a requested ref. Every upload/download in this integration must surface both the requested and resolved ref, never only one.

**Materialized dataset / materialized model**:
A W&B Artifact's contents after `download()`, sitting on local disk in the exact directory shape `LeRobotDataset` or a policy loader expects. "Materialize" implies the download already happened — no network call remains on the read path.

**Artifact collection**:
A named, versioned sequence of W&B Artifacts within a project (e.g. `so101-act-policy`, incrementing `:v0`, `:v1`, ...). Per-checkpoint collections (from the existing periodic `log_policy` upload) and the final-model collection (from `log_final_model`) are deliberately separate collections, not shared.

**Registry collection**:
A named collection inside W&B's unified Registry (`wandb-registry-model/<name>`), populated by linking an existing Artifact collection version into it via `run.link_artifact()`. Distinct from the legacy, now-unused W&B Model Registry (`run.link_model()`, hardcoded to a `model-registry` project) — this integration only uses the unified Registry.
_Avoid_: "model registry" (ambiguous between legacy and current)
