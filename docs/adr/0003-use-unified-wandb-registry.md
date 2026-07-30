# Use the unified W&B Registry, not the legacy Model Registry

wandb (0.24–0.28) ships two different ways to publish a model to a registry:
`Run.link_model(path, registered_model_name)`, a convenience call whose
source hardcodes `project = "model-registry"` (W&B's legacy, sunsetting Model
Registry); and `Run.link_artifact(artifact, target_path="wandb-registry-model/<name>")`,
which links an already-logged artifact into W&B's current, unified Registry.

`log_final_model()` uses the latter: `store.upload_directory()` does the base
`run.log_artifact()` with `model_aliases` applied directly on the project-level
collection (so `entity/project/name:candidate` resolves as the CLI examples
expect), then a separate `run.link_artifact(artifact, target_path=f"wandb-registry-model/{registered_model_name}", aliases=model_aliases)`
call links it into the named Registry collection. We do not touch
`link_model` / the legacy Model Registry at all in this integration.

Consequence: `--wandb.mode=offline` combined with `registered_model_name` must
be rejected at config-validation time — `link_artifact` raises
`NotImplementedError` outright when the run is offline, and we'd rather fail
before spending training compute than after.
