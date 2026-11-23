# SO-101 G05: one CoT/action stream with memory and grounding

This local trial starts from the demonstrated SO-101 G05 checkpoint, rather than
FineART. Worktree: `/home/maxime/github/lerobot-so101-g05`, branch
`feat/so101-g05-recipe`, based on the working G05 adapter at `e18e10636`.
No training, paid Job, commit or upload is started by preparing these files.

## Trial definition

- Initial weights: `maximellerbach/g05_so101_sim4k`, revision
  `b10e721b763a655e2df7188e86991502cb583d88`,
  `checkpoints/004000/pretrained_model`.
- Dataset: `maximellerbach/so101-fineart-mixed-v1`, revision
  `6cd89066f9768dbee8c868af4fa8f108de189089`. The name does not restrict its use to
  FineART: these are ordinary SO-101 observations/actions plus normalized annotations.
- Same deterministic 5% per-family holdout as the FineART pilot: 203 episodes held
  out, 3,797 training episodes. No automatic offline evaluation in this pilot.
- Keep the proven relative actions, joint frame transforms, front/empty/wrist camera
  slots, saved normalization, 32 predicted actions and 16 executed actions. Preserving
  processors prevents recipe fine-tuning from silently rebuilding control calibration
  from this different dataset. No stored bbox/keypoint coordinates change.
- Both discrete-action CE and continuous flow losses remain enabled, as in the
  working checkpoint. Inference still uses its flow action head. Use one flow sample
  per example during training, LR 2e-5, 500 warmup steps.
- Proposed target: 10k updates, batch 32, checkpoints every 5k. Batch/memory/throughput
  are **not profiled yet**; use the training GPU to profile before a paid full run.
  The local 16 GB GPU is for inference, not this full fine-tune.

`recipe.yaml` samples 80% structured CoT-plus-actions and 20% plain atomic
subtask-to-action examples. These are supervision weights, independent of the
existing dataset's task-family mix. There is no separate text-only planner branch:
every selected example still trains the action head.

The structured branch inputs the original goal and **previous** memory, plan and
issued subtask. Its targets use the current observed annotations. Memory preserves
initial object anchors and confirmed transfer IDs; retries do not imply success.
The current memory is never inserted into its own input prompt.

G05 already supports a text-valued `Subtask:` CoT field. This trial expands its
content, without introducing another tokenizer/template dialect:

```text
BBox: <native front-camera detections, when annotated>|
Subtask: <memory>...</memory><plan>...</plan><grasp_point>...</grasp_point><subtask>pick up the cube</subtask>|
Action: <EOV> ...
```

The existing native action expert attends to the generated CoT. Boxes use G05's
existing location-token formatter. Grasp points are sparse normalized XY JSON with
an explicit front camera and object label, carried in the structured text. Only
answers explicitly labelled `grasp point of ...` become grasp targets; other VQAs
are ignored. Empty point fields mean no supervised point on that frame. Point
predictions are auxiliary grounding, not a separate geometric grasp controller.
Wrist images still condition the model; this trial grounds boxes/points in front.

## Preparation and validation

Reuse the existing G05 environment (Transformers 5.5); the FineART environment has a
different Transformers pin. Always put this worktree's source first:

```bash
cd /home/maxime/github/lerobot-so101-g05
export PYTHONPATH="$PWD/src:$PWD"
export PYTHON=/home/maxime/github/lerobot-g05/.venv/bin/python

# Already prepared locally. Refuses to overwrite an existing output.
"$PYTHON" -m examples.training.so101_g05.prepare \
  --output=/home/maxime/github/robots/runs/so101_g05_full/initial_checkpoint

"$PYTHON" -m examples.training.so101_g05.validate \
  --root=/home/maxime/github/robots/datasets/so101-fineart-mixed-v1/prepared \
  --checkpoint=/home/maxime/github/robots/runs/so101_g05_full/initial_checkpoint
```

Preparation creates a separate config/sidecar directory and hard-links immutable
weight files. It does not modify either original checkpoint or dataset. Treat those
hard-linked weights as read-only.

Completed full validation: 4,000 episodes; 247,316 structured target checks,
184,957 box targets and 5,092 front grasp-point targets. Maximum command input:
237 tokens; maximum generated CoT including the action delimiter: 314 tokens.
Both limits are 768, checked with the actual G05 tokenizer before loading weights.
The report is `/home/maxime/github/robots/runs/so101_g05_full/validation.json`.
These are coverage counts over boundary/VQA checks, not an exhaustive count of
training examples drawn by the weighted sampler.

## Training command, when the training GPU is ready

```bash
cd /home/maxime/github/lerobot-so101-g05
bash examples/training/so101_g05/train.sh
```

The launcher repeats full token/target validation and applies the family holdout
before invoking `lerobot-train`. Start with a 1k-step pilot and inspect grasping,
false completion, memory updates and latency before committing to 10k:

```bash
OUTPUT_DIR=outputs/train/so101-g05-pilot \
  bash examples/training/so101_g05/train.sh --steps=1000 --save_freq=1000
```

No upload or experiment tracking is enabled by this local launcher. This older G05
adapter branch has W&B integration, not the newer Trackio integration on main. A
future HF Job launch needs its tracking and measured batch size configured. Public
model destination is configured as `maximellerbach/so101-g05-full-v1`, but publishing
remains disabled until requested. `uv.lock` is untouched.

## Inference contract

`memory_scratchpad=true` now enables feedback for native **G05 System 2** action
predictions: each new chunk receives the previous predicted memory/plan/subtask,
then generates updated CoT and conditioned actions. State belongs to one policy
instance and one goal. Reset/new goal clears it. Invalid structured output raises
before an action chunk is released; it does not overwrite valid previous memory.
The shared synchronous rollout engine discards actions accompanying `done` and
ends the interactive run in a hold. Direct policy callers must also honor
`policy.completed_task`. RTC/remote/batched scratchpad deployment is not included.

Use the usual interactive `/start` with the complete task. Do not add `/autosteer`:
G05 already generates its plan and subtask in the action inference itself. System 1
can use the 20% atomic branch for manual subtask control. No new PlannerSession or
external planning service is required.

The original 4k-step G05 checkpoint does not know this expanded output format yet.
Only enable this configuration on the new training initialization/outputs. Longer
CoT costs generation time each chunk; the pilot must measure whether that latency
and the change in text conditioning preserve the baseline's motion quality. The
regression tests and annotation check do not establish trained-policy success.

## Launched HF pilot

Replacement Job `6aeb09680e0338d1bb7faa2c` runs on one H200, with a 24-hour platform fallback.
It downloads pinned inputs, repeats validation, profiles native batch 32 (falling
back to smaller microbatches with accumulation if required), then trains **1,000
optimizer updates at effective batch 32**. The pilot ends with a public checkpoint
in `maximellerbach/so101-g05-full-v1` for simulation review before any continuation.
The profiling updates are disposable and are not included in the pilot weights.

The job-side `job_train.py` adapts the older logger interface to **Trackio**, so this
HF run does not require a W&B key. Project `so101-g05` is on the existing private
`maximellerbach/so101-fineart-tracking` Space. Run names include the HF Job ID; when
accumulation is used, `train/optimizer_step` exposes the applied update count.

`hf_job.py` owns validation, profiling, launch and checkpoint output. Immutable
launch artifacts are stored locally at
`/home/maxime/github/robots/runs/so101_g05_hf/launch_20261104`: source snapshot,
manifest, exact run specification, private bootstrap and sanitized Job metadata.
No source archive was published; the snapshot was passed in the private HF Job
command. No commits or pushes were made, and `uv.lock` is unchanged.

The initial Job `6aeb06820e0338d1bb7fa77c` passed full validation but failed on its
first backward pass: FLA 0.5.2 rejects Hopper kernels under Triton 3.4–3.7.0. The
replacement pins runtime Triton 3.7.1 after installing the locked environment and
checks that compatibility before downloading data. `uv.lock` itself is unchanged.
It reuses the completed full validation only after matching input revisions and
source hashes. Restart artifacts are under
`/home/maxime/github/robots/runs/so101_g05_hf/restart_triton371`.

The replacement passed GPU backward/optimizer updates. Native batch 32 exhausted
H200 memory on a later batch, so the automatic profile selected microbatch 16 with
2-step accumulation (effective batch 32), peaking at 81.04 GiB in the profile. The
pilot therefore runs 2,000 micro-steps / 1,000 optimizer updates. Trackio run:
`so101-g05-full-pilot-b32-6aeb09680e0338d1bb7faa2c`; its first logged pilot metrics
at micro-step 50 / optimizer update 25 had finite loss and gradient norm.

## Local pilot review

The completed pilot is downloaded at revision
`1cb750ba2de4d91b7fd99d4f7d6799aa9ffad1ea` to
`/home/maxime/github/robots/models/so101-g05-full-v1-pilot-1000`.

```bash
cd /home/maxime/github/lerobot-so101-g05
bash examples/training/so101_g05/simulate.sh
```

Open http://127.0.0.1:8766 and type `/start` in the terminal. Scene 23 and the goal
`Put the orange cube in the gray box.` are preselected. The camera monitor includes
predicted memory/plan/subtask and latest chunk-generation time after observation
updates. Use `/reset` to load a randomly selected saved dataset scene, restore its
matching goal and clear policy memory, then `/start` to run it. Scenes are sampled
without replacement, excluding the current scene. Use `/subtask` to change the
goal and `/stop` to end. This review launcher executes 32 actions per prediction
(1.07 seconds of simulated motion at 30 Hz). An initial,
non-executed prediction parsed successfully and took 4.98 seconds locally; this is
a loading/format smoke check, not a task-success result. Its record is at
`/home/maxime/github/robots/runs/so101_g05_full/pilot_first_prediction.json`.
