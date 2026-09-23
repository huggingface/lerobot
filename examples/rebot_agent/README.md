# Steerable ReBot policy with Astra planning

This experiment uses LeRobot main's datasets, language recipes, WALL-OSS-Flow,
processors, rollout strategies, and `/autosteer` controller. Astra chooses a steering
instruction from images and recent observation/command history. The VLA produces
all robot actions. There is no separate hybrid runtime or direct-action tool service.

See [the guide](../../docs/source/steerable_rebot.mdx) for the paper mapping, data
contract, annotation review, training, and physical evaluation. The reusable research
and deployment goal is [astra_goal.md](astra_goal.md).

For the physical comparison, use `physical_eval.py` and the guide's validation
milestone: 50 preplanned attempts, an append-only journal, separate operator/model
labels, and reports that retain failures and unknowns. It records evidence around
main's rollout; it does not start the robot or turn planner assessments into success labels.

[`evaluation_scenarios.draft.json`](evaluation_scenarios.draft.json) supplies a concrete
17-scenario starting draft: familiar targets, paraphrases, distractors, physical-arm
selection, ordered tasks, and a pastry brush and tea strainer as proposed novel targets. Its objects and
reset layouts still need an operator check on the actual table. The novel targets
are proposals, not verified inventory or novelty claims. Training annotations name
spoons and sponges; brush and strainer terms were absent from the text audit. Text
absence alone does not establish visual novelty. Confirm their absence from
the training demonstrations and retain that evidence before using those tags.
Photograph each finalized reset, keep the 90-second limit fixed across conditions,
and freeze the resulting plan before collection. The last cap/cloth scenario receives
the task-only/full pair; the other 16 receive all three conditions.

For this planner comparison, use the **same validated full-mixture checkpoint** for
all three conditions, changing only the permitted prompting styles and planner mode.
The semantic training baseline is a separate training control. Changing weights
between prompting conditions would confound the planner comparison. The draft does
not start any attempt or supply operator success labels; unavailable objects or
unverified resets must be resolved before it becomes the final physical test plan.

## Train on the science cluster

Connect with `sft ssh hpc-cluster-science-login-81-129`, obtain a single-node scheduler
allocation of **at most four H100s**, then install `uv sync --locked --extra training
--extra wallx`. This launcher runs inside that allocation; it does not allocate GPUs.

```bash
# Inspect configuration without loading a model.
uv run examples/rebot_agent/train_wall_oss_flow.py --gpus 4 --smoke --dry-run --output outputs/wall_preview
# Semantic baseline: 80% current subtask / 20% overall task.
uv run examples/rebot_agent/train_wall_oss_flow.py --gpus 4 --smoke --output outputs/wall_smoke
# Richer model: 80% uniformly sampled reviewed commands / 20% task.
uv run examples/rebot_agent/train_wall_oss_flow.py --gpus 4 --steering-manifest /path/to/reviewed/steering_manifest.json --output outputs/wall_steerable
```

Use a fresh output directory for each launch. Check finite losses, memory, held-out
loss, and checkpoint reload after the smoke test before launching the full run.
The smoke run evaluates 20 held-out samples, saves step 10, then starts a fresh
process through main's checkpoint resume path for one further update and validation.
It writes `smoke_completed.json` only if both processes succeed. This is a bounded
execution check; it does not replace full held-out validation or physical evaluation.
`training.json` pins the dataset and native base-model revision and configures WALL-OSS-Flow, using native
`policy.pretrained_name_or_path` initialization. The 14 ReBot dimensions are padded
and masked internally. The full run defaults to 20,000 steps and batch one per GPU;
these are starting settings that still require hardware validation.

Restoring a complete WALL-X policy checkpoint reads the pinned base repository's
configuration and processor assets without fetching its weights again. Non-strict
partial restores still load base weights to fill missing tensors; strict restores
reject incomplete checkpoints.

## Use Astra through the language runtime

Add these options to your existing, calibrated `lerobot-rollout` command, keeping
its robot ports/cameras and trained `--policy.path`:

```bash
--interactive=true --inference.type=sync --interpolation_multiplier=1 \
--planner.enabled=true --planner.model=gpt-6-astra \
--planner.camera_keys='["base","left_wrist","right_wrist"]' \
--planner.styles='["task","subtask"]' \
--planner.log_path=outputs/rebot_planner/decisions.jsonl \
--autosteer_interval_s=2
```

Set `OPENAI_API_KEY` on the robot host. Use an API model ID available to that account;
model access has not been tested here. Camera keys refer to processed robot
observations. After `/start`, the external planner keeps the VLA idle until you enter:

```text
/autosteer Put the white tape roll into the black bin.
```

Astra observes and emits a command, the VLA acts, and Astra observes again. Extend
`planner.styles` to `motion`, `point`, and `combination` only for a checkpoint
trained and validated on those styles. For coordinate commands, also set
`--planner.grounding_camera_keys='["base"]'` (or the other views actually trained
and validated for coordinates). The planner still observes every `camera_keys` view.
Source/destination target pairs are supported without enabling gripper paths.
Following the paper's final planner, leave
`trace` disabled by default; use it only in controlled, validated experiments.
`/subtask <text>` switches to human language
steering; `/reset` ends the segment; `/stop` closes the session. Planner failures,
uncertainty, or a reported completion hold action production; resume with an explicit
new `/autosteer` or `/subtask`. Completion is an assessment, not a verified success label.

The initial external-planner adapter uses synchronous inference: robot action
production waits while the API call runs. It reuses main's queue invalidation and
language switching. The interval is seconds of execution, not the paper's exact
20-step cadence. Measure latency and command compliance before tuning it.

## Store grounded geometry in native language annotations

After identification has finished, `extract_visual.py reparse-identify --output PATH`
can recover explicit name lists from failed response formats. For remaining failures,
`retry-identify --output PATH` makes one additional Molmo call per failed clip with a
stricter list prompt. It preserves the complete original response and its hash, skips
successful clips, and refuses to change identities after pointing or tracking.
Repeated runs do not retry the same failure indefinitely. Inspect unresolved errors
before pointing; recovered and retried candidates still require visual review.

When an image review establishes better object names, use
`extract_visual.py review-identify --output PATH --identification-review REVIEW.json`
after the extraction job has stopped and before pointing or tracking. The review
contains `parent_manifest_sha256`, an attributed `reviewer` (`kind`: `model` or
`human`, and `id`), and `corrections`. Each correction supplies `clip`,
`source_identification_sha256`, the first image's `frame_sha256`, an explicit list
of at most four distinct `objects`, and a visual `reason`. All corrections are
checked before any prediction changes. The complete previous responses remain in
the resulting identification records, including failed model retries. Corrected
names are attributed to their reviewer; they do not approve masks or training labels.
Native language exports bind detections to the identification hash and retain the
full identification history once per clip in `meta/grounding_provenance.json`.

For hollow or thin objects, a Molmo object-center point can land on background and
seed the wrong SAM2 mask. In a **separate extraction**, try
`extract_visual.py point --output PATH --point-target material` to request a point
on visible object material. The exact prompt, response, and target mode are stored;
resuming with a different mode refuses to reuse prior points. Inspect masks before
accepting candidates: a ReBot pilot improved tape-roll tracking with this prompt,
but a cable still produced a tabletop mask. Mask presence alone is not validation.

To correct a seed after inspecting its first frame, use
`extract_visual.py prepare-reviewed-points --parent ORIGINAL --seed-review review.json --output NEW`.
The review JSON contains `parent_manifest_sha256`, an attributed `reviewer`
(`kind`: `model` or `human`, and `id`), and a `corrections` list. Each correction
specifies `clip`, `object_id`, `name`, `source_point_sha256`, `frame_sha256`,
`point` in original-image `[x, y]` coordinates, and a visual `reason`.
Use `point: null` when the object cannot be localized. Frame and prediction hashes
must match the parent. The new extraction preserves original predictions and
reviewer attribution, then requires the normal `track` and `filter-objects` stages.
Correcting a seed does not accept the resulting track. Native exports retain this
provenance as `seed_point_source` and `seed_review`; model reviews are never relabeled
as human verification.

To recover an instruction-named object omitted by identification, use
`extract_visual.py prepare-required --parent ORIGINAL --output NEW --objects bin`.
Once every parent clip has a point result, add `--required-source points` to start
recovery before parent tracking finishes. This applies the same name-in-subtask
filter, snapshots the candidate evidence with hashes, and writes a separate
extraction. It does not alter the running parent or establish object visibility.
Run `point`, `track`, and `filter-objects` on the new extraction and review its masks
before accepting annotations.

```bash
uv run examples/rebot_agent/export_language_annotations.py \
  --dataset-root /path/to/source_dataset \
  --extractions outputs/rebot_visual outputs/rebot_required_objects \
  --output outputs/rebot_grounding_candidates
```

After fitting and extracting the ReBot gripper detector, add
`--grippers outputs/gripper_tracks` to the same export command. A completed detector
extraction writes `gripper_manifest.json`, binding prediction files to the exact visual
manifest, checkpoint weights, and configuration. Partial or mismatched extractions
cannot be exported. Use a fresh derived dataset when adding grippers to a previous
object-only export.

When combining original extractions with corrected versions, explicitly select the
corrected clips with `--clip-replacements replacements.json`. Supply both extraction
roots through `--extractions`. Otherwise their objects remain independent candidates.
The JSON file is a list of whole-clip replacements:

```json
[
  {
    "source": {
      "extraction_sha256": "ORIGINAL_MANIFEST_SHA256",
      "clip": "episode_007/span_003"
    },
    "target": {
      "extraction_sha256": "CORRECTED_MANIFEST_SHA256",
      "clip": "episode_007/span_003"
    },
    "reason": "Use the corrected pen track with its attributed point review"
  }
]
```

Both clips must cover exactly the same pinned source frames, timestamps, camera,
dimensions, and interval. Unknown references, duplicate sources, and replacement
chains are rejected before output creation; point each superseded version directly
to its final correction. Only the target clip contributes visual objects and traces.
The source manifest, track/filter hashes, and selection reason remain in provenance.
Independent gripper predictions are retained even when their visual clip is replaced.
Replacement selects evidence; it does not accept labels for training.

The exporter creates a derived LeRobot dataset with `language_events` in its data
Parquet files. It uses the existing camera-tagged `vqa` events for bounding boxes,
mask-centroid points, and first-frame Molmo pointing seeds, plus `trace` events for
observed object trajectories. JSON `content` retains original image dimensions,
object identity, half-open `xyxy` boxes, `xy` points, missing detections, interval,
review status, and hashes. One VQA pair per camera/frame keeps native recipe lookup
unambiguous. Source language, actions, states, episode IDs, and timestamps are preserved;
conflicting existing VQA/trace rows cause an error before export.

These exports contain **unreviewed candidates**, not accepted steering commands.
Object tracks are not gripper tracks. Each stored trajectory ends at its event frame,
retains missing samples, and contains no future points. A Molmo seed is recorded only
on the frame where it was inferred; later points are explicitly mask centroids.
`meta/grounding_provenance.json` records models, input hashes, and extraction roots.
Videos are symlinked to the local source dataset: copy the actual videos when moving
or publishing the derived dataset. No upload is performed. The original dataset and
running training jobs remain unchanged.

Gripper entries additionally carry `entity: "gripper"`, physical `arm`, detection
status, confidence, and `point_source: "detector_box_center"`. Their observed paths
remain distinct from object trajectories. Missing, ambiguous, or invalid predictions
retain null geometry in both VQA and trajectory samples; detection does not establish
verified visibility or arm identity. These predictions still require annotation review,
even when the detector was fitted on human-reviewed images.

## Compile pointing candidates from native annotations

`compile_point_features.py` connects the native geometry export to `prepare_steering.py`.
After reviewing which tracked identity is the picked object and which is its destination,
save a bindings JSON with `source`, `grounding_provenance_sha256`, `data_sha256` (every
relative `data/*.parquet` path and its SHA-256), and `segments`. Each segment contains:

```json
{
  "episode_index": 0,
  "start_frame": 0,
  "end_frame": 30,
  "subtask": "Put the cloth in the bin",
  "subtask_evidence": {
    "source": "timestamped source annotation and inspected video"
  },
  "camera": "observation.images.base",
  "image_size": [640, 480],
  "objects": [
    { "role": "pick", "object_id": "EXTRACTION_HASH:CLIP:1", "name": "cloth" },
    { "role": "place", "object_id": "EXTRACTION_HASH:CLIP:2", "name": "bin" }
  ],
  "role_review": {
    "verdict": "accepted",
    "reviewer": { "kind": "model", "id": "REVIEWER" },
    "notes": "Visible evidence identifying the picked object and destination"
  }
}
```

This is a schema illustration, not an accepted annotation. Copy exact identities and
names from the native VQA detections. The two identities can come from different
extractions, such as a corrected pick-object track and a destination recovery track,
after exporting them together into one derived dataset. Roles stay ordered pick then
place; the compiler does not infer them from names or observed gripper closure.

```bash
uv run examples/rebot_agent/compile_point_features.py \
  --dataset-root outputs/rebot_grounding_candidates \
  --bindings reviewed_object_roles.json --output outputs/point_features
uv run examples/rebot_agent/prepare_steering.py \
  --dataset-root outputs/rebot_grounding_candidates \
  --features outputs/point_features/features.candidates.json \
  --output outputs/point_command_review
```

The first step is read-only on the dataset and requires no model API. It writes
per-frame ordered points in original-image coordinates, with role review and native
file hashes retained as evidence. `coverage.json` identifies requested intervals with
missing masks, identities, or camera annotations. Those intervals remain unresolved;
the compiler neither interpolates points nor silently shortens their action horizons.
Subtask, point, and combined command candidates still require a separate visual
command review. This does not supply calibrated motion commands or gripper traces,
and it does not satisfy the full-mixture training requirements by itself.
