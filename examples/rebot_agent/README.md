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

When uncertain annotations must be skipped, add `--skip-uncovered` with the steering
manifest. Training and held-out evaluation then sample only reviewed frame intervals;
the 80/20 mixture applies within that subset. Original episode identities, observations,
action chunks, and temporal queries stay intact. Steering action targets outside each
command interval remain masked. `steering_coverage.json` records excluded frames and
the available styles separately for each split; sparse labels do not establish a
learned capability. Required styles are checked on training episodes, while held-out
evaluation reports the styles actually available there. Without this explicit option,
missing coverage remains an error.

For experiments with more subtask supervision, pass
`--style-weights examples/rebot_agent/steering_style_weights.json`. This optional
proposal assigns relative weights of 50 subtask, 15 motion, 10 point, 4 combination,
and 1 trace; the high-level task still has a separate 20% probability. At each
reviewed frame, weights are renormalized over the styles actually available there,
then alternatives within a style are sampled uniformly. Extra paraphrases therefore
do not increase a style's weight. The reported `expected_style_fraction` accounts
for coverage: these weights only yield 50/15/10/4/1 percent overall if every sampled
frame supports every style. Missing motion or trace coverage cannot be fixed by a
larger weight. Leaving the option unset preserves uniform command sampling.

This weighted proposal is an experiment, not the Steerable Policies paper's recipe.
[Appendix F](https://arxiv.org/html/2602.13193v3#A6) samples uniformly from the full
list of task, subtask, and generated commands at each frame. Our separate 20% task
allocation is also an adaptation. WALL-WM instead describes event-aligned temporal
caption levels and balances vision–language and action clusters; its
[Sections 4.3–4.4](https://arxiv.org/html/2606.01955v1#S4.SS3) do not specify a fixed
percentage for each caption level. Neither approach establishes an optimal ReBot mix.

This ReBot experiment uses autonomous annotation review; the operator does not need
to draw boxes or confirm labels. Attribute accepted reviews to the model and skip
uncertain labels. Measured gripper state can support opening/closing commands even
when the gripper is outside a wrist image, provided its state key, physical arm, and
opening sign are verified. Leave unsupported image coordinates missing; measured
gripper motion alone does not establish grasp success.

`offline_steering_eval.py` compares prompts on fixed held-out observations. Supply an
evaluation-only steering manifest containing episodes 90–99. Without `--checkpoint`,
it writes the deterministic evaluation panel; with a checkpoint, it predicts action
chunks from current images/state and each available prompt. Use identical manifest,
anchor count, and seed for both checkpoints:

```bash
uv run examples/rebot_agent/offline_steering_eval.py \
  --manifest /path/to/reviewed_heldout_manifest.json \
  --checkpoint /path/to/checkpoints/020000/pretrained_model \
  --dataset-root /path/to/source_dataset \
  --output /path/to/fresh_evaluation.json
```

The comparison verifies the checkpoint's saved episode split and uses identical
noise seeds and valid action horizons for every prompt at each anchor, including
the high-level task. Results include per-dimension action errors, normalized errors,
paired differences from task prompting, and annotation coverage. Missing styles
remain absent. These measurements describe agreement with recorded demonstrations;
they do not measure physical success or command compliance.
Each prediction retains its noise seed, current joint state, full predicted action
chunk, and demonstrated actions at the explicit valid chunk indices. This makes
individual joint/gripper errors inspectable without treating masked future targets
as evidence or confusing a predicted action with physical execution.

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
model access has not been tested here. Rollout rejects a missing or blank configured
API key before loading policy weights or connecting hardware. This local check does
not validate endpoint access or credentials; API failures still hold action production.
Camera keys refer to processed robot
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

When video review identifies an already tracked object that the name-in-subtask
filter excluded, use `extract_visual.py prepare-reviewed-objects --parent ORIGINAL
--object-review review.json --output NEW`. This also supports correcting a source
subtask that describes the wrong demonstrated object. The review JSON contains
`parent_manifest_sha256`, an attributed `reviewer` (`kind`: `model` or `human`,
and `id`), and a nonempty `selections` list. Each selection specifies `clip`,
`source_tracks_sha256`, the first source `frame_sha256`, selected `object_ids`,
a visual `reason`, and optionally corrected `subtask` text.

This stage copies the original frames, identities, points, masks, and tracks without
changing their bytes. It preserves the complete parent manifest and review, plus
the original subtask and artifact hashes in the new manifest. The explicit selection
survives subsequent `filter-objects` runs. Missing masks stay missing, object names
stay unchanged, and candidates still require command/geometry review before training.
The native annotation exporter includes this attribution in its grounding provenance;
it does not replace the source dataset's language or action columns.

To recover an instruction-named object omitted by identification, use
`extract_visual.py prepare-required --parent ORIGINAL --output NEW --objects bin`.
Once every parent clip has a point result, add `--required-source points` to start
recovery before parent tracking finishes. This applies the same name-in-subtask
filter, snapshots the candidate evidence with hashes, and writes a separate
extraction. It does not alter the running parent or establish object visibility.
Run `point`, `track`, and `filter-objects` on the new extraction and review its masks
before accepting annotations.

For parallel SAM2 tracking, use a shared JSON plan whose `manifest_sha256` matches
the exact `extraction.json` bytes and whose `shards` are nonempty lists of disjoint
clip paths. Include only unfinished clips when resuming. Each worker runs:

```bash
uv run examples/rebot_agent/extract_visual.py track --output outputs/rebot_visual \
  --tracking-plan outputs/tracking_plan.json --tracking-shard 0
```

Use a different zero-based shard index for each worker. The extractor rejects stale
plans, unknown clips, and overlap anywhere in the plan; it records runtime provenance
separately for each plan hash and shard. Already completed tracks are retained. Stop
any earlier unsharded worker before starting the shards, and run only one worker per
shard against an extraction. After all workers finish, run `filter-objects` once on
the complete extraction. Sharding does not change source images, object IDs, model
checkpoints, or the extraction manifest.

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
