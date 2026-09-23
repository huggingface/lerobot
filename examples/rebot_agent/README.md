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
