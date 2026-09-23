# Goal: train a highly steerable ReBot VLA and use Astra as its high-level planner

Build and evaluate a real-world pick-and-place system following Steerable Policies:
https://steerable-policies.github.io/ and
https://github.com/steerable-policies/steerable-policies-bridge
(reference revision b95286e7823e1f05e490a96ae98f7a3e3ac396f8).

Use LeRobot main's existing language runtime and training stack wherever possible.
Work only on `codex/rebot-physical-agent-loop`. Do not create, reopen, or merge pull
requests. The earlier draft PR was closed at the user's request; preserve the branch.
The VLA is the sole robot-action controller. Astra receives current images, the overall
task, and a bounded history of observations and issued commands; it chooses the next
steering instruction and evaluates progress. Do not build a hybrid action arbitration
loop, direct IK/joint/gripper action tools, or an intervention-based DAgger system.
Language commands such as “move the left gripper upward” are VLA inputs, never direct
motor commands. Astra is primarily the high-level planner and also reviews annotations
and evaluation evidence. Its success judgments remain distinct from operator labels.

## Fixed context

- Robot: bimanual ReBot B601, `bi_rebot_b601_follower`, diverse picking into a bin.
- Robot host: `madeleine`. No separate joint calibration was supplied. Use Seeed's
  official B601-DM URDF and verify the recording zero pose and mounting convention
  before promoting nominal FK trajectories to training labels.
- Dataset: `pepijn223/rebot_diverse_picking_100_annotated`, revision
  `93c97807c46535745d0587d4296416bf2d4aa80d`, 100 episodes, 30 Hz, 14 joint/gripper
  dimensions, base/left-wrist/right-wrist cameras. Subtask language is timestamped.
- Selected policy: WALL-OSS-Flow, native LeRobot type `wall_x`, base
  `x-square-robot/wall-oss-flow`. No trained ReBot checkpoint was available initially.
- Training: science cluster via `sft ssh hpc-cluster-science-login-81-129`, at most
  four H100 GPUs through its scheduler. Never train on the login node. Reuse existing
  jobs/checkpoints; inspect current scheduler state before launching or resuming work.
  A failed observation request does not mean the job stopped.
- Keep the last ten source episodes held out, with stable episode identities.
- Start with 80% semantic subtasks / 20% overall tasks as a control. The target model
  uses 80% diverse grounded steering commands / 20% overall tasks. This ratio and
  WALL-OSS/ReBot are our adaptations, not an exact reproduction of the paper.

## Follow the paper's data pipeline

Read Sections IV–V and Appendices A–B of https://arxiv.org/html/2602.13193v3.
The Bridge code trains ordinary behavioral cloning, replacing the instruction with
a randomly selected equivalent command for the demonstrated interval. Retain the
same observations and action targets; resample alternatives on repeated training
visits. The released Bridge repository consumes precomputed annotations and does not
supply a complete ReBot feature extractor.

First extract grounded features. Follow the paper's Molmo object identification,
SAM2 temporal mask tracking, and DETR gripper localization as closely as practical.
Preserve object identities, per-arm gripper identities, timestamps, camera identity,
visibility, and track quality. Record extractor checkpoints and parameters. Inspect
overlays on a small diverse sample before expanding to the full dataset. Missing
tracks or occluded grippers must remain missing, not be hallucinated by the reviewer.

For Cartesian motion commands, recover the actual ReBot URDF, calibrated joint-name
mapping, signs, zero offsets, and measured units. Use main's RobotKinematics forward
kinematics on measured observation.state, separately per arm. Document the base/tool
frame and axis-to-language convention. Segment reversals and gripper events before
assigning directions. FK does not identify objects or prove grasp/placement success.
Projection into images requires calibrated camera intrinsics/extrinsics, including
moving wrist-camera transforms; otherwise use visual tracks for points and traces.

Next decompose into semantic subtasks, preserving the existing annotations where
supported by the video. Compose equivalent subtask, atomic motion/gripper, pointing,
gripper-trace, and combined commands from verified features. Pixel commands use named
camera views and original-image coordinates consistently at training and inference.
Do not independently crop/flip images without transforming every coordinate label.
The checked-in compiler supplies deterministic grounded variants; richer VLM wording
must keep verified geometry and semantic meaning unchanged.

Have Astra review command/feature alignment against timestamped video and report
accepted, rejected, or uncertain, with concise visual evidence. Audit a human sample;
Astra is not ground truth. Store bounding boxes, pointing coordinates, and traces in native LeRobot language
annotation columns (camera-tagged `vqa`/`trace` events with JSON content); keep source
data intact and distinguish unreviewed evidence from accepted commands. Raw extractor
sidecars support auditing but do not replace the dataset annotations.
Save provenance, source revision, intervals, original
features, review responses, coverage, and rejection counts. Preserve source data.
Close annotation gaps before training the full mixture; never silently substitute
generic commands for missing reviewed styles. Offline future frames may help labels,
but runtime planning and policy observations must contain only currently available data.

## Train, connect, and measure

Use the checked-in training launcher and main's recipes/processors. Run a short
forward/backward, validation, checkpoint-save/reload smoke test on the GPU allocation
before the full fine-tune. Confirm camera mapping, 14-dimensional actions, padding
masks, finite losses, realized command mixture, and fixed held-out episode identities.
Track the exact base model, code, data and annotation hashes, configuration, and output.

Follow Appendix B's final off-the-shelf planner configuration: enable pointing and
validated semantic/motion styles, but exclude gripper traces by default. Trace labels
can still support VLA training and controlled steering tests.

Deploy the checkpoint with main's `lerobot-rollout --interactive=true --inference.type=sync`
and the external planner adapter. Astra selects among styles this checkpoint has
actually learned; visual history helps it change abstraction when progress stalls.
Verify camera views against the dataset and confirm physical wrist identities before
rollout. Device indices and USB serial names can collide or change; use verified
physical device paths and inspect current images. Keep the existing robot environment
intact, check storage before transferring weights, and validate imports and checkpoint
inference in the deployment environment before connecting the arms.
Preserve human stop/reset and language overrides. Keep API calls bounded. A planning
failure, uncertainty, or completion assessment should hold action production pending
an explicit next instruction. Do not confuse an issued command with observed execution.

Collect 50 real-world evaluation attempts including failures and unknown outcomes.
Compare task-only prompting, semantic-subtask planning, and full multi-style Astra
planning on matched objects, initial states, and task instructions. Measure success,
command compliance, direction/arm/point accuracy, recovery through language, latency,
and command-style choice. Include paraphrases, distractors, novel objects, and multi-step
tasks. Record Astra assessments separately from operator-verified outcomes. Improve
annotations, prompt, policy, and code from evidence; additional human demonstrations
may support later behavioral cloning. Do not claim improved success from offline loss.

Continue useful independent work when a dependency is missing. Report exact blockers:
cluster access, authoritative URDF/calibration, extracted visual features, credentials,
or robot configuration. Never mark the goal achieved solely because code/configuration
exists. Deliver reviewed annotations, trained/reloadable checkpoints, the working Astra
planner setup, and measured physical results with remaining limitations explicit.
