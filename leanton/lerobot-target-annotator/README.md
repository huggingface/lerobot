# lerobot-target-annotator

**Real-time YOLO-World target annotation for [LeRobot](https://github.com/huggingface/lerobot) data collection.**

Overlays a stabilized bounding box on the overview camera during teleoperation and streams the annotated frame directly into `lerobot-record` as a native camera channel — no LeRobot modifications required.

---

## The Problem

Training a robot arm to pick a *specific* object of choice in a cluttered environment — not just any object — requires the training data to carry a visual signal identifying the target. Without it, the policy learns at best to pick whatever is easiest to grasp, not whatever was intended. 

More fundamentally, a cluttered scene with multiple objects creates a **multi-modal action space**: different valid demonstrations from the same starting observation lead to different trajectories (episode 1 picks object A, episode 2 picks object B). A policy trained without a target signal must somehow resolve which mode to execute from an identical observation — it cannot, and either averages across modes (producing indecisive, between-objects motions) or collapses to a single dominant mode regardless of intent. 

A bounding box eliminates the ambiguity: the same cluttered table with a box around object A becomes a distinct observation from the same table with a box around object B, and each maps to a single consistent trajectory.

The standard LeRobot recording pipeline records camera frames and joint states, but has no mechanism for annotating which object the arm should have been reaching for. This annotation gap is fine for single-object demos but breaks the moment you want a policy that can be directed to a particular object in a cluttered scene.

---

## The Solution

This tool implements **visual goal conditioning** at data collection time:

1. A YOLO-World open-vocabulary detector runs on the overview camera and locks onto the largest object in the scene (or the one inside a defined pick zone).
2. A **stabilization buffer** (IoU-gated rolling median + hysteresis) commits to a single target and holds it through brief occlusions — the annotation is robust to flickering during teleoperation.
3. The annotated frame (clean overview + bounding box rectangle) is published over ZMQ, which LeRobot reads as a **native `ZMQCamera`** source alongside the physical cameras.
4. The operator sees a full debug HUD (stability bar, zones, fps) in a separate window that is never recorded.

The result: every recorded episode contains `observation.images.annotated` — the overview frame with a consistent bounding box marking the target — alongside the standard `observation.images.front` (wrist camera) and `observation.state`. A VLA policy trained on this data learns to treat the bounding box as a spatial prompt: *reach for whatever is inside the rectangle*.

---

## Why This Architecture

### Decoupling semantic intent from spatial execution

The core insight is that a robot arm policy does not need to understand *why* an object was selected — it only needs to execute a reach-and-grasp toward a visual marker. Semantic reasoning (what to pick, based on language or context) is handled by a separate high-level module.

During data collection, YOLO plays the role of the high-level orchestrator. At deployment, an LLM or VLM can replace it. Because both draw the same style of bounding box on the same overview frame, the arm policy cannot distinguish them — it learned to respond to the visual cue, not the annotator.

```
Data collection:   overview_frame (YOLO bbox) → arm policy → grasp
Deployment:        overview_frame (LLM bbox)  → arm policy → grasp
```

The arm policy is identical in both cases. Only the annotator changes.

### Why not post-hoc annotation?

[`lerobot-annotate`](https://github.com/huggingface/lerobot-annotate) is a human-driven web UI for adding *language subtask labels* to recorded episodes — a different problem entirely. It has no bounding boxes and no automatic detection; a human watches the video and types a description for each segment.

A hypothetical automated post-hoc bbox pipeline would face a harder problem: it would need to process the full video after recording, identify which object moved by comparing the end-state to the start-state, and then *infer backwards* which object the arm was about to grasp before the motion began — with no ground truth for intent. Occlusion by the arm during the grasp makes this inference particularly error-prone.

Live annotation sidesteps all of this. A simple rule (e.g. largest object in the scene, or largest object inside a drawn pick zone) selects the target *before* teleoperation begins, the operator sees the box and acts on it, and intention is unambiguous by construction. There is no post-processing step and no gap between what was annotated and what was reached for.

### Why visual prompt (burned pixels), not a separate coordinate channel?

VLA policies built on VLM backbones (π₀, SmolVLA) already understand rendered spatial annotations from pretraining — bounding boxes, arrows, and overlaid labels appear throughout their training data. Burning the bbox into the image exploits this existing capability without any architectural change. The policy receives one tensor and learns to read the box as a directive.

---

## Repeated-Experiment Workflow (State Persistence)

The script auto-saves its configuration on quit and on every meaningful state change (freeze, zone draw). On the next launch it restores the camera index, class list, pick/exclusion zones, and last frozen bounding box — so you can restart a recording session without reconfiguring anything.

- **State file:** `annotate_stream_state.json` (created automatically alongside the script)
- **Skip restore:** pass `--fresh` to start from defaults
- **What's saved:** camera, classes, `pick_zone`, `excl_zone`, frozen bbox + label
- **On restore:** the frozen bbox is pre-loaded and the stability bar starts green — ready to record immediately

---

## Camera Index Convention

This repo assumes a two-camera setup common in LeRobot recording workflows:

| Index | Camera | Role |
|-------|--------|------|
| 0 | Gripper / wrist camera | Physical OpenCV feed → `front` in lerobot-record |
| 1 | Logitech C920 (overview) | Annotated by this tool → ZMQ → `annotated` in lerobot-record |

Run `python probe_cameras.py` to identify your cameras before recording.

---

## Stabilization Design

Raw YOLO detections fluctuate frame to frame — bounding box area and rank vary with lighting and pose. Without stabilization, the annotation jumps between objects during teleoperation.

Three mechanisms prevent this:

| Mechanism | What it does |
|---|---|
| **IoU-gated rolling median** | Only accumulates frames where the new detection overlaps the current median (same object). Median is spatially stable. |
| **Challenger buffer** | A competing detection must accumulate `CHALLENGER_FILL × CHALLENGER_SIZE` consecutive self-consistent frames before displacing the current target. Single-frame anomalies (arm sweeps, shadows) can never win. |
| **Hysteresis** | Once stable, the target is held for `HYSTERESIS_SECS` after the last valid detection. Brief occlusion by the arm does not reset the annotation. |

The bbox is only shown (and published to ZMQ) once the buffer is committed (green). During accumulation (purple), the stream publishes a clean frame — the operator waits for the green box before beginning teleoperation.

---

## Installation

```bash
conda activate lerobot   # requires an existing LeRobot environment
pip install ultralytics pyzmq
```

No modifications to LeRobot are required. `ZMQCamera` is a first-class camera type in LeRobot and is already supported by `lerobot-record`.

---

## Quick Start

```bash
# Pre-flight — identify your cameras
python probe_cameras.py

# Terminal 1 — start the annotation stream (ZMQ binds on port 5555)
conda activate lerobot
cd lerobot-target-annotator
python annotate_stream.py --camera 1 --device auto

# Terminal 2 — record with lerobot (see examples/record_command.sh)
conda activate lerobot
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem<FOLLOWER_ID> \
  --robot.id=<ROBOT_ID> \
  --robot.cameras='{
    "front":     {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"},
    "annotated": {"type": "zmq", "server_address": "localhost", "port": 5555, "camera_name": "annotated", "width": 640, "height": 480, "fps": 30}
  }' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem<LEADER_ID> \
  --teleop.id=<TELEOP_ID> \
  --display_data=true \
  --dataset.repo_id=<HF_USERNAME>/<DATASET_NAME> \
  --dataset.num_episodes=50 \
  --dataset.single_task="pick the object and place it in the basket" \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=10
```

**Start Terminal 1 before Terminal 2.** LeRobot's `ZMQCamera` blocks on warmup until the publisher is available.

On subsequent runs state is auto-restored — no need to re-draw zones or re-freeze. Pass `--fresh` to start from defaults.

---

## Controls

| Key | Action |
|---|---|
| `F` | Freeze / unfreeze — locks the current stable bbox, stops inference |
| `Z` | Draw pick zone (detect only inside this region) / clear |
| `X` | Draw exclusion zone (ignore detections inside) / clear |
| `A` | Toggle show-all secondary detections in HUD |
| `R` | Toggle raw / annotated view |
| `S` | Save HUD snapshot to `snapshots/` |
| `C` | Cycle to next camera |
| `Q` | Quit |

**Tip:** press `F` to freeze the bbox before starting teleoperation if the target is near other objects. Frozen bbox is published to ZMQ identically — LeRobot records no difference.

---

## CLI Options

```
--camera N          Camera index (default: auto-detect, skips built-in 0)
--device            mps | cpu | cuda  (default: auto)
--classes "a,b,c"   YOLO-World class list (default: "toy,small object,cloth toy")
--zmq-port N        ZMQ PUB port (default: 5555)
--zmq-name NAME     Camera name — must match lerobot camera_name (default: annotated)
--no-zmq            Disable ZMQ, run HUD only
--fresh             Skip loading saved state, start from defaults
--show-all          Show all detections in HUD
--list              Print available camera indices and exit
```

---

## Tuning

All parameters are gathered in the `CONFIG` block at the top of `annotate_stream.py`:

| Parameter | Default | Effect |
|---|---|---|
| `BUFFER_SIZE` | 15 | Rolling window length — longer = smoother, slower to commit |
| `STABLE_FILL` | 0.8 | Fill fraction required before GREEN |
| `IOU_GATE` | 0.4 | Minimum IoU to accept a detection as the same object |
| `HYSTERESIS_SECS` | 1.0 | Seconds to hold a stable target through occlusion |
| `CHALLENGER_SIZE` | 8 | Frames a competing detection must sustain to win |
| `CHALLENGER_FILL` | 0.75 | Consistency fraction required to promote challenger |
| `CONF` | 0.05 | YOLO detection confidence threshold |
| `ZMQ_JPEG_QUALITY` | 85 | Published frame quality (0–100) |
| `DEFAULT_CLASSES` | `["toy","small object","cloth toy"]` | Use concrete nouns — abstract words score ~0 in CLIP |

---

## ZMQ Wire Format

The script publishes on `tcp://*:5555` in LeRobot's native `ZMQCamera` protocol:

```json
{
  "timestamps": {"annotated": 1234567890.123},
  "images":     {"annotated": "<base64-encoded JPEG, RGB channel order>"}
}
```

Frames are encoded as RGB (not OpenCV's default BGR) to match the color space expectation of LeRobot's downstream processing.

---

## Tested On

- macOS (Apple Silicon M3), `device=mps`
- LeRobot main branch (post v0.5)
- SO-101 follower/leader arms
- `yolov8s-worldv2.pt` (auto-downloaded by ultralytics on first run)

---

## Related Work

- [`lerobot-annotate`](https://github.com/huggingface/lerobot-annotate) — official HuggingFace tool for post-hoc language annotation of recorded episodes; complementary, not competing
- [`any4lerobot`](https://github.com/Tavish9/any4lerobot) — dataset format conversion utilities for LeRobot
- [CLIPort](https://cliport.github.io/) — academic precedent for language-conditioned spatial goal conditioning in manipulation
- [RoboPoint](https://robo-point.github.io/) — VLM-generated point annotations as robot manipulation goals

---

## License

MIT
