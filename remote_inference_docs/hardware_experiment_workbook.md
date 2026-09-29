# Remote inference: five practical checks

**One checkpoint, one topology, one run per case.** Start with tests 1–3 and send the results before doing 4–5. No parameter sweeps, repeated trials or other models for now.

Checkpoint: `maximellerbach/omx_pickandplace_act` — pick one cube and place it in the blue square.
GPU server: **172.18.131.152**. Robot client: **172.18.133.214**.

These commands are for those two Linux machines, not the computer where this document was prepared. **Results received 2026-09-29:** test 1 worked with residual bumps; tests 2–3 performed significantly worse. Test 2 was run twice. Results for tests 4–5 have not been supplied. See the review below before running more motion comparisons.

## Setup

Use the updated implementation on both machines. Keep the ACT hardware arrangement from your previous successful pick-and-place run: follower ACM1, wrist camera 0, top camera 2. Restore roughly the same cube position between motion tests while the robot is stopped.

### On the server — 172.18.131.152

Run this once; keep it running between tests unless instructed to stop it:

```bash
cd ~/Documents/lerobot
uv run --no-sync lerobot-policy-server \
  --config_path=examples/remote_inference/omx_act_lan.yaml \
  --model.repo_or_path=maximellerbach/omx_pickandplace_act \
  --execution.idle_timeout_s=30 \
  --execution.blendable_components='[shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos]' \
  2>&1 | tee -a server.log
```

Wait for readiness. We reuse the ACT preset with the checkpoint override; the deployment name remains `omx-act`. Keep the checkpoint's saved action settings. The five arm joints may be blended; the gripper is excluded. The server/client contract checks that the five-step blend window fits the checkpoint's execution length.

### On the robot client — 172.18.133.214

In a Bash terminal, paste this once. It stores the common command arguments; **it does not start the robot**. Each test below adds only the setting being tested.

```bash
cd ~/Documents/lerobot
ACT_CLIENT=(
  uv run --no-sync lerobot-rollout
  --strategy.type=base
  --inference.type=remote
  --inference.endpoint=tcp/172.18.131.152:7447
  --inference.deployment=omx-act
  --inference.mode=chunk
  --inference.semantics=omx-normalized-joints-gripper-percent-v1
  --inference.hold_mode=position
  --inference.max_observation_age_s=3
  --inference.refill_seconds=0.1
  --robot.type=omx_follower
  --robot.port=/dev/ttyACM1
  --robot.id=omx_follower
  --robot.use_degrees=false
  '--robot.cameras={wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}, top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}'
  '--task=Pick the cube and place it in the blue square'
  --fps=30
  --duration=30
  --interpolation_multiplier=1
)
```

Use this same terminal for the tests. If the hardware ports have changed, correct the array before running. No helper script is needed.

## Test 1 — Known-good baseline

**Question:** Does the pick-and-place setup that worked at refill 0.1 s still work?

```bash
"${ACT_CLIENT[@]}" --inference.chunk_merge=append 2>&1 | tee test1-client.log
```

Watch whether it completes the task and whether the familiar backward bumps remain. If this fails, send the logs before proceeding: we need a working baseline.

**Your conclusion (2026-09-29):** Worked fine. Bumps remain noticeable, especially toward the end of the video, despite refill 0.1 s. The clip shows a completed pick and placement.

## Test 2 — Alignment only

**Question:** Does removing predictions for already committed steps reduce the bumps?

```bash
"${ACT_CLIENT[@]}" --inference.chunk_merge=aligned 2>&1 | tee test2-client.log
```

Compare with test 1: better, similar or worse? Watch for smoother progress and any buffer exhaustion. Refill does not schedule requests in aligned mode, so there is no refill tuning to do here.

**Your conclusion (2026-09-29):** Significantly worse than baseline; did not work well at the task. Run twice. The retained client log is the second run; the server log contains two distinct sessions, not duplicate requests. The supplied clip shows the arm moving past the cube without completing placement.

## Test 3 — Alignment plus blending

**Question:** Does blending add useful smoothness without hurting grasping or releasing?

```bash
"${ACT_CLIENT[@]}" \
  --inference.chunk_merge=aligned \
  --inference.blend_steps=5 \
  --inference.blend_weight=0.5 \
  --inference.blend_components='[shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos]' \
  2>&1 | tee test3-client.log
```

Compare with test 2. Note whether the arm is smoother but less accurate, and whether grasp/release still works. We can check the logs to confirm blending actually occurred. Do not tune the window or weight yet.

**Your conclusion (2026-09-29):** Significantly worse than baseline; did not work well at the task. The clip shows approach followed by little progress near the cube. Logs confirm five-step blending was active on all 371 accepted merges after startup, with the gripper excluded.

### Review of tests 1–3 — 2026-09-29

Inputs are `test1-client.log`, `test2-client.log`, `test3-client.log`, their matching `.mp4` clips, and `server.log` in the repository root. Videos were inspected through sampled frames; clips are shorter than the logged runs and are not synchronized to log timestamps. The user's task-performance assessment is the primary outcome.

| Measured behavior | Append | Alignment | Alignment + blending |
| --- | --- | --- | --- |
| Median interval between requests | 1.009 s | 0.036 s | 0.036 s |
| Median actions between chunk first-dispatch events | 30 | 1 | 1 |
| Median full request turnaround | 60 ms | 36 ms | 36 ms |
| Median trimmed actions per accepted result | 0 | 2 | 2 |
| Median source age at a chunk's first dispatch | 136 ms | 69 ms | 237 ms, oldest blend contributor |

All recorded results were accepted; no client error, exhaustion or stale-source fault appears in these files. Alignment retained roughly 0.93 s of future playback after a typical merge. These results point toward execution behavior, not an observed shortage of actions. The blended age is a conservative oldest-contributor age, not the latest observation's age or added network latency.

**Leading hypothesis, not a confirmed root cause:** aligned mode changes both alignment and replanning cadence. It repeatedly replaces the future after only one or two actions, whereas baseline executes the full 30-step plan. This can prevent a chunk policy from following through on a useful trajectory. Five-step blending at this cadence also repeatedly mixes overlapping predictions (median six contributors per first-dispatched target); it does not ensure task progress or reproduce ACT's native temporal ensemble. Sequence alignment establishes which steps remain eligible, not that the robot reached the state assumed by the new prediction.

**Next step:** keep append/refill 0.1 as the working reference. Investigate a separately controlled replanning/replacement cadence, preserving cursor trimming, freshness and committed-action guarantees. Compare that with blending disabled first, then reconsider blending only if task performance recovers. This is a proposed follow-up, not an implemented option or an instruction to sweep parameters. Changing refill alone cannot test this hypothesis because aligned mode ignores it for scheduling. Tests 4–5 remain separate lifecycle checks; do not assume test 2 passed when selecting motion behavior for test 5.

**Stop here and send the first batch:** the three client logs, `server.log`, and a short “better/same/worse; task worked/didn't work” for each. Short videos help if differences are visible. That is enough; no detailed scoring is required.

## Test 4 — Restart after a client crash

Do this after reviewing the motion tests. Keep the same server running.

**Question:** Can a crashed client be replaced without restarting the server?

Start a stationary interactive client:

```bash
"${ACT_CLIENT[@]}" --inference.chunk_merge=aligned --interactive=true \
  2>&1 | tee test4-client-before.log
```

Do **not** type `/start`. In another terminal on the robot computer:

```bash
pgrep -af '[l]erobot-rollout'
```

Identify the actual **Python rollout process**, not its `uv` launcher, Bash or `tee`. Terminate that process with `kill -KILL PID`, replacing `PID` with its number. Keep this crash test stationary: a killed client cannot execute a software hold.

Wait roughly 35 seconds after the server reports the client absent, then in the original client terminal run:

```bash
"${ACT_CLIENT[@]}" --inference.chunk_merge=aligned --interactive=true \
  2>&1 | tee test4-client-after.log
```

It should obtain a new session without restarting the server. Type `/stop` to close it normally; no motion is needed. If it remains BUSY, save the message and server log rather than restarting away the evidence. The grace is 30 seconds from detected absence, plus any unfinished worker cleanup.

**Your conclusion:** _Not run._

## Test 5 — Server disappears during motion

Use a clear workspace and keep the normal hardware stop available. Use alignment only, assuming test 2 worked.

**Question:** Does the robot stop into a hold without returning home or resuming on its own?

```bash
"${ACT_CLIENT[@]}" --inference.chunk_merge=aligned 2>&1 | tee test5-client.log
```

After several seconds of actual motion, press **Ctrl+C in the server terminal only**.

The robot may finish eligible buffered actions, then should fault and hold. Check that it does **not** perform the normal return-to-initial-position movement. Once it has faulted, restart the server using the setup command above; the robot must not resume automatically. Record what happened and approximately how long it took to hold.

**Your conclusion:** _Not run._

## What we are leaving for later

Other checkpoints, blend tuning, JPEG, long runs, same-host comparisons, network fault variants and detailed task-race tests are deferred. Real language-model validation and other release gates remain open; these five checks are a practical next round, not a claim of full release coverage.

For now, send results as simply as:

```text
Test 1: Worked; mild bumps. [logs/video]
Test 2: Smoother; completed task. [logs/video]
Test 3: Smoothest, but missed one grasp. [logs/video]
```

For tests 4–5, report whether reconnect/hold behaved as expected and attach the relevant logs. Repeat or expand only if a result leaves a concrete question unanswered.
