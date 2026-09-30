# Remote inference: focused hardware checks and results

**ACT motion comparison completed, 2026-09-30.** The operator reran the baseline and completed both Next A and Next B successfully. A had smaller, less noticeable bumps; B was smooth with no noticeable bumps. Commands below remain as the experiment record; no further ACT sweep is requested. The focused lifecycle checks also passed; see tests 4 and revised 5 below.

**LaWAM/general motion follow-up also accepted by operator report.** The user experimented with models, tasks and settings and considers the motion/task check validated. [The LaWAM commands below](#lawam-follow-up--same-lan-topology) remain as a reproducible starting point; no further motion sweep is requested. Real-language acceptance remains deferred; the focused lifecycle checks passed.

Checkpoint: `maximellerbach/omx_pickandplace_act` — pick one cube and place it in the blue square.
GPU server: **172.18.131.152**. Robot client: **172.18.133.214**.

These commands are for those two Linux machines, not the computer where this document was prepared. **Results received 2026-09-29:** test 1 worked with residual bumps; tests 2–3 performed significantly worse. Test 2 was run twice. The updated client now applies the existing refill threshold to aligned scheduling, and the subsequent successful checks below tested whether allowing more trajectory execution restores task progress. Historical results and the completed lifecycle checks are retained below.

## Setup

Use the updated implementation on both machines and inspect their startup build metadata. In particular, an old client still ignores refill for aligned scheduling; a compatible server cannot detect that client-local timing difference.

Keep the ACT hardware arrangement from your successful pick-and-place run: follower ACM1 and the same wrist/top views. **Confirm the camera mapping from your working command before pasting the client array.** The original example used indices 0 and 2, but the latest run logs show devices 1 and 4 without establishing which is wrist or top. Do not infer that mapping from device numbers. Restore roughly the same cube position between motion tests while the robot is stopped.

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
# Fill these two values from your last successful command before continuing.
ACT_WRIST_CAMERA=''
ACT_TOP_CAMERA=''
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
  --robot.type=omx_follower
  --robot.port=/dev/ttyACM1
  --robot.id=omx_follower
  --robot.use_degrees=false
  "--robot.cameras={wrist: {type: opencv, index_or_path: ${ACT_WRIST_CAMERA:?Set the wrist camera index}, width: 640, height: 480, fps: 30, fourcc: MJPG}, top: {type: opencv, index_or_path: ${ACT_TOP_CAMERA:?Set the top camera index}, width: 640, height: 480, fps: 30, fourcc: MJPG}}"
  '--task=Pick the cube and place it in the blue square'
  --fps=30
  --duration=30
  --interpolation_multiplier=1
)
```

Use this same terminal for the tests. The array refuses an empty camera index. If hardware ports have changed, correct the array before running. No helper script is needed. Refill is supplied by each test, so there is only one value to change.

## Next A — Alignment with playback-driven requests

**Question:** Does allowing more of each trajectory to execute restore pick-and-place progress while reducing the baseline's bumps?

```bash
"${ACT_CLIENT[@]}" \
  --inference.chunk_merge=aligned \
  --inference.blend_steps=0 \
  --inference.refill_seconds=0.5 \
  2>&1 | tee next-a-aligned-client.log
```

Compare against the recorded successful append/refill 0.1 run. Watch task completion/progress and bumps, not smoothness alone. The logged 30 actions at 30 Hz cover one second; refill 0.5 is a half-horizon starting point, not an established optimum. The effective threshold also includes recent maximum turnaround plus one policy interval. Request logs and first-dispatch events should show longer request spacing and more actions between replacements than the previous aligned run, without exhaustion.

If task performance is still poor, stop the motion comparisons and send the evidence. The next analysis should inspect anchoring and predicted-versus-executed trajectories, rather than add parameters or try a refill sweep. A threshold near the usable post-trim horizon can still produce frequent requests; it is not a minimum-action commitment. A warning that refill covers the full execution horizon is useful evidence to retain.

**Your conclusion (2026-09-30):** Task completed. Motion was much better than the previous aligned implementation, with smaller bumps that were harder to notice. Baseline was rerun to verify the setup.

## Next B — Add blending only if A restores useful task progress

**Question:** At the same request timing, does blending improve motion without hurting grasping, placement or release?

```bash
"${ACT_CLIENT[@]}" \
  --inference.chunk_merge=aligned \
  --inference.refill_seconds=0.5 \
  --inference.blend_steps=5 \
  --inference.blend_weight=0.5 \
  --inference.blend_components='[shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos]' \
  2>&1 | tee next-b-blended-client.log
```

Keep the same refill as A and the existing five-step, 0.5 incoming, arm-only blend; the gripper stays excluded. Do not tune the window or weight in this comparison. If blending harms the task, keep it disabled for this checkpoint and retain the logs for separate analysis.

**Your conclusion (2026-09-30):** Task completed. Motion was smooth, with no noticeable bumps.

The operator also tried aligned refill 0.4/0.6, ten blended steps and interpolation multiplier 2. Behavior varied; no complete configuration combinations or preferred setting were supplied. The operator considers this motion/task check validated, leaving deployment-specific tuning to users. These results are qualitative reports; no new logs/videos were attached for timing analysis.

**Requested follow-up model:** `maximellerbach/omx_multicubes_lingbot_lowres`. Inspection found that LingBot-VA collects observation keyframes through `select_action()` and feeds them, together with the previous action sequence, into its persistent cache. The current one-observation remote chunk contract rejects this history requirement. A correct history-aware integration would be separate scope; there is no compatible remote command to add here yet.

Send the client log, matching `server.log`, a short video if the difference is visible, and one sentence: “better/same/worse; task worked/didn't work.” This is enough for the next analysis. It tests the timing hypothesis without isolating every difference from append.

## Previous motion checks — 2026-09-29

These commands and conclusions describe the earlier implementation. **Do not rerun tests 2–3 as another batch.** That implementation requested aligned chunks whenever a fresh advanced observation and a free worker were available, independently of refill. The current implementation uses the playback gate in A/B.

### Test 1 — Known-good baseline

**Question:** Does the pick-and-place setup that worked at refill 0.1 s still work?

```bash
"${ACT_CLIENT[@]}" --inference.chunk_merge=append --inference.refill_seconds=0.1 \
  2>&1 | tee baseline-repeat-client.log
```

Repeat only if the setup has changed enough to invalidate the recorded reference. This command uses a new log name to preserve `test1-client.log`. Watch whether it completes the task and whether the familiar backward bumps remain. If a required baseline repeat fails, send the logs before proceeding.

**Your conclusion (2026-09-29):** Worked fine. Bumps remain noticeable, especially toward the end of the video, despite refill 0.1 s. The clip shows a completed pick and placement.

### Test 2 — Alignment only (historical)

**Question:** Does removing predictions for already committed steps reduce the bumps?

```bash
"${ACT_CLIENT[@]}" --inference.chunk_merge=aligned --inference.refill_seconds=0.1 \
  2>&1 | tee test2-client.log
```

The comparison sought smoother progress without buffer exhaustion. Refill did not schedule aligned requests in that implementation.

**Your conclusion (2026-09-29):** Significantly worse than baseline; did not work well at the task. Run twice. The retained client log is the second run; the server log contains two distinct sessions, not duplicate requests. The supplied clip shows the arm moving past the cube without completing placement.

### Test 3 — Alignment plus blending (historical)

**Question:** Does blending add useful smoothness without hurting grasping or releasing?

```bash
"${ACT_CLIENT[@]}" \
  --inference.chunk_merge=aligned \
  --inference.refill_seconds=0.1 \
  --inference.blend_steps=5 \
  --inference.blend_weight=0.5 \
  --inference.blend_components='[shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos]' \
  2>&1 | tee test3-client.log
```

The comparison sought smoother motion without losing grasp/release performance. Logs confirm blending actually occurred.

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

**Leading hypothesis, not a confirmed root cause:** the earlier aligned implementation changed both alignment and replanning cadence. It repeatedly replaced the future after only one or two actions, whereas baseline executed the full 30-step plan. This can prevent a chunk policy from following through on a useful trajectory. Five-step blending at this cadence also repeatedly mixed overlapping predictions (median six contributors per first-dispatched target); it does not ensure task progress or reproduce ACT's native temporal ensemble. Sequence alignment establishes which steps remain eligible, not that the robot reached the state assumed by the new prediction.

**Implemented follow-up:** aligned mode now uses the existing `refill_seconds` and turnaround floor to gate requests by remaining playback. Cursor trimming, freshness, committed-action guarantees and blend semantics remain unchanged. No separate cadence parameter was added. Next A/B above test this correction, while append/refill 0.1 remains the working reference. Tests 4–5 below remain separate lifecycle checks; the earlier alignment task failures are not evidence that those lifecycle checks passed or failed.

## LaWAM follow-up — same LAN topology

Checkpoint `maximellerbach/omx_multicubes_lawam` already has a compatible current-observation serving contract. Its saved execution horizon is **24 actions at 30 Hz = 0.8 s**, despite internal `chunk_size=50`. The existing preset and processor handle 640×480 input and resizing to 256×256. No policy edits or RTC are involved.

Start with **refill 0.4**, half that execution horizon. This is a comparison starting point, not a measured optimum; the previous append run improved at 0.2. Keep interpolation at 1 for the first two runs so the blend comparison changes only blending. Use the same physical top/wrist views as the successful ACT setup, with the checkpoint mapping below.

**Server — 172.18.131.152:** stop the ACT server first, then run:

```bash
cd ~/Documents/lerobot
uv run --no-sync lerobot-policy-server \
  --config_path=examples/remote_inference/omx_lawam_lan.yaml \
  --execution.blendable_components='[shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos]' \
  2>&1 | tee lawam-server.log
```

Wait for readiness and confirm `execution_steps=24` and action interval about 0.0333 s. Keep this server running for both client runs.

**Client — 172.18.133.214:** in Bash, fill the camera indices from the successful ACT command. The last tested robot port was ACM1; retain the working port if it has changed.

```bash
cd ~/Documents/lerobot
LAWAM_WRIST_CAMERA=''
LAWAM_TOP_CAMERA=''
LAWAM_CLIENT=(
  uv run --no-sync lerobot-rollout
  --strategy.type=base
  --inference.type=remote
  --inference.endpoint=tcp/172.18.131.152:7447
  --inference.deployment=omx-lawam
  --inference.mode=chunk
  --inference.chunk_merge=aligned
  --inference.refill_seconds=0.4
  --inference.semantics=omx-normalized-joints-gripper-percent-v1
  --inference.hold_mode=position
  --inference.max_observation_age_s=3
  --robot.type=omx_follower
  --robot.port=/dev/ttyACM1
  --robot.id=omx_follower
  --robot.use_degrees=false
  "--robot.cameras={wrist: {type: opencv, index_or_path: ${LAWAM_WRIST_CAMERA:?Set the wrist camera index}, width: 640, height: 480, fps: 30, fourcc: MJPG}, top: {type: opencv, index_or_path: ${LAWAM_TOP_CAMERA:?Set the top camera index}, width: 640, height: 480, fps: 30, fourcc: MJPG}}"
  '--rename_map={"observation.images.wrist":"observation.images.image2","observation.images.top":"observation.images.image"}'
  '--task=pick all the cubes and place them one by one in the blue square'
  --fps=30
  --duration=30
  --interpolation_multiplier=1
)
```

**Run A — alignment.** Does this checkpoint make useful task progress with reduced bumps, without buffer exhaustion?

```bash
"${LAWAM_CLIENT[@]}" --inference.blend_steps=0 \
  2>&1 | tee lawam-aligned-client.log
```

**Run B — blending, if A works.** Does blending further improve motion while preserving grasp/release and task progress? Restore the scene while stopped and keep all other settings unchanged.

```bash
"${LAWAM_CLIENT[@]}" \
  --inference.blend_steps=5 \
  --inference.blend_weight=0.5 \
  --inference.blend_components='[shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos]' \
  2>&1 | tee lawam-blended-client.log
```

Gripper commands are excluded from blending. If A fails or behaves unexpectedly, retain the client/server logs before trying B; this is not a request for another parameter sweep. No baseline repeat is required unless the scene/camera arrangement makes the previous LaWAM reference unsuitable.

**Results (operator report, 2026-09-30):** After experimenting with models, tasks and settings, the user reports generally expected behavior and considers the motion/task check validated. For LaWAM, a reported request interval around 0.6 s with server inference around 0.1 s gave task performance similar to sync without its pauses; an interval around 0.7 s left too little latency headroom. These are qualitative/operator timing estimates, not log-verified distributions or literal `refill_seconds` settings. Exact run configurations were not supplied. No further parameter sweep is needed now; proceed to the remaining lifecycle/language checks.

## Lifecycle checks — test 4 and revised test 5 passed

These are retained acceptance checks, not additions to the next A/B motion comparison. Schedule them separately after reviewing that comparison.

**Same-host alternative:** tests 4–5 may use separate server/client processes on the original SmolVLA machine and its OMX devices. This validates absent-client cleanup and the real robot's response to server-process loss; it does not validate LAN/router loss. Use the following setup instead of the ACT setup above. SmolVLA alone is sufficient; no repeat with XVLA is required.

Server terminal (stop any previous policy server first):

```bash
cd ~/Documents/lerobot
uv run --no-sync lerobot-policy-server \
  --config_path=examples/remote_inference/omx_smolvla_local.yaml \
  2>&1 | tee -a lifecycle-server.log
```

Wait for readiness. In a separate Bash terminal, define the client below. It retains the original follower ACM0, leader ACM1, front `/dev/video2`, wrist `/dev/video0`; correct device paths if enumeration has changed. The client starts stationary until `/start`.

```bash
cd ~/Documents/lerobot
LOCAL_CLIENT=(
  uv run --no-sync lerobot-rollout
  --strategy.type=base
  --interactive=true
  --inference.type=remote
  --inference.endpoint=tcp/127.0.0.1:7447
  --inference.deployment=omx-smolvla
  --inference.mode=chunk
  --inference.chunk_merge=append
  --inference.refill_seconds=0.2
  --inference.max_observation_age_s=3
  --inference.semantics=omx-normalized-joints-gripper-percent-v1
  --inference.hold_mode=position
  --robot.type=omx_follower
  --robot.port=/dev/ttyACM0
  --robot.id=omx_follower
  --robot.use_degrees=false
  '--robot.cameras={front: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2}, wrist: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2}}'
  '--rename_map={"observation.images.wrist":"observation.images.camera1","observation.images.front":"observation.images.camera2"}'
  --teleop.type=omx_leader
  --teleop.port=/dev/ttyACM1
  '--task=Pick up the blue cube, and the yellow cube, and drop them in the green box one by one.'
  --fps=30
  --duration=30
  --interpolation_multiplier=2
)
```

For test 4, run `"${LOCAL_CLIENT[@]}" 2>&1 | tee test4-client-before.log`, **do not `/start`**, kill only the actual Python rollout process as described below, wait for absence grace/cleanup, then rerun the same command with `test4-client-after.log`. Confirm admission and `/stop` normally. Keep the server running throughout.

The original test 5 below is retained as historical evidence. For the revised shutdown check, use the command in **Test 5 follow-up** below. Both lifecycle checks use the original working append/refill 0.2/interpolation ×2 settings on SmolVLA; they are not another merge comparison. The LAN/ACT commands below remain an alternative.

### Test 4 — Restart after a client crash

Do this after reviewing the motion tests. Keep the same server running.

**Question:** Can a crashed client be replaced without restarting the server?

Start a stationary interactive client:

```bash
"${ACT_CLIENT[@]}" --inference.chunk_merge=append --inference.refill_seconds=0.1 --interactive=true \
  2>&1 | tee test4-client-before.log
```

Do **not** type `/start`. In another terminal on the robot computer:

```bash
pgrep -af '[l]erobot-rollout'
```

Identify the actual **Python rollout process**, not its `uv` launcher, Bash or `tee`. Terminate that process with `kill -KILL PID`, replacing `PID` with its number. Keep this crash test stationary: a killed client cannot execute a software hold.

With the revised defaults, wait for the server's session-release log (10 seconds from detected absence, plus worker cleanup), then in the original client terminal run:

```bash
"${ACT_CLIENT[@]}" --inference.chunk_merge=append --inference.refill_seconds=0.1 --interactive=true \
  2>&1 | tee test4-client-after.log
```

It should obtain a new session without restarting the server. Type `/stop` to close it normally; no motion is needed. If it remains BUSY, save the message and server log rather than restarting away the evidence. Historical results below used the previous 30-second grace; they remain valid evidence and do not require another stationary hardware matrix.

**Result (supplied logs, 2026-09-30): Passed on the same-host SmolVLA setup.** The server detected client absence at 16:00:41, queued cleanup and released the session at 16:01:11 (30 seconds later), then admitted a new session at 16:02:10 on the same server instance. The replacement client closed normally at 16:02:35 and the server released that session too. Both clients stayed in interactive idle (`control_tick=0`); no inference was pending at cleanup. This validates stationary crash/re-admission, not crash cleanup during an in-flight inference or server-loss hold during motion. Sources: repository-root `lifecycle-server.log`, `test4-client-before.log`, `test4-client-after.log`. The replacement used refill 0.1 rather than the initial 0.2; that does not affect this idle lifecycle check. For SmolVLA test 5, retain the original working 0.2 setting from `LOCAL_CLIENT`.

**Additional stationary lifecycle run (2026-09-30): Passed.** On one unchanged SmolVLA server instance, Ctrl+C closed the first client and released its session at 16:06:44 without the absence grace. A second client was admitted at 16:07:04, then disappeared after the operator's `pkill` (detected 16:07:26). Retries at 16:07:37 and 16:07:53 were correctly rejected with `admission_blocker=absence_grace` and approximately 19.16 s / 2.97 s remaining. Both rejected clients disconnected their cameras and robot. The retries did not extend the grace: cleanup released the old session at 16:07:56. A final client was admitted at 16:08:20 and Ctrl+C released it normally at 16:08:32. Expected admission rejection currently prints a full `ProtocolError` traceback; this is presentation noise, not a recovery failure. All runs stayed idle, so test 5 remains pending. Source attachments: server `72b032b1-41cc-48fd-a523-8e310528b6e4/Pasted text.txt`, client `bbcc5c2a-5183-4a84-a20e-7b740191689c/Pasted text.txt` under the operator's Codex attachments directory.

### Test 5 — Server disappears during motion (historical pre-change run)

Use a clear workspace and keep the normal hardware stop available. Use the working append/refill 0.1 behavior; this checks fault handling rather than alignment performance.

**Question:** Does the robot stop into a hold without returning home or resuming on its own?

```bash
"${ACT_CLIENT[@]}" --inference.chunk_merge=append --inference.refill_seconds=0.1 \
  2>&1 | tee test5-client.log
```

After several seconds of actual motion, press **Ctrl+C in the server terminal only**.

The robot may finish eligible buffered actions, then should fault and hold. Check that it does **not** perform the normal return-to-initial-position movement. Once it has faulted, restart the server using the setup command above; the robot must not resume automatically. Record what happened and approximately how long it took to hold.

**Result (2026-09-30): Terminal policy shutdown confirmed; physical controlled-stop acceptance failed.** On same-host SmolVLA, using append/refill 0.1 and interpolation ×2, the operator interrupted the server during motion. The client reported `Active motion buffer exhausted` at 18:15:44, skipped homing and began disconnect at 18:15:46, and completed teardown at 18:15:47. The operator reports that the robot lost torque and fell into the environment rather than visibly holding. This matches the current fault teardown/OMX torque-release path; a software hold call does not establish a sustained physical hold. Server Ctrl+C is not timestamped, so no precise stop latency is claimed. Server restart/no-resumption was not evidenced. Source: server log pasted in the conversation and client attachment `d477d5b9-c57c-4d25-8c00-645436cbedb4/Pasted text.txt`. Implement the agreed configured-homing/teardown change and verify the physical outcome afterward; do not repeat the unchanged failing run. Instructions above describe the original pre-change experiment, not the approved future shutdown behavior.

### Test 5 follow-up — Configured shutdown return (passed by operator report)

**Question:** With the server unavailable, does the client stop policy execution, perform the configured local return movement, and then disconnect without resuming policy motion?

Use updated server/client code and restart the server with the same local SmolVLA preset. Keep the return path and resting position clear: OMX still releases torque on disconnect by default, so reaching the commanded initial pose does not guarantee support against gravity. This is one focused check of changed behavior, not a repeat of the unchanged failing run.

```bash
"${LOCAL_CLIENT[@]}" --return_to_initial_position=true --inference.refill_seconds=0.2 \
  2>&1 | tee test5-client-revised.log
```

Type `/start`, allow a few seconds of motion, then Ctrl+C **only the server**. Expect terminal policy shutdown, a local return movement while torque is still enabled, then disconnect with the logged torque setting. A robot I/O failure should instead skip further homing and explain why. Report the actual motion/torque outcome, not just the hold log. Restart the server afterward and confirm policy motion does not resume automatically. Retain both logs; approximate stop/return timing is sufficient. Do not test `return_to_initial_position=false` with an unsupported elevated arm expecting it to stay powered: that option deliberately skips return and still disconnects. Its conditional behavior is covered in software tests.

**Result (operator report, 2026-09-30): Passed.** The server was killed during robot motion. When actions exhausted, the robot returned smoothly to its initial position and the client exited cleanly. The client finished shutdown before the server could be restarted; a server restart cannot resume an exited rollout process. Restart during a still-live faulted client was not exercised; the user accepts this limitation, with no artificial timing/repeat experiment needed. No new logs/video or quantitative timing supplied. Retain the earlier failed Test 5 as evidence of the previous implementation, distinct from this successful revised run.

## What we are leaving for later

Other checkpoints, blend tuning, JPEG, long runs, same-host motion comparisons, network fault variants and detailed task-race tests are deferred. Real language-model validation remains open; the completed motion comparison and focused lifecycle checks are not a claim of full release coverage.

For now, send results as simply as:

```text
Next A: Completed task; fewer bumps than the baseline. [logs/video]
Next B: Smoother; grasp/release still worked. [logs/video]
```

For tests 4–5, report whether reconnect/hold behaved as expected and attach the relevant logs. Repeat or expand only if a result leaves a concrete question unanswered.

## Deferred follow-up experiments

These are queued, not requests to run another matrix now. Use one known working checkpoint: collect full turnaround/tail latency and playback margin as a LAN baseline; run through a Zenoh router; compare raw/JPEG task behavior and transport cost; then try a private remote or explicitly secured public-network path when available. Language/VQA/autosteering waits for a suitable checkpoint. No additional hardware-cleanup hardening belongs to this feature integration.

**Preferred follow-up to the router test: one client with a GPU Space server.** First prove that a dedicated GPU Docker Space can connect outbound to the same authenticated/encrypted router, with normal request/reply and presence behavior. Check actual Spaces egress restrictions before preparing hardware commands; port 443 alone is not proof of compatibility. Then move one already-working checkpoint/server to the Space and repeat one task with the robot client unchanged except deployment connectivity/tuning. Question: *Can this hosted server sustain useful task execution within the existing playback and freshness limits, and does interruption still lead to the expected local shutdown?* Record full turnaround variability, payload/codec, visible behavior and restart outcome. No Space, router or public exposure is provisioned by this plan; commands follow only after transport feasibility is established. Details and platform references are in [proposal section 15](async_proposal.md#single-client-gpu-space-after-the-router-experiment).

**Separate later experiment: two robots sharing one loaded model.** This first requires state isolation, per-session pending work and scheduling/admission implementation. It is not a current CLI test or a prerequisite for Spaces. Question: *Can both clients meet their action budgets without one client's state, controls or long-running work affecting the other?* Start with a verified shareable policy and trusted clients; public multi-tenant hosting and batching remain separate scope.

For checkpoint selection, consult the [policy support audit](policy_support_audit.md). Reuse ACT/SmolVLA/XVLA/LaWAM for the transport experiment; do not combine a new deployment topology with an unverified policy adapter.

Experiment logs/videos referenced by filename in this workbook remain on `test/remote_inference_super_chatton`; they are deliberately not copied into the feature branch. Conclusions above retain their evidence provenance. Educational HTML/PPTX is retained as dated documentation, not experiment footage.
