# Asynchronous inference: implementation and design reference

**Implementation baseline:** `191a70177` plus the uncommitted C8–C16 cleanup recorded in [implementation_progress.md](../../../implementation_progress.md), 2026-10-06. This document describes the implemented system. It replaces [async_proposal.md](../../../async_proposal.md) as the active engineering reference; the proposal remains historical rationale. [FUTURE_WORK.md](FUTURE_WORK.md) owns pending work, not current behavior.

For installation, commands and tuning, start with the [user guide](../../../docs/source/remote_inference.mdx). This reference explains why the system behaves that way and where those decisions live in code. Software coverage, real-model execution and physical acceptance are distinct; see [validation](#13-validation-and-current-limits).

**Post-hardware update — 2026-10-06:** C8–C16 are implemented/audited: generic observation-error cleanup, explicit package exports, no runtime Git diagnostics, shared context setup, clarified snapshot ownership, consolidated operator docs, richer lifecycle logs, automatic driver-capability checks and aligned plain chunks without default blending. H1–H3 have physical evidence on the prior tested build; H4 is withdrawn. H5 recording and remaining acceptance are tracked separately in the roadmap.

## 1. Product contract

A robot computer captures observations, dispatches motor commands, handles operator interaction and records data. A policy server loads one checkpoint and its saved processors, then predicts actions for one admitted client. Inference can overlap execution of an earlier chunk. The server may run in another process on the same host or on a reachable remote host.

The objectives are uninterrupted local control while inference runs, explicit policy/robot compatibility before motion, bounded waiting and failure, and useful operator interaction. It is not a hard-real-time controller: camera, actuator and local processing calls can still delay a control tick. Neither transport delivery nor a successful model call alone authorizes motion.

The current system supports ordinary chunks, optional alignment/blending, and guided or trained RTC when the checkpoint supports the requested mode. Language generation uses the same serialized model worker and an immediate planned local wait. It does not overlap buffered policy motion today.

One server process owns one deployment/model and admits one session. Sequential clients can acquire it after cleanup. Concurrent model sharing, batching, automatic reconnection, history-aware policy execution, model selection and public multi-tenant hosting remain outside this implementation.

### Local versus remote inference

| Backend  | Model/processor owner            | Transport | Execution                                                                             |
| -------- | -------------------------------- | --------- | ------------------------------------------------------------------------------------- |
| `sync`   | Rollout control thread           | None      | Existing synchronous policy path, including the policy's own `select_action` behavior |
| `rtc`    | Local inference worker           | None      | Asynchronous RTC through shared prediction/execution components                       |
| `remote` | Server's exclusive policy worker | Zenoh     | Chunk, aligned/blended chunk, guided RTC or trained RTC                               |

A same-host server/client pair uses the remote backend. Local RTC means the separate, transport-free backend. Sync remains a distinct path; remote support cannot be inferred solely from a successful synchronous rollout.

## 2. Ownership and package boundaries

```mermaid
flowchart LR
  subgraph RobotComputer[Robot computer]
    R[Robot and cameras] --> C[Rollout control thread]
    C --> R
    C --> E[Inference engine and ChunkRuntime]
    E --> C
    E --> W[Remote worker and RemoteClient]
  end
  W <--> Z[Zenoh transport]
  subgraph PolicyServer[Policy server]
    Z <--> P[Protocol pump]
    P <--> S[SessionWorker]
    S --> M[PolicyRunner and saved processors]
    M --> S
  end
```

| Package/component                                     | Responsibility                                                                                                          | Must not own                                               |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| [`inference`](../inference/__init__.py)               | Engine interface, configuration/local factory, local engines, immutable runtime contracts, chunk runtime and prediction | Transport routing policy or rollout strategy orchestration |
| [`remote_inference`](../remote_inference/__init__.py) | Remote engine, wire protocol/codec, compatibility/admission, client exchanges, server sessions and model ownership      | Motor dispatch or robot-specific action interpretation     |
| [`rollout`](../rollout/__init__.py)                   | Setup, strategies/controller, capture and dispatch, intervention, recording and teardown                                | Remote model loading on the robot client                   |
| [`transport.zenoh`](../transport/zenoh.py)            | Explicit connectivity, bounded pub/sub, queries and liveliness                                                          | Policy state, robot safety decisions or action eligibility |
| Policies/processors                                   | Trained input preparation, prediction and canonical output semantics                                                    | Network/session scheduling                                 |

Public cross-package symbols are explicitly exported through package APIs. Policy declarations belong to `policies`; shared execution belongs to `inference`; the remote engine and serving belong to `remote_inference`. This dependency direction avoids import cycles without dynamic initializer exports. Optional dependencies are guarded in implementation modules and checked when used. The existing transport submodule import convention remains supported.

`InferenceRobot` is a structural typing protocol: it describes the action metadata, observation capture time and hold capabilities engines consume, without importing the rollout's concrete `ThreadSafeRobot`. It is neither another hardware implementation nor a class users must inherit. Context construction has one public entry point, with separate local-model and remote-descriptor preparation; common helpers resolve robot processors, connect hardware, aggregate recording features and clean up failed setup.

### Threads and locks

The control thread captures observations and asks for the next action or interpolation step. It never waits for a model or network round trip. The remote worker performs encoding/exchanges and acceptance; local RTC instead owns a policy worker. The server's protocol pump admits bounded work to `SessionWorker`, whose one execution thread owns runtime policy/processor calls, including reset and language. Loading, warmup and the initial full reset run on the startup thread before that handoff.

The engine's coordination lock protects observations and control/hold transitions. Task/query locks order instruction changes and language intent. `ChunkRuntime` owns the atomic request/queue/generation transition; `ActionQueue` provides atomic snapshots and replacement. Retargeting and result acceptance must have one order. A worker can replace eligible future actions, but only control-thread consumption advances the commitment cursor.

Cancellation ends local interest in work. It does **not** interrupt a GPU call or free policy ownership. Ordered invalidation/reset acknowledgment establishes when the server worker has actually applied a transition.

## 3. Deployment startup and admission

The server entry point is [`lerobot_policy_server.py`](../scripts/lerobot_policy_server.py). The operator supplies the checkpoint/revision/device, canonical observation/action schemas, units/semantics, supported modes and timing limits. Connecting clients cannot choose a model to download or alter trained policy parameters.

Startup loads the policy and saved processors, validates declarations and runs warmup before advertising readiness. Warmup establishes actual prediction compatibility, including model action width. Artifact identity describes the effective serving artifact/configuration; deployment name is an address, not artifact identity. Software diagnostics contain the package version, without runtime Git inspection.

The client obtains a descriptor, selects exactly one ready instance (or an explicit `instance`), and checks protocol/capability/schema compatibility. It validates feature names, order, shapes, modality, semantics, action interval, requested mode, model/canonical layout, blending selection and optional expected artifact. Remote rollout automatically configures position hold on a supported robot. The generic driver capability defaults to false; unsupported robots fail before motion. There is no separate `hold_mode` configuration option.

Admission grants a session scoped to that server boot. An acknowledged generation-control exchange also establishes session endpoint readiness before data publication. Rollout setup requires compatibility and ownership before policy motion. Compatibility is not an equality test on the package version: the protocol, execution contract and schema are authoritative; package versions are diagnostics. No transparent mode downgrade occurs.

See [`RemoteClient.connect/admit`](../remote_inference/client.py), [`SessionWorker`](../remote_inference/server.py) and [`build_remote_rollout_context`](../rollout/remote_context.py).

## 4. One observation-to-action cycle

```mermaid
sequenceDiagram
  participant C as Control thread
  participant E as Engine/runtime
  participant W as Remote worker
  participant S as Session worker
  C->>C: Read robot and cameras
  C->>E: Publish owned snapshot and task
  E->>W: Reserve eligible request/continuation
  W->>S: Observation + correlated execution context
  Note over C,E: Existing eligible actions continue locally
  S->>S: Preprocess, predict, postprocess
  S-->>W: Action chunk + identity + durations
  W->>E: Validate, trim/merge atomically
  C->>E: Check permission and pop endpoint
  E-->>C: Canonical action + provenance
  C->>C: Interpolation, robot processing, permission check, send
```

1. `ThreadSafeRobot.get_observation` timestamps the start of the read with the client's monotonic clock. Camera waits therefore contribute to source age. `ObservationSnapshot` copies arrays into immutable storage; recycled camera buffers cannot mutate a queued request.
2. The control thread records the task/version. Aligned snapshots also capture the action commitment cursor and execution generation before another endpoint can be popped.
3. The runtime checks active/held/fault state, pending work, playback need and snapshot eligibility. It reserves one `ChunkRequest` containing the observation, request identity, atomic continuation snapshot and local submission time.
4. The remote worker encodes and sends the request. `SessionWorker` validates ownership, generation and request context before the exclusive runner performs inference.
5. The runner prepares a batch, runs the saved preprocessing/policy/postprocessing and returns canonical actions. RTC also carries the corresponding model-space actions. Server durations aid diagnostics; they are not timestamps used to align remote clocks.
6. The client validates reply identity and payload. The runtime independently checks request/generation/task, timing, finiteness and shape before updating the queue.
7. The control thread checks permission before consuming and sending, including interpolation ticks. A queue pop commits an endpoint: a later response cannot rewrite that in-progress endpoint. Recording uses the action's instruction identity, which can differ from the newly requested task.

No new reply may revive a terminal fault or a superseded session/generation. Retained valid playback and accepted future replacement are separate from a packet arriving.

Observation copying establishes ownership across threads, not distinct copies for encoding and transmission. Local RTC also deep-copies its observation before background inference, despite having no transport. `dataclasses.replace` currently reconstructs remote snapshots when anchoring or rebinding task metadata, which copies their arrays again. This is a possible optimization after measuring cost, not an additional correctness requirement. Keep original capture/task/generation provenance and immutable ownership; do not remove a copy merely because the producer currently returns fresh buffers.

## 5. Data, policy and processor contracts

### Canonical wire values

`FeatureSpec` describes data, not a captured value: name, shape, dtype, modality, ordered components and semantic convention. A six-joint state needs agreed order/units in addition to shape `(6,)`; a front RGB frame might have shape `(480, 640, 3)` and dtype `uint8`. Images are explicitly RGB HWC; arbitrary tensors are never inferred to be images. Action names/order/units describe canonical postprocessor output before robot-side processing and interpolation.

The default policy contract requires current-observation inference and a truthful `predict_action_chunk` path. It bypasses the private per-step queues that `select_action` may maintain. Checkpoint history, temporal ensembling or executed-action feedback cannot be emulated by merely taking the latest frame. Policy-specific declarations may expose existing preparation, masking/resizing or horizon behavior; they cannot remove trained semantics to satisfy admission.

A policy's prediction length and intended ordinary execution slice are distinct. `ChunkPolicySpec` declares both, supported modes, whether the current observation suffices and whether session state is retained. `PolicyCapabilities` combines that declaration with the deployment's feature schema, action interval and language/RTC settings for admission. A server-owned plain-chunk slice may shorten execution but must fit the prediction and is unavailable for RTC deployments. Actual returned horizons must match declarations.

### Model versus canonical coordinates

[`predict_chunk`](../inference/prediction.py) shares post-preprocessing prediction between local RTC and the server runner:

- Validate finite floating `[1, prediction_steps, model_width]` output. Retain the first validated model width across subsequent calls.
- Clone model coordinates **before** postprocessing can mutate them.
- Validate canonical `[1, prediction_steps, canonical_width]` output independently. Canonical width may be smaller because the saved postprocessor crops padding.
- Plain mode returns the declared execution slice. RTC keeps its supported prediction/continuation extent.

RTC leftovers are model inputs, not canonical motor commands. Relative-action RTC must reanchor the canonical continuation against the current raw state through the corresponding relative/normalization transforms. The current relative-RTC path requires matching model/canonical widths; unsupported mappings need a separately validated adapter. Prefix padding repeats the last model target: zeros in normalized coordinates could decode to the dataset mean.

Guided RTC runs under `no_grad` with ordinary tensors, allowing the policy's local guidance calculation to enable gradients. Inference-mode tensors would break that correction path. Ordinary chunk and trained inference can use inference mode. Owners retain preprocessing and policy lifetime; the shared predictor owns neither queues nor transport.

### Stateful processors and text

Saved normalization, relative/absolute action pairing and camera preparation remain authoritative. Action processors belong to the serialized owner. Language gets an isolated processor pair, copied together so the absolute-action step points to its own relative-state anchor. Both pairs still share one serialized policy owner. Full reset clears policy and processor state; motion invalidation drops queued action continuation while preserving intended planner context.

See [`PolicyRunner`](../inference/policy_runner.py), [`PreTrainedPolicy`](../policies/pretrained.py) and the policy-specific contract tests. Conformance is checkpoint/configuration-specific; a family name, successful download or warmup is not task validation.

## 6. Playback, alignment, blending and RTC

### Append

Plain append executes the current chunk and then its accepted successor. At most one request or accepted successor is prefetched beyond current execution; the runtime does not build an unbounded chain of old-observation chunks. Prefetch avoids an inference pause, but a fast response may still wait behind old playback. This explains why append can visibly return toward a position predicted from an earlier observation.

### Alignment

Alignment uses sequence progress, not a synchronized wall clock. Let `c_obs` be the commitment cursor captured with an observation and `c_now` the cursor when its reply is accepted. The first `c_now - c_obs` predicted endpoints are no longer eligible. The runtime trims them and atomically replaces only the future queue.

The currently committed/interpolating endpoint cannot be replaced. An entirely consumed suffix is rejected without destroying an otherwise valid existing future. Acceptance-time snapshot checks prevent a stale cursor/generation from replacing newer work. Fresh advanced captures and one-request-in-flight gates prevent unlimited requests from an unchanged snapshot.

Alignment establishes which temporal positions remain usable. It does not prove the robot physically matches the predicted state at those positions, or that repeatedly replacing a policy's early trajectory improves task success.

### Blending

Optional blending applies only to aligned plain chunks and explicitly selected continuous canonical components approved by the server. For matching future positions in the overlap window:

```text
blended = q_t + b_w * (i_t - q_t)
```

`q_t` is the queued target and `i_t` the incoming target for the same future slot. `b_w` (`blend_weight`) weights the **incoming** prediction. One means replacement. `blend_steps=0` disables blending. Other coordinates take the incoming values; gripper channels must not be selected merely because they are numeric. Relative-to-absolute processing occurs before blending.

A future endpoint may be blended again before it is committed. Provenance retains a bounded summary, including the oldest contributor's capture time, count and digest. Repeated blending cannot make old data fresh. Contributors from incompatible context or beyond source-age bounds are excluded.

### RTC

Guided/trained RTC conditions prediction on an ongoing model-space continuation rather than averaging independently produced canonical chunks. Requested modes must be supported and agree with deployment RTC configuration. The delay estimate uses complete observed turnaround and available continuation; trained mode also enforces checkpoint training-delay/horizon bounds. Acceptance considers both elapsed inference steps and actual commitment progress. Persistent out-of-range local trained predictions fault rather than retry indefinitely.

RTC and ordinary aligned blending remain different execution mechanisms. A plain chunk checkpoint does not gain RTC by enabling a CLI mode, and extra interpolation does not add policy feedback.

## 7. Timing and the freshness/continuity tradeoff

Client motion eligibility deadlines and source-age checks use client monotonic time. The server treats client capture timestamps as opaque correlation data and independently bounds queue/execution work with its own monotonic clock. No clock synchronization is needed. Sequence/cursor identity handles commitment; time bounds handle lateness/freshness. Both are necessary.

| Quantity             | Meaning                                                                                                          |
| -------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Action interval      | Negotiated policy interval, `1 / action_fps`                                                                     |
| Playback             | Queued endpoints × action interval; diagnostic estimates exclude the already committed endpoint                  |
| Turnaround           | Client request submission through response processing/acceptance, including encoding/network/model/decoding work |
| Source age           | Current local time minus original observation capture bound, checked through dispatch/interpolation              |
| Request deadline     | Absolute time bound for an outstanding action operation                                                          |
| Startup deadline     | Time allowed to establish motion after the applicable startup/control transition                                 |
| Starvation grace     | Fixed additional bounded waiting state after ongoing playback needs an unavailable endpoint                      |
| Server absence grace | Ownership cleanup window for an absent client; unrelated to client motion recovery                               |

For remote plain chunks, the effective request threshold is:

```text
effective_refill = max(refill_seconds, recent_max_turnaround + action_interval)
```

The first accepted turnaround supplies an initial floor until steady-state samples exist; the window then holds up to 100 steady-state turnarounds. Request when playback is at or below the threshold and other gates permit it. Aligned retargeting may bypass the playback threshold, but not freshness, ownership or pending-operation gates. Local RTC uses its own configured queue threshold with shared runtime behavior; it is not configured through remote `refill_seconds`.

Earlier requests create latency headroom while interrupting/replacing trajectories more often or allowing append results to age. Later requests allow more follow-through but less margin. Stronger blending may smooth motion while weakening task progress. Begin with useful synchronous-like follow-through and sufficient measured margin, then increase replanning only if the task benefits.

`N` policy steps at rate `f` represent `N/f` seconds. Interpolation creates finer motor commands at the corresponding faster rate; it does not extend that policy horizon. For aligned/RTC use the remaining horizon **after** consumed steps are trimmed. No timeout or refill value can make persistently insufficient compute throughput meet that horizon.

## 8. Waiting, recovery and terminal faults

```mermaid
stateDiagram-v2
  [*] --> Idle
  Idle --> Starting: operator start / ordered control
  Starting --> Playing: first valid fresh prediction
  Playing --> PlannedWait: language / retarget / intervention / recording
  PlannedWait --> Starting: authorized healthy resumption
  Playing --> StarvationWait: next endpoint unavailable
  StarvationWait --> Playing: fresh accepted prediction within grace
  StarvationWait --> Faulted: grace or earlier deadline expires
  Starting --> Faulted: startup / protocol / hardware failure
  Playing --> Faulted: stale action / deadline / invalid result / hardware failure
  PlannedWait --> Faulted: applicable bound / terminal failure
  Playing --> Idle: stop / segment end
  StarvationWait --> Idle: stop cancels recovery
  Faulted --> Shutdown
```

This diagram describes observable states, not a single enum in the implementation. A fault latch cannot be cleared by reset or a late response.

### What a local wait commands

`ThreadSafeRobot` retains the complete finite position target **returned by the driver after a send**, including interpolation/clipping and gripper values. Waiting refreshes that target without camera reads or policy inference. Before any applied command, it falls back to the latest cached, validated measured pose. It never substitutes the unverified requested command for an invalid driver return.

Target retention may still allow the robot to settle toward that target; it does not prove instantaneous stillness. Torque behavior depends on the driver. Capability declaration is explicit, not a universal “torque on means hold” assumption. Remote admission requires supported position-only hold. Local RTC remains available on other robots but cannot use recoverable starvation waiting there.

### Starvation recovery

On supported robots, `action_starvation_grace_s` defaults to one second. Zero selects immediate terminal exhaustion. Queue length zero alone is insufficient: the current committed endpoint may still be interpolating. Recovery begins only when execution actually needs another endpoint.

1. Invalidate old motion/generation once, remember an outstanding call's original absolute deadline, and start one fixed grace deadline.
2. The control thread clears interpolation, commands local retention and acknowledges it. Captures from before that acknowledgment are ineligible for resumption.
3. Drain old work and complete serialized invalidation. Remote control acknowledgment establishes that server ordering; cancelling a local exchange alone does not.
4. Request a fresh prediction from an eligible post-wait capture. Resume only if it is accepted in the healthy current generation before grace and all earlier applicable bounds expire.

Grace includes draining, invalidation and new inference. Packets, repeated observations, query upgrades and unusable results cannot extend it. The old operation deadline remains until worker invalidation proves it drained. Stop/reset revoke recovery. Deadline checks continue on control ticks even while a worker is blocked.

Starvation grace cannot override stale sources, malformed replies, known hardware faults or latched server loss. Repeated stop/start recoveries indicate a throughput/tuning problem rather than a reason to keep extending the grace.

### Other waits are intentionally distinct

Operator pause requires operator resumption. Recording save waits resume after healthy completion. Planned language work pauses immediately and uses its language/transition bounds. A query arriving during starvation cannot renew the existing starvation grace. Shared motor behavior does not remove the need to track reason, context and resumption authority.

### Terminal shutdown and observation failures

Once failure is terminal, revoke policy dispatch and run local teardown. Honor `return_to_initial_position` when trustworthy actuator feedback/control remain available, then disconnect with the driver's configured torque semantics. A brief hold before teardown does not promise indefinite powered retention. Server restart cannot resume a terminated or faulted client.

Ordinary observation errors propagate through the existing rollout failure/cleanup path without camera-versus-motor classification. Homing checks connection state and reads the full observation; a persistent camera failure can prevent it. Failed command/hold application latches a hardware failure and suppresses additional shutdown movement. Camera-independent homing belongs to a separate hardware/rollout change, outside this feature.

See [`robot_wrapper.py`](../rollout/robot_wrapper.py) and [`strategies/core.py`](../rollout/strategies/core.py). Interactive strategy failures must also produce an unsuccessful CLI outcome after cleanup, even if inference itself remained healthy.

## 9. Instructions, language, intervention and recording

Tasks carry a version, so switching away and back to identical text still changes intent. In-flight obsolete task results cannot execute. Supported in-flight retargets use a planned wait and fresh resumption; between-request changes can preserve eligible continuity. Reset clears episode state, whereas retarget/invalidation need not erase language planner memory.

VQA is an informational answer to the operator. Autosteering asks for a next subtask and applies it to subsequent action inference. Query identity, task version, generation and sequencer intent must still match when an answer finishes. Cancelling a VQA reports cancellation once without exposing its obsolete answer or declaring a still-busy worker free. A malformed/failed generated subtask leaves the previous instruction intact and stops the applicable sequencer; transport/operation failures can instead fault the execution path.

The worker waits for a local hold acknowledgment and ordered generation transition before language execution, using a fresh observation. After language work, action resumption also requires fresh input. Autosteering intervals start when a subtask is applied; at least one actual command must dispatch before another turn, even with zero interval. One queued/in-flight text operation prevents unbounded query accumulation.

Strategies use the shared dispatch helper rather than calling `get_action` and sending unconditionally. DAgger and operator interventions explicitly pause/invalidate policy continuation. Blocking recording saves likewise pause before blocking the control loop and resume only after healthy completion. Ordinary datasets retain actions/observations/task labels; inference diagnostics do not add sidecar events or model/dataset upload artifacts.

## 10. Sessions, transport and wire compatibility

### Identity and lifecycle

An envelope carries protocol version, server instance, session, execution generation and request ID. Replies also preserve observation/task/artifact identity. These fields isolate both different clients and successive intents within one client. Cached/deduplicated reply state is bounded; request IDs cannot authorize cross-session replay.

Admission is exclusive. Close/absence cleanup waits for worker-owned reset before readmission. A missing client starts the configured absence grace (10 seconds by default); restored presence can cancel cleanup while the session is still recoverable on the server side. Denied retries do not extend it. A failed runner reset leaves readiness/availability false and prevents takeover of corrupted state. The process remains available for diagnostics; restart is an operator/supervisor responsibility.

Client server-presence loss is latched. Returning liveliness does not restore that client's session. Already accepted eligible actions can drain under local freshness/deadline rules before bounded waiting/shutdown. Language waits detect loss promptly. The server's presence-restoration mechanism is therefore not a promise of robot auto-reconnection.

### Zenoh semantics

The deployment name supplies the configurable addressing namespace. Let:

```text
D = lerobot/inference/v1/deployments/<deployment>
I = D/instances/<instance>
S = I/sessions/<session>
```

| Key                                        | Mechanism / purpose                                              |
| ------------------------------------------ | ---------------------------------------------------------------- |
| `D/describe`                               | Query/reply discovery of ready serving instances                 |
| `I/open`                                   | Query/reply exclusive admission                                  |
| `S/control`                                | Query/reply worker-applied reset/invalidate/close acknowledgment |
| `S/obs` → `S/act`                          | Published observations and action/error replies                  |
| `S/language/request` → `S/language/result` | Published text requests and correlated replies                   |
| `I/alive`, `S/alive`                       | Server/client liveliness tokens                                  |

Exact spelling is defined in [`protocol.py`](../remote_inference/protocol.py), [`client.py`](../remote_inference/client.py) and [`server.py`](../remote_inference/server.py). Explicit direct peer or router-client endpoints are used, with automatic discovery paths disabled by transport configuration. Installing a router changes connectivity; it does not add model scheduling or application failover.

Pub/sub uses bounded handoffs and DROP congestion control so application threads do not wait indefinitely on publication. This is not an exactly-once delivery promise. Applications correlate replies, inspect overflow and enforce deadlines. Query collection, cancellation and cleanup are also bounded. Network-library liveness is separate from model readiness.

Raw arrays and explicit RGB records use a bounded MessagePack codec, without pickle/Python-object deserialization. It rejects malformed types/shapes, nonfinite values, duplicate map keys, oversized structures and unsupported extensions. Tensor/image storage is constrained; JPEG dimensions are checked against decoder headers before decompression. Raw images are lossless; JPEG is a configured RGB bandwidth tradeoff.

### Compatibility and deployment security

The envelope requires the exact integer protocol version 1; there is no minor-version negotiation. Execution contracts/capabilities negotiate supported layouts and merge behavior separately. Package versions aid diagnosis but do not substitute for compatibility checks. Runtime does not invoke Git. The protocol must evolve if future changes break these contracts.

Direct LAN mode is for an appropriately trusted/restricted network. Deployment names and session IDs isolate addressing; they are not authentication. Secured routed deployments use explicit authentication/encryption and deployment-scoped ACLs. Keep client/server routes through the configured router rather than accidentally permitting peer bypass. The examples are starting configurations, not evidence of every network/platform's security or reachability.

Zenoh Python support is `>=1.9.0,<1.11.0`, with lockfile baseline 1.10.1. Binding and router versions must be recorded independently. Current router validation evidence belongs in progress, not inferred from this dependency range.

## 11. Configuration and observability

The server YAML owns model selection and canonical schemas. Nested CLI fields override corresponding configuration values. One maintained generic YAML is provided; machine-specific experiments remain research material. Client configuration owns request/refill/freshness/wait limits and robot connection details, while deployment capability and mode validation constrain what it may request.

Key remote defaults, defined in [`factory.py`](../inference/factory.py), are aligned plain chunks, refill 0.5s, source age 5s, handshake/startup 10s, action timeout 5s, language timeout 60s, starvation grace 1s, raw RGB and JPEG quality 90 when selected. Blending defaults to zero steps, incoming weight 0.5 and no selected components. `chunk_merge=auto` resolves to aligned for plain chunks and append for RTC; the latter retains RTC's own continuation handling. Explicit append remains available. New clients send resolved settings; a peer omitting optional chunk settings retains the historical append wire meaning. Explicit deployment and semantics are required, and driver hold support is checked automatically. These are starting configuration values, not latency guarantees.

The server has its own operation bounds, warmup settings and absence cleanup limit in [`configs.py`](../remote_inference/configs.py). Control acknowledgments use a separate bound covering handshake plus the larger advertised operation deadline (close is capped more tightly); they do not consume fresh motion startup time. Redesigning that control budget is deferred.

INFO logs summarize readiness, admission, effective settings, bounded operating summaries, actionable faults and shutdown outcome. Five-second summaries distinguish interval counters from rolling timing windows. The client reports cancelled/failed requests, waits/resumptions and invalidation ACK delay separately from completed-result turnaround; cancelled wait time does not reveal when the server finishes. The server includes errors/stale results and control waits. DEBUG adds request/capability details. Request spacing, committed endpoints, usable playback, source age and full turnaround help distinguish early stale append playback from excessively frequent aligned replacement or insufficient throughput. Diagnostic deltas compare endpoints, not measured physical motion.

Server shutdown logs identify the stop reason and active/queued work, transport closure, and worker completion or its bounded one-second stop timeout. Process shutdown does not drain queued operations or promise completed session cleanup; a still-running model call is reported as such.

Control-thread reporting uses bounded queues drained by the worker; diagnostic I/O is kept out of motor dispatch/runtime locks. Dropped diagnostics do not grant motion or alter eligibility. Logs retain correlation IDs; ordinary recording stays free of automatic inference-event sidecars.

## 12. Engineering decisions to preserve

| Decision                                        | Reason and accepted tradeoff                                                                               |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Local dispatch and final eligibility gate       | Network/model progress cannot directly command hardware; synchronous local I/O still limits tick timing    |
| One pending operation and exclusive model owner | Bounds memory/state races and supports mutable policies; limits concurrency/throughput                     |
| Sequence alignment plus local time bounds       | No cross-host clock dependency, while stale/late data remains rejectable                                   |
| Fresh prediction after a wait                   | Old trajectory cannot resume a changed state; recovery must budget another full inference                  |
| Applied-target retention                        | Preserves clipped/gripping targets; requires truthful driver results and explicit robot capability         |
| Absolute deadlines and terminal latches         | Prevent indefinite renewal or resurrection by late data; transient server loss requires a new client       |
| Explicit policy/processor contract              | Preserves trained semantics; some synchronously usable checkpoints need targeted adaptation                |
| Immediate planned language wait                 | Keeps serialized state and resumption simple; buffered-motion VQA remains a later extension                |
| Shared predictor, separate owners               | Reuses coordinate/gradient validation without moving rollout or transport concerns into policies           |
| Explicit public APIs and directed dependencies  | Exposes package boundaries without lazy initializer machinery or requiring unrelated optional dependencies |

## 13. Validation and current limits

Automated tests protect contracts and transitions at distinct levels: deterministic runtime/queue invariants, worker ordering/cancellation, saved-processor/policy conformance, real transport/session exchange, and rollout hardware-boundary/CLI behavior. These levels are not substitutes for one another. Test consolidation should remove redundant combinations, not unique failure paths.

Historical hardware evidence covers the named ACT, SmolVLA, XVLA and LaWAM configurations, corrected aligned/blended motion, stationary client crash/re-admission and configured return after server loss. The 2026-10-06 OMX/XVLA round passed remote H1–H3 (target retention/resumption, temporary starvation recovery and controlled grace expiry/shutdown); the local RTC counterpart remains separate. H4 was withdrawn and camera-specific behavior removed under C8. Unplanned shortages observed before H3's injection remain a diagnostic follow-up. H5 recording, real language, additional checkpoint acceptance and representative routed/hosted robot deployments remain separate pending evidence; consult the roadmap for individual model outcomes.

`FUTURE_WORK.md` contains the definitive pending checklist. Do not infer universal robot support, physical stillness, autonomous recovery, history-aware execution or cloud availability from passing software tests. Update this reference when implemented behavior changes; keep experimental results and historical decisions in their respective records.
