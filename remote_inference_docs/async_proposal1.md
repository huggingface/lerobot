# Decoupled VLA Inference & Edge Control: System Design Proposal

## 1. Executive Summary

This document proposes a production-grade system for decoupling GPU-bound VLA (Vision-Language-Action) policy inference from high-frequency, CPU-bound robot control in LeRobot. The system adopts a **Model-as-a-Service (MaaS)** paradigm using **Zenoh** as the sole transport protocol, enabling multiple edge devices to be served by centralized GPU servers with minimal latency and high reliability.

An initial prototype exists in `src/lerobot/async_inference/` (gRPC-based, single-client). This proposal defines the target architecture, identifies gaps between the prototype and production requirements, documents known bugs, and establishes the design for the new system.

---

## 2. Motivation

LeRobot's standard control loop runs policy inference and robot I/O in the same process. This works for lightweight policies on local GPUs, but breaks down when:

- **The policy is too large for edge hardware** (e.g., Pi0 at ~3B parameters requires a dedicated GPU).
- **Multiple robots need the same policy** (redundant GPU allocation per robot).
- **Inference latency exceeds the control deadline** (e.g., 200ms inference on a 33ms control loop at 30 FPS).

Decoupling inference from control solves all three: the edge device runs a tight I/O loop on a CPU, while a GPU server handles inference for one or more clients.

---

## 3. Core Architectural Principles

### 3.1 Model-as-a-Service (MaaS)

Servers initialize models **once at startup** from a configuration manifest. Edge devices do **not** trigger dynamic model loading — they route to pre-warmed servers and validate compatibility via a status endpoint.

### 3.2 Multi-Tenant & Stateless Inference

A single GPU server handles multiple edge devices executing the same task. The server is stateless per inference call — `predict_action_chunk()` is a pure function with no side effects on the model. Client isolation is achieved through per-client observation slots and Zenoh key-expression routing.

> **Invariant**: `predict_action_chunk()` must remain a pure function (no mutation of `self`) for all supported policies. This is what enables safe multi-tenant sharing of a single model instance. This invariant must be documented and tested.

### 3.3 Zenoh as primary Transport

The system uses Zenoh's pub/sub model, replacing the current gRPC implementation. Zenoh provides:

- **Hierarchical key expressions** for routing (natural fit for the cluster/experiment/model/task topology).
- **Built-in discovery** (no external service discovery needed).
- **Non-blocking publish** for observations (fire-and-forget with best-effort QoS).
- **Reliable delivery** configurable per-topic (required for action chunks).
- **Shared-memory transport** for same-machine deployments (zero-copy) (if available).

### 3.4 Local Edge CPU

Edge devices rely on standard CPUs for sensor polling, image compression, payload serialization, motor control, and data logging. No edge-GPU dependency.

---

## 4. System Topology

![alt text](MaaS_async_inference_diagram.png)

- **Cluster**: A set of GPU machines. Identified by `cluster_uuid`.
- **Experiment**: A logical grouping of servers and clients. Identified by `experiment_tag`.
- **Server**: One model + one task, pre-warmed. Serves N clients for that model/task combination.
- **Client**: One robot, one task. Publishes observations, subscribes to actions.

The number of clients a single server can handle is a **user decision** based on model inference time and acceptable latency.

---

## 5. Component Specifications

### 5.1 The Edge Device (Client)

**Responsibilities:**

1. **Observation capture**: Read sensors (cameras, motors) at the control loop frequency.
2. **Image compression**: JPEG-encode RGB images before transmission.
3. **Observation publishing**: Non-blocking Zenoh put to the observation topic.
4. **Action subscription**: Zenoh callback receives action chunks, deposits into local buffer.
5. **Action execution**: Pop actions from buffer, send to robot at control frequency.
6. **Action blending**: When a new action chunk overlaps with the current buffer, blend via configurable aggregation function (weighted average, latest-only, etc.).
7. **Latency compensation**: Calculate one-way latency from RTT, discard expired initial steps of incoming action chunks.
8. **Fail-safe**: If action buffer empties, logs a warning.
9. **Data logging**: Record raw observations and executed actions to local `LeRobotDataset` storage for deferred upload.

**Threading model:**

- **Control loop thread** (main): Capture observation → deposit in outbox → pop action from buffer → send to robot → sleep to maintain frequency.
- **Zenoh action callback** (Zenoh-managed): Receives action chunks, processes RTT, trims stale steps, deposits into action buffer.
- **Observation publisher thread**: Drains the outbox, compresses images, serializes, publishes via Zenoh.

> **Design note**: The current prototype blocks on `send_observation` inside the control loop (BUG-1, see Section 9). The new design decouples observation publishing from the control loop entirely, using a separate thread and Zenoh's non-blocking put.

### 5.2 The Inference Server (GPU Pod)

**Responsibilities:**

1. **Model pre-warming**: Load model and processor pipelines at startup from config manifest (including expected clients & policy parameters).
2. **Status publishing**: Expose model capabilities (policy type, expected camera names, resolutions, action dimensions) via Zenoh queryable.
3. **Observation subscription**: Subscribe to observation topics for all clients of this model/task. Maintain per-client observation slots (newest-only semantics).
4. **Inference**: Single inference thread processes observations sequentially (round-robin across clients). Calls `policy.predict_action_chunk()`.
5. **Action publishing**: Publish action chunks to per-client action topics with reliable QoS.

> **Thread safety**: PyTorch's `model.forward()` is not guaranteed thread-safe. Inference will be sequential, latency is mostly about the capabilities of the server to serve multiple requests.

---

## 6. Zenoh Routing & Key Expressions

### 6.1 Key Expression Schema

```
[cluster_uuid] / [experiment_tag] / [model_id] / [model_version] / [application_tag] / [client_uuid] / [topic]
```

**Example key expressions:**

| Key Expression                                   | Direction         | Purpose                            |
| ------------------------------------------------ | ----------------- | ---------------------------------- |
| `jupiter/fabio2/pi0/v1/cookie/robot_a4b9/obs`    | Client → Server   | Observation payload                |
| `jupiter/fabio2/pi0/v1/cookie/robot_a4b9/action` | Server → Client   | Action chunk                       |
| `jupiter/fabio2/pi0/v1/cookie/*/obs`             | Server subscribes | All observations for pi0/v1/cookie |
| `jupiter/fabio2/pi0/v1/cookie/status`            | Server publishes  | Model capabilities (queryable)     |

### 6.2 QoS Configuration

| Topic    | Reliability | Rationale                                                            |
| -------- | ----------- | -------------------------------------------------------------------- |
| `obs`    | Best-effort | Dropping stale observations is expected behavior.                    |
| `action` | Reliable    | Every action chunk must be delivered; loss causes action starvation. |
| `status` | Reliable    | Client needs accurate capability info before starting.               |

### 6.3 Discovery Flow

0. Server goes up with the static configuration.
1. Client constructs its target key prefix: `cluster/experiment/model/version/task/`.
2. Client queries `cluster/experiment/model/version/task/status` (Zenoh queryable).
3. Server responds with its capabilities (expected camera names, image resolutions, action dimensions, model metadata).
4. Client validates its own configuration against server capabilities.
5. On match: client starts publishing observations and subscribing to actions.
6. On mismatch: client logs an error and refuses to start.

No dynamic client discovery for now.

---

## 7. Message Schema

### 7.1 Observation Payload (Client → Server)

| Field         | Type               | Purpose                                                     |
| ------------- | ------------------ | ----------------------------------------------------------- |
| `seq_id`      | `uint64`           | Incrementing ID for causality tracking and RTT computation. |
| `client_uuid` | `string`           | Identifies the sending client.                              |
| `state`       | `bytes`            | Proprioceptive state vector (`numpy.tobytes()`).            |
| `images`      | `dict[str, bytes]` | JPEG-compressed camera images, keyed by camera name.        |
| `task`        | `string`           | Natural-language task instruction (for VLA conditioning).   |

### 7.2 Action Payload (Server → Client)

| Field                | Type      | Purpose                                                         |
| -------------------- | --------- | --------------------------------------------------------------- |
| `response_to_seq_id` | `uint64`  | Echoes the observation `seq_id` this action corresponds to.     |
| `inference_time_ms`  | `float32` | Server-side compute duration (for edge RTT math).               |
| `actions`            | `bytes`   | Action chunk as numpy array bytes (`(chunk_size, action_dim)`). |

### 7.3 Status Payload (Server, Queryable)

| Field                   | Type                | Purpose                                    |
| ----------------------- | ------------------- | ------------------------------------------ |
| `model_id`              | `string`            | Policy identifier (e.g., `pi0`).           |
| `model_version`         | `string`            | Model version or checkpoint path.          |
| `expected_cameras`      | `dict[str, (H, W)]` | Expected camera names and shapes.          |
| `action_dim`            | `int`               | Dimensionality of the action space.        |
| `max_actions_per_chunk` | `int`               | Maximum chunk size the model supports.     |
| `observation_features`  | `dict`              | Full feature specification for validation. |

### 7.4 Serialization Format

**MessagePack** for all structured metadata (compact, fast, cross-language). Image payloads are raw JPEG bytes embedded in the MessagePack structure. State vectors use `numpy.tobytes()` with shape/dtype metadata for zero-copy reconstruction.

**No pickle.** The current prototype uses `pickle.dumps`/`pickle.loads` throughout, which allows arbitrary code execution. This is replaced entirely.

---

## 8. Latency Compensation

### 8.1 RTT Calculation

The edge device tracks in-flight observations:

```python
in_flight: dict[int, float] = {}  # seq_id -> time.perf_counter() at send

# On send:
in_flight[seq_id] = time.perf_counter()

# On receive action chunk:
rtt = time.perf_counter() - in_flight[response_to_seq_id]
# delete older keys than the one received
```

> **Important**: Delete only the exact `response_to_seq_id` key from `in_flight`, not all keys `<= response_to_seq_id`. With Zenoh's best-effort transport, messages can arrive out of order. Clearing earlier keys would make their RTT unmeasurable.

### 8.2 Stale Action Trimming

When an action chunk arrives, the edge calculates how many initial steps have already expired:

```python
expired_steps = int(rtt / environment_dt)
valid_actions = action_chunk[expired_steps:]
```

The valid actions are then blended into the action buffer using the configured aggregation function.

### 8.3 Edge Cases

| Scenario                               | Behavior                                                                               |
| -------------------------------------- | -------------------------------------------------------------------------------------- |
| **First observation** (no RTT history) | Apply all action steps without trimming.                                               |
| **Dropped observations**               | Server infers on next received observation. No special handling needed.                |
| **Dropped action chunks**              | Edge continues executing current buffer. If buffer empties, warn & hold last position. |
| **Server crash**                       | Edge exhausts buffer, holds position, warns & re-validates via status query.           |

> **Assumption**: All currently supported robots are position-controlled (SO100, SO101, OMX). For velocity-controlled robots, the fail-safe must send zero-velocity instead of holding position. This should be configurable per-robot.

---

## 9. Known Bugs in Current Prototype

These issues exist in `src/lerobot/async_inference/` and must be addressed in the new implementation.

### BUG-1: `send_observation` Blocks the Control Loop (Critical)

**Location**: `robot_client.py:207`

`self.stub.SendObservations(observation_iterator)` is a synchronous gRPC call inside the 33ms control loop. For multi-camera observations (several MB after pickle), this consumes 10-20ms on the network, leaving no headroom for sensor capture and motor commands. The robot stutters.

**Resolution in new design**: Observation publishing is moved to a dedicated thread. Zenoh's `session.put()` is non-blocking by default. The control loop only deposits observations into a local outbox.

### BUG-2: Race Condition in Action Queue Aggregation (Correctness)

**Location**: `robot_client.py:236-267`

The lock on `self.action_queue` is acquired to read `internal_queue = self.action_queue.queue` (a reference to the internal deque), then **released** at line 238. The aggregation logic iterates over this reference outside the lock. Meanwhile, the control loop thread can `get_nowait()` from the same queue, mutating the deque during iteration. At line 267, the entire queue is replaced, but actions popped between 238-267 are silently lost.

**Fix**: Either hold the lock for the entire aggregation, or `list(self.action_queue.queue)` to copy contents before releasing.

### BUG-3: No RPC Deadlines (Reliability)

**Location**: `robot_client.py:278`

`GetActions` blocks indefinitely if the server hangs (GPU OOM, deadlock). The retry policy handles `UNAVAILABLE` but not a hung connection.

**Resolution in new design**: The polling `GetActions` pattern is replaced by Zenoh subscription callbacks. The client needs a watchdog timer or check when action queue is empty: if no actions are received for `T` seconds, trigger re-validation via the status service.

### BUG-4: Similarity Check Ignores Images (Correctness for VLAs)

**Location**: `helpers.py:280-297`

`observations_similar()` + `must_go` is a workaround for current architecure limitations to avoid filling up the server queue the first seconds of the task & the robot remaining idle.

**Resolution in new design**: the server always processes the latest observation per client in its inference loop, and doesn't need similarity gating at all. The client can always push.

---

## 10. Gaps Between Prototype and Target Architecture

### 10.1 Critical (Must Address)

| #   | Gap                       | Current State                                                                                                                                                   | Target State                                                                                                                              |
| --- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| G1  | **Single-client server**  | One `observation_queue(maxsize=1)`, one `last_processed_obs`, one `_predicted_timesteps`. `_reset_server()` flushes all state on any new connection.            | Per-client state (`ClientState` dataclass) keyed by `client_uuid`. Zenoh key-expression routing provides client isolation.                |
| G2  | **Dynamic model loading** | Client sends `RemotePolicyConfig` → server calls `from_pretrained()` on demand.                                                                                 | Server loads models at startup from config manifest. `SendPolicyInstructions` RPC eliminated. Client validates via status query.          |
| G3  | **gRPC transport**        | Entire `transport/` directory: proto definitions, generated stubs, chunking utils. 4 RPCs: `Ready`, `SendPolicyInstructions`, `SendObservations`, `GetActions`. | Zenoh pub/sub. Client publishes obs, subscribes to actions. Server subscribes to obs, publishes actions. Dispatching via key expressions. |
| G4  | **Pickle serialization**  | `pickle.dumps`/`pickle.loads` throughout (arbitrary code execution risk, `# nosec` suppression).                                                                | MessagePack for structured metadata + raw JPEG bytes for images + `numpy.tobytes()` for state vectors.                                    |

### 10.2 Important

| #   | Gap                              | Current State                                                                                                                                                                  | Target State                                                                                                                           |
| --- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| G5  | **No RTT/latency compensation**  | No `seq_id`, no `response_to_seq_id`, no `inference_time_ms`. Timestamps use `time.time()` (unreliable across machines).                                                       | Edge-local `perf_counter` + echoed `seq_id` + server inference duration. Stale action step trimming.                                   |
| G6  | **No hierarchical routing**      | Direct gRPC channel to `host:port`.                                                                                                                                            | Zenoh key expressions: `cluster/experiment/model/version/task/client/topic`.                                                           |
| G7  | **No data logging**              | `control_loop` has access to obs and actions but doesn't persist them.                                                                                                         | Edge records via `LeRobotDataset` (`build_dataset_frame` + `dataset.add_frame`).                                                       |
| G8  | **No authentication**            | `grpc.insecure_channel`.                                                                                                                                                       | Zenoh TLS + access control lists on key expressions.                                                                                   |
| G9  | **ProcessorPipeline divergence** | Server reimplements observation prep in `helpers.py` (custom `resize_robot_observation_image` with `F.interpolate` bilinear). Diverges from standard `RobotProcessorPipeline`. | Use the standard `RobotProcessorPipeline` + `build_dataset_frame` to ensure behavioral equivalence between record and async inference. |

### 10.3 Nice-to-Have

| #   | Gap                                   | Current State                                                                                             | Target State                                                                                                                              |
| --- | ------------------------------------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| G11 | **No status/discovery service**       | Bare `Ready()` ping.                                                                                      | Zenoh queryable at `cluster/exp/model/version/task/status`.                                                                               |
| G12 | **No monitoring**                     | `FPSTracker` + `logging.debug`.                                                                           | Structured metrics via Zenoh telemetry topics. Wildcard subscriptions for centralized monitoring.                                         |
| G13 | **No entry points**                   | Module-level `__main__`.                                                                                  | `lerobot-policy-server` and `lerobot-robot-client` console scripts in `pyproject.toml`.                                                   |
| G14 | **Ratio-based observation threshold** | `chunk_size_threshold` (0-1 ratio of queue fill). Scales oddly with different `actions_per_chunk` values. | Absolute time threshold: `buffer_time_s` calibrated to observed RTT. Send observation when `queue_size * environment_dt < buffer_time_s`. |

---

## 11. Design Decisions & Rationale

### 11.1 Why Zenoh Over gRPC

| Aspect                    | Zenoh                                                                      | gRPC                                                                               |
| ------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Communication model       | Pub/sub — natural fit for "client publishes obs, server publishes actions" | Request/response — requires polling (`GetActions` loop) or bidirectional streaming |
| Multi-tenant routing      | Hierarchical key expressions provide built-in per-client topic isolation   | Requires manual per-client channel/stream management                               |
| Discovery                 | Built-in discovery                                                         | Requires external service (mDNS, Consul, etc.)                                     |
| Observation publishing    | Non-blocking put (fire-and-forget) — resolves BUG-1 automatically          | Synchronous stream-unary call — blocks the control loop                            |
| Same-machine optimization | Shared-memory transport (zero-copy)                                        | Loopback TCP                                                                       |
| Telemetry                 | Wildcard subscriptions (`+/+/+/+/+/metrics`)                               | Requires separate monitoring infrastructure                                        |

**Tradeoffs of going Zenoh-only:**

- Smaller community, less tooling for monitoring/tracing vs. gRPC's mature ecosystem.
- No built-in schema enforcement (Zenoh sends raw bytes) — serialization correctness is entirely on us.
- Default QoS is best-effort (like UDP). Must explicitly configure reliable delivery for action chunks.
- `zenoh-python` bindings are less battle-tested than `grpcio`. Needs integration testing under network stress.

### 11.2 Why Single Inference Thread (Not Batching)

True GPU batching across clients requires collecting observations from multiple clients and running a single forward pass. This is difficult because:

- Clients send observations at different times — waiting to batch adds latency.
- Different clients may have slightly different image resolutions.
- Error in one client's observation shouldn't affect others.

**Decision**: Start with sequential processing (single inference thread, round-robin across clients). Profile GPU utilization.

### 11.4 Why MessagePack (Not Protobuf, Not FlatBuffers)

- **Protobuf**: Strong schema enforcement but heavier toolchain (proto compilation, generated code). Since we're dropping gRPC, the protobuf dependency becomes unnecessary overhead.
- **MessagePack**: Fast, compact, schema-less (enforced by application), excellent Python support (`msgpack` package), good for nested dicts with mixed types. Natural fit for observation/action payloads.

Images are embedded as raw JPEG bytes within the MessagePack structure. State vectors use `numpy.tobytes()` with shape/dtype metadata for zero-copy reconstruction.

### 11.5 Action Aggregation Strategy

When a new action chunk overlaps with the existing buffer, the overlapping timesteps must be blended. The current prototype supports configurable aggregation functions:

| Function           | Formula                 | Character                                  |
| ------------------ | ----------------------- | ------------------------------------------ |
| `weighted_average` | `0.3 * old + 0.7 * new` | Smooth transitions, favors new predictions |
| `latest_only`      | `new`                   | Most responsive, can cause discontinuities |
| `average`          | `0.5 * old + 0.5 * new` | Equal weight                               |
| `conservative`     | `0.7 * old + 0.3 * new` | Smooth, slow to adapt                      |

Ultimately, this should be the user's decision. Default to `weighted_average`. The goal of async is not to do temporal ensembling, but to provide a solution when we want to decouple inference and execution.

---

## 12. Configuration

### 12.1 Server Configuration (Manifest)

Servers are configured via a YAML manifest that declares which models to pre-warm & clients to serve:

```yaml
cluster_uuid: jupiter
experiment_tag: fabio2
server:
  - model_id: pi0
    model_version: v1
    pretrained_path: lerobot/pi0-cookie-v1
    application_tag: cookie
    device: cuda:0
    fps: 30
    endpoint: tcp/192.168.1.50:7447
clients:
  - client_uuid: cookie-worker-4269
```

### 12.2 Client Configuration

Clients are configured via draccus dataclass (CLI-compatible):

```python
@dataclass
class AsyncClientConfig:
    # Zenoh routing
    cluster_uuid: str
    experiment_tag: str
    model_id: str
    model_version: str
    application_tag: str
    client_uuid: str
    endpoint: str

    # Robot
    robot: RobotConfig

    # Control
    fps: int = 30
    actions_per_chunk: int = 50
    aggregate_fn_name: str = "weighted_average"
    jpeg_quality: int = 90

    # Fail-safe
    max_empty_cycles_before_warning: int = 10

    # Datset recording
    dataset_repo_id: str | None = None  # None = no logging

    # Task
    task: str = ""
```

---

## 14. Data Logging Integration

The client records observations and executed actions into a local `LeRobotDataset` for deferred upload to the training dataset:

```python
# In control_loop, after executing an action:
if self.dataset is not None:
    frame = build_dataset_frame(
        self.dataset.features,
        processed_observation,
        prefix=OBS_STR,
    )
    frame["action"] = executed_action_tensor
    self.dataset.add_frame(frame)
```
