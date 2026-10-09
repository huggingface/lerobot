# Simulator migration acceptance

The simulator boundary and CI matrix are implemented, but this migration is **not accepted** until all eight GPU benchmark jobs and the LIBERO training smoke test pass.

## Checkpoint metadata blocks admission

The manifest pins the existing smoke checkpoints. All eight declare `observation.state` with shape `[6]`. The native observation adapters and the saved normalization statistics instead agree on the following widths:

| Benchmark   | Native state / saved statistics |                 Action width |
| ----------- | ------------------------------: | ---------------------------: |
| LIBERO      |                               8 |                            7 |
| MetaWorld   |                               4 |                            4 |
| RoboTwin    |                              14 |                           14 |
| RoboCasa    |                              16 |                           12 |
| RoboCerebra |                               8 |                            7 |
| RoboMME     |      8 (`state` statistics key) | 8 (`actions` statistics key) |
| LIBERO-plus |                               8 |                            7 |
| VLABench    |                               7 |                            7 |

Admission deliberately requires corrected checkpoint metadata. No profile repair or environment-derived replacement is applied to checkpoint feature declarations. Corrected artifacts must declare their actual state shape, canonical `action` output, camera set and raw image shapes, and compatible saved processor mappings. Update the manifest to pin those new artifact revisions after validating them. Existing profiles explicitly configure placeholder cameras; placeholders do not satisfy missing ordinary camera declarations.

Additional known mismatches:

- All checkpoints declare camera1/2/3 as CHW 256×256. Existing benchmark configurations emit other resolutions for several simulators; LIBERO emits 360×360, MetaWorld 480×480, and VLABench 480×480. Model resizing remains on the policy side and does not alter the raw input contract.
- LIBERO, MetaWorld, RoboCerebra, and LIBERO-plus emit fewer ordinary cameras than those checkpoint configs declare.
- RoboMME declares `actions` instead of canonical `action`, and its saved statistics use `state`/`actions`. Its processor mappings need explicit validation against corrected canonical metadata.
- Saved LIBERO-plus/RoboMME placeholder-camera declarations must match the profile's explicit placeholder configuration.

The existing in-process path remains available during migration. Passing that path does not waive server admission checks.

## Validation performed locally

- Combined regression suite after the quality audit: 544 passed, 2 skipped, covering shared NumPy codec compatibility, remote admission and inference, simulator lifecycle, adapters, and hardware hold.
- Real Zenoh loopback sessions with a deterministic backend, including batching, seeded resets, generation rejection, terminal freezing, ownership, rendering, timeouts, client loss, and server loss.
- Realtime RemoteRobot with synchronous, RTC, and remote inference engines, command hold and episode resets.
- Actual RemoteRobot dataset recording preserves state/action order, RGB pixels, and task descriptions; the subsequent simulator/rollout/CI run passed 148 tests.
- The policy/evaluation image builds from the locked checkout and imports both evaluation and rollout entry points offline.
- Separate MetaWorld and LIBERO images built; actual software-rendered reset/step runs without torch in either runtime.
- LIBERO and MetaWorld matched all canonical observations (including RGB), rewards, and terminal flags for five seeded software-rendered transitions. Seeded native/server transition parity is checked by `scripts/ci/check_sim_parity.py`, and runs in the LIBERO/MetaWorld CI jobs before checkpoint evaluation.

Full checkpoint evaluation and the training smoke still require the existing GPU runners. The local Docker GPU runtime fails device discovery; software rendering validates physics and communication, not GPU checkpoint acceptance. The other six simulator images require runner validation as well.

## Code quality audit

All pre-commit hooks, including the repository-wide mypy hook, pass on branch changes. The 20 shared/runtime modules also pass mypy with `--check-untyped-defs`. Running mypy in the development environment reports dependency-sensitive errors also present on the upstream #4836 base; none are introduced by this branch. New Python and Docker sources carry Apache-2.0 headers, preserving existing notices on moved code. Shared/server imports remain covered by the torch-disabled subprocess guard.

Personal checkpoint profiles and examples are excluded from the committed benchmark suite. The benchmark checkpoint metadata blockers above remain unresolved.

## Version and dependency boundary

The server's minimal dependencies are pinned in `docker/sims/requirements.txt`. MetaWorld is pinned at 3.0.0; LIBERO at hf-libero 0.1.4, robosuite 1.4.0, and MuJoCo 3.3.1. Other simulator sources and assets are pinned in their Dockerfiles. The policy image uses this checkout and `uv.lock`, replacing the previous nightly image. Baseline checkpoint results must be compared on GPU runners before recording any success-rate change caused by that image migration.

RoboTwin retains torch for planning. RoboMME also requires torch through its native ManiSkill runtime; it is an explicit additional exception to the torch-free backend import guard. Neither simulator receives policy dependencies from the evaluation image.
