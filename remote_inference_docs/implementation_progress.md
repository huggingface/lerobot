# Async remote inference implementation progress

Authoritative requirements: [async_proposal.md](async_proposal.md), 2026-09-28.
Baseline: `e595b7902`. Existing proposal and other untracked documentation are user work.

## Milestones and gates

- [x] Stage 1 implementation and automated gate: policy contract/runner, atomic queue snapshots, shared `ChunkRuntime`, worker-owned local RTC reset, task provenance and planned local language hold. Relevant rollout/RTC and shared-component regressions pass.
- [x] Stage 2 implementation and automated gate: bounded codec/Zenoh channels, exclusive ordered worker, descriptor-based startup without client model loading, remote engine and dispatch/hold gate. Fake-robot, real processor, direct TCP and authenticated router scenarios pass.
- [x] Stage 3 implementation and automated lifecycle coverage: live task versions, language topics, hold acknowledgment, query-intent generations, serialized controls and fresh action resumption. **Acceptance gate remains open: VQA/autosteering have not run with a real text-capable checkpoint.**
- [ ] Stage 4 hardware/performance validation and legacy replacement. New user documentation, examples, extra and script registration are implemented. **Removal gate requires the complete real action/language workflow first.**

## Inspection and decisions

- Scope comes only from `async_proposal.md`; older documents and draft branch are not requirements.
- Keep synchronous `select_action()` execution intact.
- Inspection found independently locked continuation getters and policy/processor reset on the control thread. These now use atomic snapshots and worker-owned resets.
- Plain chunks must honor execution length independently of predicted horizon; RTC needs both canonical and model-space continuation.
- Fail closed for unsupported robot stop semantics, policy history/preparation, feature semantics, and explicitly requested modes.
- Do not remove legacy async/gRPC service before the proposal's real workflow validation gate passes; unrelated RL protobuf definitions and dependencies remain.
- Explicit semantic profile plus named feature conventions are required on the deployment/client; wire camera resolution is exact. SmolVLA and XVLA declare their existing internal resize/optional-camera behavior through policy-owned validation hooks; other policies retain exact checkpoint shapes. Rename mapping happens once client-side.
- Position hold is implemented for `SOFollower` and `OmxFollower` robots with only `.pos` actions. Unsupported/mixed control modes are rejected for async rollout. Physical hold validation is outstanding.
- Timing defaults are provisional lab starting points, not measured profiles. Local RTC gains additive age/action/startup/language bounds; synchronous action execution is unchanged.
- Server content identity hashes checkpoint/processor files, optional resolved adapter/base contents and effective settings. Warmup is server-only and is followed by a full reset before readiness.
- The server uses one serialized worker, capacity-one inference admission and bounded control/reply queues. A hung call cannot release model ownership to another session.
- Monotonic data sequence numbers supplement bounded request-ID history, preventing duplicate execution after ID-cache eviction.
- Zenoh binding is pinned to 1.9.0; the matching 1.9.0 router was tested. MessagePack is constrained to `>=1.1,<2`, with 1.2.2 selected in `uv.lock`. Direct callbacks only perform bounded handoff; query handles outlive callbacks until worker acknowledgments. Explicit subscriber readiness is required before data publication.
- Recording uses asynchronously written, bounded JSONL events keyed by session/control tick. Canonical targets, post-robot-processor commands and scalar measured state remain distinct.
- Planned holds preserve pending full resets and clear motion before acknowledging the hold. Language uses an isolated canonical processor pair. Autosteering requires an actual dispatched action after applying a subtask before requesting another, including with a zero interval.
- Startup/compilation and steady action deadlines are distinct. A deadline fault cannot restart warmup or restore dispatch. Local plain-chunk playback uses the declared execution slice; history-dependent configurations are rejected explicitly.
- No older draft code was reused. No future scheduling, recovery or multi-client framework was added. Legacy async/gRPC removal is deferred by the proposal's gate, not replaced with a compatibility shim.

## Validation performed

- Repository status/baseline and complete proposal inspected.
- Baseline: `UV_CACHE_DIR=/private/tmp/lerobot-uv-cache uv run --no-sync pytest tests/policies/rtc/test_action_queue.py tests/test_rollout.py tests/test_interactive_rollout.py tests/test_rollout_action_ordering.py -q --disable-warnings --maxfail=3`: **198 passed, 2 skipped**.
- Final aggregate command (2026-09-28):

  ```bash
  LEROBOT_ZENOHD=/private/tmp/lerobot-zenoh-router/zenohd \
  UV_CACHE_DIR=/private/tmp/lerobot-uv-cache uv run --no-sync pytest \
    tests/inference tests/remote_inference tests/test_remote_rollout.py \
    tests/test_rollout.py tests/test_interactive_rollout.py \
    tests/test_rollout_action_ordering.py tests/policies/rtc \
    tests/policies/pi0_pi05/test_pi05_training_time_rtc.py \
    tests/policies/test_pretrained_interactive_contracts.py \
    -q --disable-warnings --maxfail=5
  ```

  **499 passed, 4 skipped**. Test configuration selected MPS on this Mac. Loopback tests needed sandbox escalation. The router binary is temporary and is not a repository dependency; install the matching official router and set `LEROBOT_ZENOHD` to rerun its integration tests.
- The aggregate covers shared scheduling/continuations, raw codec exactness and malformed bounds, real tiny ACT plus canonical processor pipelines, saved-checkpoint server load/warm/reset/identity, no client weight loading, fake-robot dispatch/interpolation/fault hold, JSONL provenance, duplicate/stale/session races, held and cancelled language queries, task labeling and autosteering. Language behavior uses controlled executors; this is not real language-model validation.
- Real direct/router tests use eclipse-zenoh/zenohd 1.9.0 and the checked-in JSON5 examples: successful mTLS action/text/control exchange; rejection of forged action publications, unauthorized observation/control access and missing client certificates; presence revocation and router loss. No WAN or hardware performance conclusion follows from loopback tests.
- Ruff check and format check pass for changed Python files. Targeted mypy with `--follow-imports=silent` passes across 16 new/shared modules; additional local base/core checks pass. Full-repository mypy/pre-commit and the entire test suite were not claimed or run as final gates.
- `uv lock --check --offline`, `git diff --check`, example server YAML parsing and `uv run --no-sync python -m lerobot.scripts.lerobot_policy_server --help` pass. CLI smoke exposed a postponed-annotation/parser incompatibility; removing postponed function annotations fixed it.

## Remaining work / handoff

1. Run a supported action checkpoint and a real text-capable checkpoint (including VQA and autosteering) on an actual position-controlled robot. Validate cold startup, task changes, intervention, pause/reset, long text generation, starvation and fault teardown without homing. Verify the physical hold behavior before relying on the supported robot declaration.
2. Measure GPU/model turnaround tails, edge encoding/decoding cost, playback coverage, task-change latency and JPEG policy impact over wired LAN and representative private remote conditions. Replace provisional deadline/refill profiles with measured settings. Wire camera schemas require exact resolution; checkpoint shape exceptions require a policy-owned input validation contract.
3. Once the real action/language gate passes, remove the legacy async package/tests/docs and dedicated configuration. Remove only its protobuf service/messages, regenerate bindings and run transport/RL checks; retain unrelated gRPC services/dependencies. Finish stale-reference cleanup then.

Implementation and automated checks are complete to the available environment. No robot, real language weights, CUDA server or representative private-network deployment was available. No commits were created. User-staged proposal and other pre-existing documentation were preserved.

Entry points for continuation: [user guide](../docs/source/remote_inference.mdx),
[server example](../examples/remote_inference/server.yaml),
[router configuration](../examples/remote_inference/zenoh/router.json5).

## OMX/SmolVLA same-machine follow-up

- Added `examples/remote_inference/omx_smolvla_local.yaml` for the user's existing `imstevenpmwork/super_chatton_smolvla` checkpoint, two 640×480 cameras, normalized OMX joints, CUDA server and localhost-only listener. This is ordinary chunk mode with RTC disabled; natural-language task conditioning remains enabled, while text-query serving is disabled.
- Inspected the checkpoint's public `config.json` and processor metadata without downloading weights: one observation, 50 predicted/executed actions, six state/action coordinates, camera1/camera2 plus unused camera entries, internal 512×512 resize/padding and one empty-camera slot. Preserve wrist→camera1/front→camera2 and the existing `use_degrees=false` units.
- Fixed two setup blockers: OMX's position-only driver now declares hold support; the runner delegates input compatibility to a policy-owned validator, with SmolVLA allowing its existing missing-camera/resize behavior. No synthetic unmasked cameras or changed preprocessing were introduced; other policies retain the strict default.
- Follow-up validation: **54 passed** across `tests/inference`, `tests/test_remote_rollout.py`, server startup and session tests. The new tests compare actual SmolVLA image preparation/masks for local versus remote inputs, reject incompatible schemas, and exercise OMX's actual position command path with a fake motor bus. Hardware/model weights were not exercised.
- Initially suggested refill at 1 second with maximum source age 3 seconds. The first hardware logs below show that refill was unnecessarily early; use 0.2 seconds for the next controlled test on that measured setup.

## XVLA same-machine follow-up

- Added `examples/remote_inference/omx_xvla_local.yaml` for `imstevenpmwork/xvla_super_chatton_2`, deployment `omx-xvla`, CUDA, localhost port 7447 and the same normalized OMX joint units. Front maps to `observation.images.image`; wrist maps to `observation.images.image2`. Stop the previous server before switching presets. Client `--device=cuda` is omitted; compute placement belongs to the server.
- Inspected public checkpoint and processor metadata without downloading weights: one observation, 30 predicted/executed actions, six output coordinates, `action_mode=auto`, identity state normalization, nominal eight-wide state padded by the existing policy to 20, three declared image views with 224×224 internal resizing, and domain ID 0. The physical wire state remains six explicitly named joints; no extra camera or state values are manufactured on the client.
- Added XVLA's input-validation override to preserve the existing policy's camera masking/resizing and state zero-padding. Differing state widths require identity normalization and must fit `max_state_dim`; truncation is rejected. Ordinary chunk mode only; XVLA does not declare RTC support.
- **57 focused tests passed** across `tests/inference`, `tests/test_remote_rollout.py`, `tests/remote_inference/test_server_startup.py` and `tests/remote_inference/test_session.py`. XVLA-specific coverage compares actual local/remote input preparation through image/domain/normalization processors and checks camera masks, state padding, invalid schemas and RTC rejection. Tokenization/model weights were not downloaded or run. Ruff check/format and `git diff --check` pass.
- Hardware and CUDA inference remain manual checks. At 30 Hz this checkpoint has one second of playback per chunk; an increased request timeout cannot prevent starvation if steady inference is slower than available playback.

## First SmolVLA/OMX hardware run — 2026-09-28

- User supplied server/client logs and a 21-second video (`20260928_173004.mp4`) from separate processes on the same Linux machine. Examined video frames: the arm repeatedly pulls back while progressing toward the yellow cube and transporting it to the box. This is partial action-path hardware evidence; smooth motion, stop/hold behavior and full action/language acceptance remain open.
- Logs: 22/22 action results accepted, no fault events, one execution generation throughout motion. Steady turnaround median **103.756 ms**, maximum **133.264 ms**. Cadence **29.86 Hz policy / 59.71 Hz commands**, no loop-body budget overruns. Shutdown was operator Ctrl+C; the final return-to-initial-pose log is separate from recurring pullbacks during execution.
- Leading diagnosis: `refill_seconds=1.0` requests an independent plain chunk too early. At acceptance the queue contains 50 new actions plus typically 27 old ones: `queue_playback_s=2.5667`, leaving **0.9 s** of old playback. Estimated median source age at the successor's first action is **1.008 s**, despite source age near 0.108 s at result acceptance. Existing plain append semantics intentionally do not condition/align the new trajectory like RTC. No per-action commands were logged, so exact video-to-boundary correlation is not proven.
- Ran a diagnostic using the actual shared runtime with a simple monotonic position predictor, 50 actions at 30 Hz and 100 ms inference. With refill 1.0, the boundary went from target 50 back to 21; with refill 0.2, from 50 to 45. This illustrates the scheduling mechanism, not learned-policy or hardware performance. No execution semantics were changed.
- Next hardware check: change only `--inference.refill_seconds=0.2`; retain chunk mode, interpolation multiplier 2 and max observation age 3.0. The observed max turnaround plus one policy tick is 0.1666 s, below 0.2 s. No server change is required. This should reduce observation lag at the boundary but is not yet a verified fix; plain chunks may retain smaller discontinuities. If needed, test guided RTC explicitly as a subsequent, separate change.
- Updated the usage guide and corrected the earlier timing suggestion. Documentation-only changes this turn; no regression suite rerun. Diagnostic log parsing/runtime reproduction succeeded and `git diff --check` passed.

## Improved plain playback and guided RTC preflight

- The user reports the 0.2-second refill retest worked substantially better. This supports the early-prefetch diagnosis; no updated per-action trace, timing distribution or fault/hold validation was supplied.
- Preparing the next guided RTC experiment exposed a correctness bug: both `PolicyRunner.predict()` and the local async action loop entered `torch.inference_mode()`, which suppresses RTCProcessor's inner `torch.enable_grad()` correction graph. New regression tests with the actual `RTCProcessor.denoise_step` failed on the successor in both paths before the fix. Previous tests checked RTC prefix forwarding but missed the autograd context.
- Fixed both execution paths to disable inference mode and use `no_grad` for **guided** RTC only, allowing the guidance step to enable its own gradient graph. Plain chunks, trained RTC and text keep inference mode; returned remote actions remain detached.
- Validation: **304 passed, 3 skipped** across `tests/inference`, `tests/policies/rtc`, `tests/test_interactive_rollout.py` and `tests/remote_inference/test_server_startup.py`. Real autograd correction regressions now pass for runner and local worker. Ruff and targeted two-module mypy pass; guided server CLI overrides parse successfully. No real guided checkpoint/GPU run was performed here.
- Next hardware experiment: use the existing SmolVLA server YAML with `execution.supported_modes=[rtc_guided]`, `execution.rtc.enabled=true`, `execution.rtc.mode=guided`, `execution.rtc.execution_horizon=15`. Client mode `rtc_guided`, initial refill 0.5 s; retain raw encoding, 30 Hz, interpolation ×2 and max age 3 s. The larger provisional refill accounts for guidance compute and does not imply plain mode's append delay; measure turnaround before tuning. Deploy the code fix before this run.
