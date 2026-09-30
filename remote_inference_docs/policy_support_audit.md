# Remote policy support audit

Source audit: 2026-09-30, feature integration `9d4b931d5`. Covers all 21 built-in policy families with a registered `PreTrainedConfig`; third-party plugins require their own conformance checks. This is an implementation assessment, not a claim that every checkpoint has been loaded or physically validated.

## How to read this assessment

- **Physically exercised:** operator-reported remote action/task evidence exists for the named checkpoints. This does not validate every configuration in that family.
- **Candidate without policy changes:** no structural blocker found for the stated configuration; checkpoint loading, processors, warmup and real behavior still need validation.
- **Targeted adaptation:** an identifiable declaration, shape or preparation mismatch blocks some/all configurations. Estimated scope is based on source inspection, not an implemented fix.
- **Execution extension:** the current snapshot/chunk contract cannot preserve the policy's required observation or executed-action history, cadence, or action semantics.

“Works without changes” still requires a deployment with correct feature names, shapes, order, units, normalization and robot stop semantics. The server loads the checkpoint's saved policy configuration and processors; its CLI does not expose arbitrary `--policy.*` overrides. Do not change a checkpoint's trained history/configuration merely to bypass admission checks.

## What the runner actually requires

The [default declaration and input validator](../src/lerobot/policies/pretrained.py) require current observations (`n_obs_steps=1`, no temporal sampling or memory), no temporal ensembling, a positive `chunk_size`, and exact checkpoint input names/shapes. RGB is negotiated as HWC and converted to batched CHW floats before the saved policy preprocessor runs. Policy overrides can describe existing resize/masking/preparation behavior without relaxing the negotiated wire schema.

The [runner](../src/lerobot/inference/policy_runner.py) calls `predict_action_chunk()` directly, never `select_action()`. It requires both the prediction **before** postprocessing and the canonical result **after** postprocessing to have exactly `[1, prediction_steps, canonical_action_dim]`. Only then does plain playback take the first `execution_steps`. This means a model-space padded width or an already-cropped horizon can fail even if ordinary synchronous inference works.

The [server loader](../src/lerobot/scripts/lerobot_policy_server.py) currently constructs this runner directly. A suggested “custom runner” in an error is an engineering extension point, not a runner that users can select in today's server YAML. Warmup catches many runtime mismatches but does not prove semantic equivalence, physical task success or correct history sampling.

## Physically exercised families

| Family | Current status and tested checkpoints | Adaptation already present / limits |
| --- | --- | --- |
| [ACT](../src/lerobot/policies/act/modeling_act.py) | Physically exercised: `maximellerbach/omx_multicubes_act`, `maximellerbach/omx_pickandplace_act`. Pick-and-place also validated the corrected alignment/blending cadence. | Uses the default contract; direct chunk prediction prepares its own images. No family-specific remote adapter. Requires temporal ensembling disabled and matching feature schema. |
| [SmolVLA](../src/lerobot/policies/smolvla/modeling_smolvla.py) | Physically exercised: `imstevenpmwork/super_chatton_smolvla`. | Input validation now reflects existing camera-subset masking and configured resize/padding. Nonvisual features remain exact; unknown cameras are rejected. These changes are already integrated. |
| [XVLA](../src/lerobot/policies/xvla/modeling_xvla.py) | Physically exercised: `imstevenpmwork/xvla_super_chatton_2`. | Existing camera masking/resizing and bounded state zero-padding are declared. Different state width requires identity normalization; truncation is rejected. Chunk only; no RTC. Changes are already integrated. |
| [LaWAM](../src/lerobot/policies/lawam/modeling_lawam.py) | Physically exercised: `maximellerbach/omx_multicubes_lawam`. | Declares `action_horizon` as the returned prediction length, distinguishes future training images from current-only inference, and validates existing image resizing/optional unused state. Chunk only. More than an input-validation exception; already integrated and covered by conformance tests. |

Evidence and tuning observations remain in the [progress record](implementation_progress.md) and [workbook](hardware_experiment_workbook.md). The field evidence does not establish real VQA/autosteering support.

## Other current-observation candidates and bounded mismatches

These families have not been established as working remote hardware deployments by this audit. All candidates inherit exact input-schema validation unless stated otherwise: an internal image resize or missing-camera mask does **not** currently authorize a different declared wire input. Adapting that validation would require demonstrating equivalence with local preparation, as done for SmolVLA/XVLA.

| Family | Can run unchanged today? | Restriction, evidence and next useful check |
| --- | --- | --- |
| [π0](../src/lerobot/policies/pi0/modeling_pi0.py) | Candidate with exact schema. | Direct chunk method prepares images/state/tokens and unpads action width. Check a real checkpoint's saved processors and matching cameras; flexible resolution/camera subsets would need a validation override. |
| [π0.5](../src/lerobot/policies/pi05/modeling_pi05.py) | Candidate with exact schema and both memory options disabled. | Direct method returns the full chunk with canonical width. Visual/proprioceptive memory configurations are explicitly rejected; meaningful memory support needs cadence-aware sampling, not simply removing the guard. |
| [π0-FAST](../src/lerobot/policies/pi0_fast/modeling_pi0_fast.py) | Candidate when `n_action_steps == chunk_size`. | Decoding uses `n_action_steps` as its returned horizon while the inherited declaration expects `chunk_size`. A shorter execution slice fails shape validation. Needs an accurate prediction declaration or a full-horizon prediction path for that case. |
| [EO1](../src/lerobot/policies/eo1/modeling_eo1.py) | Candidate with exact schema. | Direct current-observation chunk path and action-width unpadding; saved multimodal processors are essential. Declares text generation, but real action/language acceptance remains pending. |
| [WALL-X](../src/lerobot/policies/wall_x/modeling_wall_x.py) | Candidate with exact schema. | Diffusion/FAST paths predict the configured chunk and unpad width. FAST additionally requires generation-prompt tokens from its processor. Declares text generation; real action/language acceptance remains pending. |
| [GR00T](../src/lerobot/policies/groot/modeling_groot.py) | Conditional candidate with matching native/policy/postprocessed horizons and widths. | `_resolve_prediction_horizon()` takes the minimum of returned/native horizon, `chunk_size` and `n_action_steps`; native action decoding can trim again. All must produce the runner's declared shape. Check saved embodiment mappings and paired processor state, especially native relative actions. Shorter slices need a targeted declaration/output-contract adaptation. |
| [MolmoAct2](../src/lerobot/policies/molmoact2/modeling_molmoact2.py) | Candidate when `n_action_steps == chunk_size` and schemas match. | `predict_action_chunk()` already slices to `n_action_steps`; reducing it below `chunk_size` conflicts with the runner. Validate continuous/discrete mode, saved action transforms and actual returned horizon. RTC is declared only for continuous inference. |
| [EVO1](../src/lerobot/policies/evo1/modeling_evo1.py) | Conditional candidate only when model and canonical action widths agree. | Prediction returns `max_action_dim`, while [postprocessing](../src/lerobot/policies/evo1/processor_evo1.py) can crop to robot width. Common smaller-robot outputs fail the runner's pre/post shape requirement. Targeted work must represent model versus canonical coordinates correctly, including RTC leftovers; do not crop normalized tensors blindly before their matching normalization. |
| [VLA-JEPA](../src/lerobot/policies/vla_jepa/modeling_vla_jepa.py) | Candidate with `enable_world_model=False`, matching schema and dimensions; default world-model-enabled configuration is rejected. | Future observation indices describe training video targets. Direct inference calls `_prepare_model_inputs(training=False)` and consumes current frames. A policy declaration distinguishing training targets from inference requirements is a likely targeted adaptation for world-model-enabled checkpoints. Verify with the saved checkpoint; do not disable a trained component solely to satisfy the gate. |

## Currently blocked families/configurations

| Family/configuration | Why the current implementation cannot serve it as-is | Likely scope |
| --- | --- | --- |
| [FastWAM](../src/lerobot/policies/fastwam/modeling_fastwam.py) | Training future-frame indices trip the inherited guard, and config uses `action_horizon` rather than `chunk_size`. Direct prediction itself consumes current images and returns a chunk. | Targeted inference declaration plus input/processor/output conformance, similar in motivation to LaWAM; not evidence of required temporal-history infrastructure. |
| [FLUX.3](../src/lerobot/policies/flux3/modeling_flux3.py), frame conditioning | Training image-window indices trip the inherited guard even though the frame path takes the current observation. | Targeted declaration and processor/schema audit for a checkpoint actually trained/configured in frame mode. Do not convert a history checkpoint into a frame checkpoint. |
| FLUX.3, history conditioning | [History processors](../src/lerobot/policies/flux3/processor_flux3.py) accumulate observations and prior commands at synchronous call cadence. The remote runner supplies snapshots per inference request, not every executed robot tick; whole-chunk postprocessing also needs its feedback semantics reviewed. | Execution/history extension, especially when conditioning on past actions. Even `n_obs_steps=1` does not automatically make command-feedback behavior correct. |
| [LingBot-VA](../src/lerobot/policies/lingbot_va/modeling_lingbot_va.py) | The generic temporal guard rejects it. More importantly, `select_action()` gathers observed keyframes as actions execute, and later chunk calls update the KV cache from those frames and action feedback. The first chunk also drops a conditioning frame's actions, so returned horizons vary. | Substantial execution/protocol/runner work: correctly sampled observations, actual executed-action feedback, cache/reset/task semantics and variable horizons. Alignment/blending makes predicted-versus-executed feedback particularly important. `maximellerbach/omx_multicubes_lingbot_lowres` is not a configuration-only test. |
| [Diffusion](../src/lerobot/policies/diffusion/modeling_diffusion.py) | Default `n_obs_steps=2`; config uses `horizon`, not `chunk_size`. The direct offline path and core model also expect appropriate temporal tensor layout, and generation returns its execution slice. | Default history checkpoints need sampling support and an accurate runner contract. A genuinely single-observation checkpoint might need only a bounded preparation/declaration adapter; setting `n_obs_steps=1` alone does not establish compatibility. |
| [Multi-task DiT](../src/lerobot/policies/multi_task_dit/modeling_multi_task_dit.py) | Default `n_obs_steps=2`, `horizon` naming, and direct chunk prediction stacks queues populated by `select_action()` after `_prepare_batch()`. | History sampling plus direct-call preparation/shape contract; no remote support by merely renaming a field. |
| [VQ-BeT](../src/lerobot/policies/vqbet/modeling_vqbet.py) | Default five-observation history; `action_chunk_size` naming; direct prediction unconditionally reads queues filled by `select_action()`. | History/preparation and horizon-contract adaptation. |
| [TD-MPC](../src/lerobot/policies/tdmpc/modeling_tdmpc.py) | Training future indices trip the guard, no `chunk_size`, prediction relies on select-action queues and returns time-major `[T,B,A]`. Planning warm-start state and action-repeat behavior live outside the default chunk contract. | Dedicated preparation/layout/execution-semantics work. Future training indices alone are not the substantive blocker; changing only validation would be insufficient. |
| [Gaussian actor](../src/lerobot/policies/gaussian_actor/modeling_gaussian_actor.py) | No chunk horizon; `predict_action_chunk()` explicitly raises `NotImplementedError`. It emits single actions, optionally with a discrete component. | A single-step adapter may be small mechanically, but requires action/stop semantics and latency assessment. A one-action buffer gives little room for remote turnaround; not the same difficulty as LingBot's history support. |

## RTC and language are separate axes

Source declares RTC for SmolVLA, π0, π0.5, EVO1, GR00T and continuous MolmoAct2. This does not bypass the restrictions above or certify each checkpoint's remote continuation semantics. Guided/trained modes require enabled deployment configuration, correct horizons and matching continuation coordinates. π0.5 additionally declares trained RTC when the checkpoint's `rtc_training_max_delay` is positive. ACT, XVLA and LaWAM can use plain alignment/blending without RTC.

Only EO1 and WALL-X currently override `supports_text_generation()` to return true. Task-conditioned action prediction is not VQA support. Real VQA/autosteering remains deferred; processor isolation, repeated queries, task transitions and action resumption still need checkpoint validation.

## Sensible follow-up order

1. Use the four physically exercised families for transport/Spaces experiments. No new policy adapter is needed for that work.
2. When an appropriate checkpoint exists, prioritize EO1 or WALL-X for real language acceptance; first verify its exact action/processor contract and task behavior.
3. For another requested family, perform a small local-versus-runner conformance check: identical captured observation/task, equivalent preprocessing and canonical output, repeated predictions/reset, and a non-default execution slice. Control randomness or compare deterministic stages; do not require unrelated stochastic samples to match exactly.
4. Treat future training-image metadata, cropped horizons and padded model action widths as concrete contract follow-ups. Add only adapters justified by an intended checkpoint; preserve failures until equivalence is demonstrated.
5. Keep LingBot/history execution extensions separate from checkpoint-validation fixes. No framework or blanket policy opt-in is authorized by this audit.

## Audit evidence and limits

Reviewed all 21 registered config families, direct chunk/select-action paths, relevant processor behavior and shared loader/runner gates. Ran a lightweight default-config probe of the inherited declaration without constructing models; LaWAM's intentional override is separately covered by its existing conformance tests. A base-gate pass is not a warmup or deployment pass. No Hub model weights, new policy servers or robot connections were used.

Focused existing tests and their actual result are recorded in the [implementation progress](implementation_progress.md). The broader integration result of 784 passed / 7 skipped remains historical; it is not evidence of physical support for every row above.
