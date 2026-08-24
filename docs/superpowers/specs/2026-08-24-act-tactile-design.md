# ACT Tactile Input — Design

**Date:** 2026-08-24
**Status:** Approved (pending spec review)
**Scope:** Add optional tactile sensing to the ACT policy, encoding each tactile
sensor grid into a configurable number of transformer-encoder tokens via a small CNN.

## Goal

Let ACT optionally consume tactile observations alongside vision / proprioception.
Tactile data is provided in the dataset as one integer feature per sensor, following
the grabette export convention:

- Key: `observation.tactile.sensor_<addr>` (in general `observation.tactile.<name>`)
- Shape: `(rows, cols)` — a 2D taxel grid (e.g. `6x6`, `4x8`, `12x32`)
- dtype: `int16`, raw 12-bit ADC values (0–4095)
- A dataset may contain multiple tactile keys (multiple sensors), possibly with
  differing grid shapes.

The approach mirrors LeFlexiTac (`TNA001-AI/lerobot_tactile`): tactile maps become a
handful of extra transformer-encoder tokens; everything else in ACT is unchanged.

## Non-goals

- Wiring tactile into other policies (Diffusion, Pi0.5, SmolVLA). ACT only for now.
- Any hardware / recording / robot-side integration.
- Feeding tactile into the VAE encoder. Tactile is observation conditioning for the
  transformer encoder only, exactly like image feature tokens.
- Temporal filtering / multi-step tactile history (`n_obs_steps` stays 1 for ACT).

## Design decisions (locked)

1. **First-class feature type.** Add `FeatureType.TACTILE`. `observation.tactile.*`
   keys are typed `TACTILE` on dataset load (not silently `STATE`).
2. **Normalization: `MEAN_STD`, per cell.** ACT's `normalization_mapping` gets
   `"TACTILE": NormalizationMode.MEAN_STD`. Per-taxel mean/std removes each cell's
   resting baseline and gain so the network sees deviation-from-rest. No extra input
   BatchNorm is added; the CNN keeps its internal BatchNorm.
3. **Per-sensor CNNs.** Each tactile key gets its own `TactileTokenEncoder` (its own
   input grid shape), so sensors of differing shapes are supported. Grid shapes are
   auto-derived from each feature's `PolicyFeature.shape` — no manual
   `tactile_input_shape` config.
4. **Explicit opt-in flag.** `use_tactile: bool = False`. When enabled but no tactile
   features are present, config validation raises.
5. **Two backbone variants.** `tactile_encoder_type ∈ {"cnn", "attention"}`.
6. **Token count default 4.** `n_tactile_tokens: int = 4` (LeFlexiTac's best-performing
   default). Each sensor emits `n_tactile_tokens` tokens.
7. **ACT-local encoder.** The encoder modules live in `modeling_act.py`; no new shared
   `policies/tactile/` package.

## Components & changes

### Shared (small, cross-cutting)

- **`src/lerobot/configs/types.py`** — add `TACTILE = "TACTILE"` to `FeatureType`.
- **`src/lerobot/utils/constants.py`** — add `OBS_TACTILE = OBS_STR + ".tactile"`.
- **`src/lerobot/utils/feature_utils.py`** — in `dataset_to_policy_features`, add a
  branch that maps keys starting with `OBS_TACTILE` to `FeatureType.TACTILE`, preserving
  the 2D `(rows, cols)` shape. It MUST be placed **before** the generic
  `key.startswith(OBS_STR)` → `STATE` branch (tactile keys also start with
  `observation.`). All other policies keep ignoring TACTILE features (the normalizer
  defaults unknown feature types to `IDENTITY`, so no other config needs changes).

### ACT config (`src/lerobot/policies/act/configuration_act.py`)

New fields (additive, default off):

```python
use_tactile: bool = False
tactile_encoder_type: str = "cnn"   # "cnn" | "attention"
n_tactile_tokens: int = 4
tactile_dropout: float = 0.3
```

- `normalization_mapping` default gains `"TACTILE": NormalizationMode.MEAN_STD`.
- New property:

```python
@property
def tactile_features(self) -> dict[str, PolicyFeature]:
    if not self.input_features:
        return {}
    return {k: ft for k, ft in self.input_features.items() if ft.type is FeatureType.TACTILE}
```

- `__post_init__`: if `use_tactile` and `tactile_encoder_type not in {"cnn", "attention"}`
  → `ValueError`.
- `validate_features`: accept tactile as a valid standalone input, i.e. require at least
  one of images / env_state / tactile. If `use_tactile` is True but `tactile_features`
  is empty → `ValueError` with a clear message.

### ACT model (`src/lerobot/policies/act/modeling_act.py`)

Three new `nn.Module`s (ACT-local), faithful to LeFlexiTac but made shape-robust:

- **`TactileCNN`** — `conv(1→32→64→128)` blocks, each `Conv2d(k=3, pad=1)` + `BatchNorm2d`
  + `ReLU`. Then an `AdaptiveAvgPool2d((2, 2))` to a fixed spatial size, flatten, `fc1 →
  512` + `ReLU` + `Dropout`, `fc2 → feature_dim`. Output `(B, feature_dim)`.
  - **Rationale for adaptive pooling:** LeFlexiTac's plain CNN uses three fixed
    `MaxPool2d(2,2)` and computes the FC input as `128 * (H//8) * (W//8)`. That collapses
    for small grids (e.g. `6x6` → `6//8 = 0`). grabette sensors are `6x6` / `4x8`, so we
    replace the fixed pooling tail with `AdaptiveAvgPool2d`, making the encoder valid for
    any grid ≥ `1x1` while keeping the conv stack faithful.
- **`TactileAttentionCNN`** — `conv(1→64→128→256)` blocks (BN+ReLU), a 1×1 spatial-attention
  gate (`Conv 256→128→1` + `Sigmoid`) multiplied back in, then concatenated
  `AdaptiveAvgPool2d((1,1))` + `AdaptiveMaxPool2d((1,1))` → `512` → FC → `feature_dim`.
  Already shape-robust (global pooling), kept as in LeFlexiTac.
- **`TactileTokenEncoder`** — wraps a backbone with `feature_dim = dim_model`. When
  `n_tokens > 1`, a `Linear(dim_model, n_tokens * dim_model)` splits the embedding into
  `n_tokens` tokens; when `n_tokens == 1`, unsqueezes. Forward input `(B, H, W)` (casts to
  the module's float dtype, since dataset tactile is `int16`), output
  `(B, n_tokens, dim_model)`.

Model wiring:

- In `ACT.__init__`, when `config.use_tactile`, build
  `self.tactile_encoders = nn.ModuleDict()` — one `TactileTokenEncoder` per tactile key,
  each constructed with that feature's `shape` as `input_shape`. ModuleDict keys sanitize
  `.` → `_`; a stable iteration order follows `config.tactile_features` insertion order.
- Increase the transformer-encoder 1D token count:
  `n_1d_tokens += len(config.tactile_features) * config.n_tactile_tokens`, so
  `encoder_1d_feature_pos_embed` (learned) covers the tactile tokens.
- In `ACT.forward`, after the latent / robot_state / env_state tokens and **before** the
  image-feature tokens, loop over `config.tactile_features` in order:
  `tokens = self.tactile_encoders[sanitize(key)](batch[key])` → append each of the
  `n_tactile_tokens` tokens to `encoder_in_tokens`. This ordering aligns tactile tokens
  with the trailing rows of `encoder_1d_feature_pos_embed` (the 1D pos-embed list is built
  in full up front, then extended by image 2D pos-embeds; 1D token append order must match
  its row order: latent, robot_state, env_state, tactile…).
- **Robust batch_size / device.** Base ACT derives `batch_size` from `OBS_IMAGES` or
  `OBS_ENV_STATE`, and the no-VAE latent-zeros path uses `batch[OBS_STATE].device`. Extend
  both so a tactile-only ACT (no images, no env_state, no state) works: fall back to the
  first tactile key for batch size, and derive the zeros device from any present input
  tensor rather than assuming `OBS_STATE`.

### Data flow (training + inference)

1. Dataset feature `observation.tactile.sensor_1` `(rows, cols)` int16 → typed
   `FeatureType.TACTILE` by `dataset_to_policy_features`.
2. Pre-processing pipeline normalizes it `MEAN_STD` per cell using dataset stats; other
   generic steps batch it and move it to device. It stays in `batch` under its own key
   (unlike images, which ACT collects into an `OBS_IMAGES` list).
3. `ACT.forward` runs each tactile key through its encoder → `n_tactile_tokens` tokens →
   appended to the transformer-encoder input sequence with learned 1D positional
   embeddings. Loss / action head unchanged.

## Error handling

- `use_tactile=True` with no `FeatureType.TACTILE` inputs → `ValueError` at config init.
- Invalid `tactile_encoder_type` → `ValueError` at config init.
- Tactile stats missing from the dataset while `MEAN_STD` is requested → surfaced by the
  existing normalizer error path (unchanged behavior).

## Testing

One focused, parametrized test (single vs. two sensors × `cnn` / `attention`) in a new
`tests/policies/test_act_tactile.py`:

- Build an ACT config with synthetic `observation.tactile.*` input features (mix of grid
  shapes for the multi-sensor case, e.g. `6x6` and `4x8`) plus the usual state/action (and
  optionally an image) features, `use_tactile=True`.
- Run a training forward pass: assert the returned loss is finite and `l1_loss` is present.
- Run `predict_action_chunk`: assert action shape `(B, chunk_size, action_dim)`.
- Assert the transformer-encoder input sequence length increased by exactly
  `n_sensors * n_tactile_tokens` versus the same config with `use_tactile=False`.

Keep it in a single parametrized test rather than many separate functions.

## Acceptance criteria

- ACT trains and runs inference with one or more tactile sensors of differing shapes.
- With `use_tactile=False` (default), ACT behavior and parameters are byte-for-byte
  unchanged; TACTILE features are ignored.
- Tactile is normalized `MEAN_STD` per cell.
- No other policy's behavior changes.
- New/changed code passes `pre-commit` (ruff, typos) and the new test passes.

## Files touched

- `src/lerobot/configs/types.py`
- `src/lerobot/utils/constants.py`
- `src/lerobot/utils/feature_utils.py`
- `src/lerobot/policies/act/configuration_act.py`
- `src/lerobot/policies/act/modeling_act.py`
- `tests/policies/test_act_tactile.py` (new ACT tactile test)
