# ACT Tactile Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional tactile sensing to the ACT policy, encoding each tactile sensor grid into a configurable number of transformer-encoder tokens via a per-sensor CNN.

**Architecture:** Introduce a first-class `FeatureType.TACTILE`; `observation.tactile.*` dataset keys are typed TACTILE and normalized per-cell with `MEAN_STD`. In ACT, an opt-in `use_tactile` flag builds one `TactileTokenEncoder` (CNN or attention backbone) per tactile key; each emits `n_tactile_tokens` tokens that are appended to the transformer-encoder input sequence (after latent/state/env, before image tokens), with learned 1D positional embeddings. The VAE encoder and action head are untouched.

**Tech Stack:** Python 3.12+, PyTorch, draccus dataclass configs, pytest.

## Global Constraints

- Environment management is **conda** (never `uv`/`pip` directly). Run commands inside the project's conda env (e.g. `conda activate lerobot` first, or prefix with `conda run -n lerobot`).
- With `use_tactile=False` (default), ACT behavior and parameters must be byte-for-byte unchanged; TACTILE features are ignored by all other policies.
- Tactile grids are 2D `(rows, cols)`, dtype `int16`, raw 12-bit ADC (0–4095); keys follow `observation.tactile.<name>` (grabette: `observation.tactile.sensor_<addr>`).
- Tactile normalization is `MEAN_STD` per cell.
- `n_tactile_tokens` default is `4`; `tactile_encoder_type ∈ {"cnn", "attention"}`.
- New/changed code must pass `pre-commit run --all-files` (ruff, typos) and the new tests.
- Keep changes minimal; do not refactor or reformat surrounding code.

---

### Task 1: First-class tactile feature type

**Files:**
- Modify: `src/lerobot/configs/types.py` (add `TACTILE` to `FeatureType`)
- Modify: `src/lerobot/utils/constants.py` (add `OBS_TACTILE`)
- Modify: `src/lerobot/utils/feature_utils.py` (map `observation.tactile.*` → `FeatureType.TACTILE` in `dataset_to_policy_features`)
- Test: `tests/policies/test_act_tactile.py` (new file; typing test added here, reused by later tasks)

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `FeatureType.TACTILE` enum member (value `"TACTILE"`).
  - `OBS_TACTILE = "observation.tactile"` string constant.
  - `dataset_to_policy_features(features)` now returns `PolicyFeature(type=FeatureType.TACTILE, shape=(rows, cols))` for keys starting with `OBS_TACTILE`.

- [ ] **Step 1: Write the failing test**

Create `tests/policies/test_act_tactile.py`:

```python
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.feature_utils import dataset_to_policy_features


def test_tactile_feature_typing():
    features = {
        "observation.state": {"dtype": "float32", "shape": (10,), "names": None},
        "observation.tactile.sensor_1": {"dtype": "int16", "shape": (6, 6), "names": ["rows", "columns"]},
        "action": {"dtype": "float32", "shape": (6,), "names": None},
    }
    policy_features = dataset_to_policy_features(features)
    assert policy_features["observation.tactile.sensor_1"].type is FeatureType.TACTILE
    assert policy_features["observation.tactile.sensor_1"].shape == (6, 6)
    assert policy_features["observation.state"].type is FeatureType.STATE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n lerobot python -m pytest tests/policies/test_act_tactile.py::test_tactile_feature_typing -v`
Expected: FAIL — `AttributeError: TACTILE` (enum member missing) or the tactile key typed as `STATE`.

- [ ] **Step 3: Add `TACTILE` to `FeatureType`**

In `src/lerobot/configs/types.py`, add the member after `LANGUAGE`:

```python
class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"
    REWARD = "REWARD"
    LANGUAGE = "LANGUAGE"
    TACTILE = "TACTILE"
```

- [ ] **Step 4: Add `OBS_TACTILE` constant**

In `src/lerobot/utils/constants.py`, add directly after the `OBS_STATE` line:

```python
OBS_STATE = OBS_STR + ".state"
OBS_TACTILE = OBS_STR + ".tactile"
```

- [ ] **Step 5: Map tactile keys in `dataset_to_policy_features`**

In `src/lerobot/utils/feature_utils.py`, update the import line:

```python
from .constants import ACTION, DEFAULT_FEATURES, OBS_ENV_STATE, OBS_STR, OBS_TACTILE
```

Then, inside `dataset_to_policy_features`, add a tactile branch **before** the generic `OBS_STR` branch (tactile keys also start with `observation.`):

```python
        elif key == OBS_ENV_STATE:
            type = FeatureType.ENV
        elif key.startswith(OBS_TACTILE):
            type = FeatureType.TACTILE
        elif key.startswith(OBS_STR):
            type = FeatureType.STATE
        elif key.startswith(ACTION):
            type = FeatureType.ACTION
        else:
            continue
```

- [ ] **Step 6: Run test to verify it passes**

Run: `conda run -n lerobot python -m pytest tests/policies/test_act_tactile.py::test_tactile_feature_typing -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/lerobot/configs/types.py src/lerobot/utils/constants.py src/lerobot/utils/feature_utils.py tests/policies/test_act_tactile.py
git commit -m "feat(act): add FeatureType.TACTILE and map observation.tactile.* features"
```

---

### Task 2: ACT tactile config surface

**Files:**
- Modify: `src/lerobot/policies/act/configuration_act.py`
- Test: `tests/policies/test_act_tactile.py` (add config tests)

**Interfaces:**
- Consumes: `FeatureType.TACTILE` (Task 1).
- Produces (on `ACTConfig`):
  - Fields: `use_tactile: bool = False`, `tactile_encoder_type: str = "cnn"`, `n_tactile_tokens: int = 4`, `tactile_dropout: float = 0.3`.
  - `normalization_mapping` default includes `"TACTILE": NormalizationMode.MEAN_STD`.
  - Property `tactile_features -> dict[str, PolicyFeature]` (TACTILE-typed input features, insertion order preserved).
  - Validation: `ValueError` if `use_tactile` with bad `tactile_encoder_type`; `ValueError` if `use_tactile` but no tactile features; `validate_features` accepts tactile as a standalone input.

- [ ] **Step 1: Write the failing tests**

Append to `tests/policies/test_act_tactile.py`:

```python
from lerobot.policies.act.configuration_act import ACTConfig


def _tactile_input_features(shapes):
    feats = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        f"{OBS_IMAGES}.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
    }
    for key, shape in shapes.items():
        feats[key] = PolicyFeature(type=FeatureType.TACTILE, shape=shape)
    return feats


def test_tactile_config_property_and_validation():
    input_features = _tactile_input_features(
        {"observation.tactile.sensor_1": (6, 6), "observation.tactile.sensor_2": (4, 8)}
    )
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        use_tactile=True,
    )
    assert list(config.tactile_features) == ["observation.tactile.sensor_1", "observation.tactile.sensor_2"]
    assert config.normalization_mapping["TACTILE"].value == "MEAN_STD"

    with pytest.raises(ValueError):
        ACTConfig(
            input_features=input_features,
            output_features=output_features,
            use_tactile=True,
            tactile_encoder_type="bogus",
        )

    with pytest.raises(ValueError):
        # use_tactile enabled but no tactile features present
        ACTConfig(
            input_features={
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
                f"{OBS_IMAGES}.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
            },
            output_features=output_features,
            use_tactile=True,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n lerobot python -m pytest tests/policies/test_act_tactile.py::test_tactile_config_property_and_validation -v`
Expected: FAIL — `TypeError: unexpected keyword argument 'use_tactile'`.

- [ ] **Step 3: Add the config fields, mapping entry, property, and validation**

In `src/lerobot/policies/act/configuration_act.py`:

Update the import to bring in `FeatureType` and `PolicyFeature`:

```python
from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.optim import AdamWConfig
```

Add `"TACTILE"` to the `normalization_mapping` default factory:

```python
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
            "TACTILE": NormalizationMode.MEAN_STD,
        }
    )
```

Add the tactile fields directly after the `temporal_ensemble_coeff` field:

```python
    # Inference.
    # Note: the value used in ACT when temporal ensembling is enabled is 0.01.
    temporal_ensemble_coeff: float | None = None

    # Tactile sensor configuration (optional; disabled by default).
    use_tactile: bool = False
    tactile_encoder_type: str = "cnn"  # "cnn" | "attention"
    n_tactile_tokens: int = 4
    tactile_dropout: float = 0.3
```

In `__post_init__`, after the existing checks, add:

```python
        if self.use_tactile and self.tactile_encoder_type not in ("cnn", "attention"):
            raise ValueError(
                f"`tactile_encoder_type` must be 'cnn' or 'attention'. Got {self.tactile_encoder_type}."
            )
        if self.use_tactile and not self.tactile_features:
            raise ValueError(
                "`use_tactile=True` but no tactile features were found in `input_features`. "
                "Tactile features are keys starting with 'observation.tactile.'."
            )
```

Add the property next to `image_features` / `env_state_feature` (place after `action_delta_indices`):

```python
    @property
    def tactile_features(self) -> dict[str, PolicyFeature]:
        if not self.input_features:
            return {}
        return {k: ft for k, ft in self.input_features.items() if ft.type is FeatureType.TACTILE}
```

Update `validate_features` to accept tactile as a standalone input:

```python
    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature and not self.tactile_features:
            raise ValueError(
                "You must provide at least one image, the environment state, or a tactile sensor among the inputs."
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n lerobot python -m pytest tests/policies/test_act_tactile.py -v`
Expected: PASS (typing test + config test)

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/policies/act/configuration_act.py tests/policies/test_act_tactile.py
git commit -m "feat(act): add tactile config fields, property, and validation"
```

---

### Task 3: Tactile encoder + model wiring

**Files:**
- Modify: `src/lerobot/policies/act/modeling_act.py`
- Test: `tests/policies/test_act_tactile.py` (add end-to-end forward test)

**Interfaces:**
- Consumes: `ACTConfig.use_tactile`, `ACTConfig.tactile_encoder_type`, `ACTConfig.n_tactile_tokens`, `ACTConfig.tactile_dropout`, `ACTConfig.tactile_features` (Task 2); `OBS_TACTILE` (Task 1).
- Produces:
  - Module-level classes `TactileCNN`, `TactileAttentionCNN`, `TactileTokenEncoder` in `modeling_act.py`.
  - `ACT.tactile_encoders: nn.ModuleDict` and `ACT.tactile_encoder_keys: list[str]` (ordered original keys) when `use_tactile`.
  - `TactileTokenEncoder.forward(x: (B, H, W)) -> (B, n_tokens, dim_model)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/policies/test_act_tactile.py`:

```python
from lerobot.policies.act.modeling_act import ACTPolicy

_B = 2
_CHUNK = 5
_ACTION_DIM = 6
_STATE_DIM = 10
_IMG = (3, 96, 96)
_SHAPES = {
    "single": {"observation.tactile.sensor_1": (6, 6)},
    "multi": {"observation.tactile.sensor_1": (6, 6), "observation.tactile.sensor_2": (4, 8)},
    "tactile_only": {"observation.tactile.sensor_1": (6, 6)},
}


def _encoder_seq_len_hook(policy):
    captured = {}

    def hook(module, args, kwargs):
        captured["seq"] = args[0].shape[0]

    handle = policy.model.encoder.register_forward_pre_hook(hook, with_kwargs=True)
    return captured, handle


def _build_batch(shapes, with_image_state):
    batch = {
        ACTION: torch.randn(_B, _CHUNK, _ACTION_DIM),
        "action_is_pad": torch.zeros(_B, _CHUNK, dtype=torch.bool),
    }
    if with_image_state:
        batch[OBS_STATE] = torch.randn(_B, _STATE_DIM)
        batch[f"{OBS_IMAGES}.cam"] = torch.rand(_B, *_IMG)
    for key, shape in shapes.items():
        batch[key] = torch.randint(0, 4096, (_B, *shape), dtype=torch.int16)
    return batch


@pytest.mark.parametrize("encoder_type", ["cnn", "attention"])
@pytest.mark.parametrize("scenario", ["single", "multi", "tactile_only"])
def test_act_tactile_forward(encoder_type, scenario):
    shapes = _SHAPES[scenario]
    with_image_state = scenario != "tactile_only"
    use_vae = with_image_state  # no proprioceptive state in tactile_only -> skip VAE encoder

    input_features = {}
    if with_image_state:
        input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(_STATE_DIM,))
        input_features[f"{OBS_IMAGES}.cam"] = PolicyFeature(type=FeatureType.VISUAL, shape=_IMG)
    for key, shape in shapes.items():
        input_features[key] = PolicyFeature(type=FeatureType.TACTILE, shape=shape)
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(_ACTION_DIM,))}

    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=_CHUNK,
        n_action_steps=_CHUNK,
        use_tactile=True,
        tactile_encoder_type=encoder_type,
        n_tactile_tokens=4,
        use_vae=use_vae,
    )
    policy = ACTPolicy(config)
    policy.train()

    batch = _build_batch(shapes, with_image_state)
    captured, handle = _encoder_seq_len_hook(policy)
    loss, loss_dict = policy.forward(batch)
    handle.remove()
    assert torch.isfinite(loss)
    assert "l1_loss" in loss_dict
    seq_with_tactile = captured["seq"]

    policy.eval()
    with torch.no_grad():
        actions = policy.predict_action_chunk(batch)
    assert actions.shape == (_B, _CHUNK, _ACTION_DIM)

    if with_image_state:
        base_features = {k: v for k, v in input_features.items() if v.type is not FeatureType.TACTILE}
        base_config = ACTConfig(
            input_features=base_features,
            output_features=output_features,
            chunk_size=_CHUNK,
            n_action_steps=_CHUNK,
            use_tactile=False,
            use_vae=use_vae,
        )
        base_policy = ACTPolicy(base_config)
        base_policy.train()
        base_batch = {k: v for k, v in batch.items() if not k.startswith("observation.tactile")}
        base_captured, base_handle = _encoder_seq_len_hook(base_policy)
        base_policy.forward(base_batch)
        base_handle.remove()
        expected_extra = len(shapes) * config.n_tactile_tokens
        assert seq_with_tactile - base_captured["seq"] == expected_extra
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n lerobot python -m pytest tests/policies/test_act_tactile.py::test_act_tactile_forward -v`
Expected: FAIL — `ImportError`/`AttributeError` (no tactile encoder / wiring) or a KeyError in `forward`.

- [ ] **Step 3: Add the tactile encoder modules**

In `src/lerobot/policies/act/modeling_act.py`, add these three classes immediately before `class ACT(nn.Module):`:

```python
class TactileCNN(nn.Module):
    """Conv backbone for a tactile grid. Outputs (B, feature_dim).

    Uses AdaptiveAvgPool2d before the FC head so it is valid for any grid size
    (grabette sensors are as small as 6x6, which collapses fixed /2 pooling).
    """

    def __init__(
        self,
        input_shape: tuple[int, int],
        feature_dim: int = 512,
        dropout: float = 0.3,
        pooled_size: tuple[int, int] = (2, 2),
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d(pooled_size)
        self.dropout = nn.Dropout(dropout)
        conv_output_dim = 128 * pooled_size[0] * pooled_size[1]
        self.fc1 = nn.Linear(conv_output_dim, 512)
        self.fc2 = nn.Linear(512, feature_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TactileAttentionCNN(nn.Module):
    """Conv backbone with spatial attention and global pooling. Outputs (B, feature_dim)."""

    def __init__(self, input_shape: tuple[int, int], feature_dim: int = 512, dropout: float = 0.4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.attention = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, feature_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x * self.attention(x)
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        x = torch.cat([avg_pool, max_pool], dim=1).flatten(1)
        x = self.fc(x)
        return x


class TactileTokenEncoder(nn.Module):
    """Encode a tactile grid (B, H, W) into (B, n_tokens, feature_dim)."""

    def __init__(
        self,
        encoder_type: str,
        input_shape: tuple[int, int],
        feature_dim: int,
        n_tokens: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.feature_dim = feature_dim
        if encoder_type == "cnn":
            self.backbone = TactileCNN(input_shape, feature_dim, dropout)
        elif encoder_type == "attention":
            self.backbone = TactileAttentionCNN(input_shape, feature_dim, dropout)
        else:
            raise ValueError(f"Unknown tactile encoder type: {encoder_type!r}. Choose 'cnn' or 'attention'.")
        self.token_proj = nn.Linear(feature_dim, n_tokens * feature_dim) if n_tokens > 1 else None

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(dtype=next(self.backbone.parameters()).dtype)  # tactile is int16 in the dataset
        feat = self.backbone(x)  # (B, feature_dim)
        if self.n_tokens == 1:
            return feat.unsqueeze(1)
        feat = self.token_proj(feat)
        return feat.view(feat.size(0), self.n_tokens, self.feature_dim)
```

- [ ] **Step 4: Import `OBS_TACTILE` in the model module**

In `src/lerobot/policies/act/modeling_act.py`, update the constants import:

```python
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE, OBS_TACTILE
```

- [ ] **Step 5: Build per-sensor encoders and grow the 1D positional embedding**

In `ACT.__init__`, replace the positional-embedding setup block:

```python
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
        # Transformer encoder positional embeddings.
        n_1d_tokens = 1  # for the latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)
```

with:

```python
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
        # Per-sensor tactile encoders. Each sensor grid is encoded to n_tactile_tokens tokens.
        self.tactile_encoder_keys: list[str] = []
        if self.config.use_tactile:
            self.tactile_encoders = nn.ModuleDict()
            for key, ft in self.config.tactile_features.items():
                self.tactile_encoder_keys.append(key)
                self.tactile_encoders[key.replace(".", "_")] = TactileTokenEncoder(
                    encoder_type=config.tactile_encoder_type,
                    input_shape=ft.shape,
                    feature_dim=config.dim_model,
                    n_tokens=config.n_tactile_tokens,
                    dropout=config.tactile_dropout,
                )
        # Transformer encoder positional embeddings.
        n_1d_tokens = 1  # for the latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        if self.config.use_tactile:
            n_1d_tokens += len(self.config.tactile_features) * config.n_tactile_tokens
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)
```

- [ ] **Step 6: Make `batch_size`/`device` resolution robust and append tactile tokens**

In `ACT.forward`, replace the `batch_size` line:

```python
        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]
```

with:

```python
        if OBS_IMAGES in batch:
            batch_size = batch[OBS_IMAGES][0].shape[0]
        elif OBS_ENV_STATE in batch:
            batch_size = batch[OBS_ENV_STATE].shape[0]
        elif self.config.use_tactile and self.tactile_encoder_keys:
            batch_size = batch[self.tactile_encoder_keys[0]].shape[0]
        else:
            batch_size = batch[OBS_STATE].shape[0]
```

Replace the non-VAE latent-zeros block:

```python
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[OBS_STATE].device
            )
```

with a version that does not assume `observation.state` is present:

```python
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            if self.config.robot_state_feature:
                latent_device = batch[OBS_STATE].device
            elif OBS_IMAGES in batch:
                latent_device = batch[OBS_IMAGES][0].device
            elif OBS_ENV_STATE in batch:
                latent_device = batch[OBS_ENV_STATE].device
            else:
                latent_device = batch[self.tactile_encoder_keys[0]].device
            # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
            latent_sample = torch.zeros(
                [batch_size, self.config.latent_dim], dtype=torch.float32
            ).to(latent_device)
```

Then append tactile tokens after the env-state token and **before** the image block. Locate:

```python
        # Environment state token.
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
```

and insert the tactile loop between them:

```python
        # Environment state token.
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        # Tactile tokens. Each sensor is encoded to n_tactile_tokens tokens; appended in
        # the same order as the trailing rows of encoder_1d_feature_pos_embed.
        if self.config.use_tactile:
            for key in self.tactile_encoder_keys:
                tactile_tokens = self.tactile_encoders[key.replace(".", "_")](batch[key])  # (B, n_tokens, D)
                for i in range(self.config.n_tactile_tokens):
                    encoder_in_tokens.append(tactile_tokens[:, i])

        if self.config.image_features:
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `conda run -n lerobot python -m pytest tests/policies/test_act_tactile.py -v`
Expected: PASS (all parametrizations: single/multi/tactile_only × cnn/attention, plus the Task 1/2 tests).

- [ ] **Step 8: Run pre-commit and the broader ACT tests**

Run: `conda run -n lerobot pre-commit run --files src/lerobot/configs/types.py src/lerobot/utils/constants.py src/lerobot/utils/feature_utils.py src/lerobot/policies/act/configuration_act.py src/lerobot/policies/act/modeling_act.py tests/policies/test_act_tactile.py`
Run: `conda run -n lerobot python -m pytest tests/policies/test_policies.py -k act -v`
Expected: PASS / no lint errors.

- [ ] **Step 9: Commit**

```bash
git add src/lerobot/policies/act/modeling_act.py tests/policies/test_act_tactile.py
git commit -m "feat(act): encode tactile sensors into transformer tokens (optional)"
```

---

## Self-Review

**1. Spec coverage:**
- `FeatureType.TACTILE` → Task 1, Step 3.
- `OBS_TACTILE` + `dataset_to_policy_features` mapping → Task 1, Steps 4–5.
- `MEAN_STD` tactile normalization → Task 2, Step 3 (`normalization_mapping`).
- Config fields, `tactile_features` property, validation, `validate_features` update → Task 2.
- `TactileCNN` (adaptive pooling), `TactileAttentionCNN`, `TactileTokenEncoder` → Task 3, Step 3.
- Per-sensor `nn.ModuleDict`, `n_1d_tokens` growth → Task 3, Step 5.
- Forward integration (tactile after env_state, before images) + robust batch_size/device → Task 3, Step 6.
- Testing (single/multi/tactile-only × cnn/attention, loss finite, action shape, seq-length delta) → Task 3, Step 1.
- "Unchanged when disabled" → guaranteed by gating every change on `use_tactile`/`tactile_features`; verified by `test_policies.py -k act`.

**2. Placeholder scan:** No TBD/TODO-style placeholders introduced (the one pre-existing `TODO(rcadene...)` comment is retained verbatim from the original code, not added by us). All code steps include full code.

**3. Type consistency:** `tactile_encoder_keys` (list of original keys) and ModuleDict keys (`key.replace(".", "_")`) are used identically in `__init__` and `forward`. `TactileTokenEncoder(encoder_type, input_shape, feature_dim, n_tokens, dropout)` signature matches its construction call. `tactile_features` returns `dict[str, PolicyFeature]` and is iterated for both count and construction consistently.

