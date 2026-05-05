#!/usr/bin/env python3
"""
Shared utilities for all XAI scripts in this directory.

Provides:
  - Lerobot stub installation (_install_lerobot_stubs)
  - Source package registration (_register_source_package)
  - Florence-2 vision tower loading (load_vision_tower)
  - Image preprocessing (load_image_pil, pil_to_tensor, resize_with_pad)
  - VRAM reporting (report_vram)

All scripts in xai/ import from this module instead of duplicating setup.
"""

import importlib.util
import json
import os
import sys
import types

import torch
import torch.nn.functional as F

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(UTILS_DIR)
SOURCE_DIR = os.environ.get(
    "XVLA_SOURCE_DIR",
    os.path.join(PROJECT_DIR, "XVLA original source"),
)
MODEL_DIR = os.environ.get(
    "XVLA_MODEL_DIR",
    os.path.join(PROJECT_DIR, "xvla-pouring-0.1"),
)
OUTPUT_DIR = os.path.join(UTILS_DIR, "outputs")
PACKAGE_NAME = "xvla_src"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
DAVIT_INPUT_SIZE = (224, 224)
CAMERA_INPUT_SIZE = (256, 256)


def _install_lerobot_stubs() -> None:
    """Registers no-op stub modules for every lerobot.* symbol used by XVLA sources."""

    def _make(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    _make("lerobot")

    cfg = _make("lerobot.configs")

    class _PreTrainedConfig:
        pass

    class _FeatureType:
        VISUAL = "VISUAL"
        STATE = "STATE"
        ACTION = "ACTION"

    class _NormalizationMode:
        IDENTITY = "IDENTITY"
        MEAN_STD = "MEAN_STD"

    _PreTrainedConfig.register_subclass = staticmethod(lambda key: lambda cls: cls)

    cfg.PreTrainedConfig = _PreTrainedConfig
    cfg.PolicyFeature = type("PolicyFeature", (), {})
    cfg.FeatureType = _FeatureType
    cfg.NormalizationMode = _NormalizationMode
    cfg.PipelineFeatureType = _FeatureType

    _make("lerobot.utils")
    ucc = _make("lerobot.utils.constants")
    ucc.ACTION = "action"
    ucc.OBS_LANGUAGE_TOKENS = "observation.language_tokens"
    ucc.OBS_STATE = "observation.state"
    ucc.OBS_IMAGES = "observation.images"
    ucc.OBS_PREFIX = "observation."
    ucc.IMAGENET_STATS = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    ucc.POLICY_PREPROCESSOR_DEFAULT_NAME = "policy_preprocessor"
    ucc.POLICY_POSTPROCESSOR_DEFAULT_NAME = "policy_postprocessor"

    uiu = _make("lerobot.utils.import_utils")
    uiu._transformers_available = True
    uiu.require_package = lambda *a, **kw: None

    _make("lerobot.utils.utils")
    sys.modules["lerobot.utils"].populate_queues = lambda queues, batch, **kw: queues

    opt = _make("lerobot.optim")
    opt.CosineDecayWithWarmupSchedulerConfig = object
    opt.XVLAAdamWConfig = object

    pre = _make("lerobot.pretrained")

    class _PreTrainedPolicy(torch.nn.Module):
        def __init__(self, config, **kw):
            super().__init__()
            self.config = config

    pre.PreTrainedPolicy = _PreTrainedPolicy
    pre.T = None
    _make("lerobot.policies").PreTrainedPolicy = _PreTrainedPolicy

    proc = _make("lerobot.processor")

    class _NoOpStep:
        def __init__(self, *a, **kw): pass
        def __call__(self, x): return x
        def transform_features(self, f): return f
        def get_config(self): return {}

    class _NoOpPipeline:
        def __init__(self, *a, **kw): pass
        def __class_getitem__(cls, item): return cls

    class _StepRegistry:
        @staticmethod
        def register(name):
            return lambda cls: cls

    for _name in [
        "AddBatchDimensionProcessorStep", "DeviceProcessorStep",
        "NormalizerProcessorStep", "ObservationProcessorStep",
        "PolicyAction", "ProcessorStep", "RenameObservationsProcessorStep",
        "TokenizerProcessorStep", "UnnormalizerProcessorStep",
    ]:
        setattr(proc, _name, _NoOpStep)

    proc.PolicyProcessorPipeline = _NoOpPipeline
    proc.ProcessorStepRegistry = _StepRegistry
    proc.policy_action_to_transition = lambda x: x
    proc.transition_to_policy_action = lambda x: x

    typ = _make("lerobot.types")

    class _EnvTransition(dict):
        pass

    class _TransitionKey:
        OBSERVATION = "observation"
        ACTION = "action"
        COMPLEMENTARY_DATA = "complementary_data"

    typ.EnvTransition = _EnvTransition
    typ.TransitionKey = _TransitionKey


def _register_source_package() -> None:
    """
    Registers XVLA original source/ as xvla_src nested under a fake parent (xvla_parent)
    so that relative imports inside modeling_xvla.py resolve without errors.
    """
    parent_name = "xvla_parent"
    parent_pkg = types.ModuleType(parent_name)
    parent_pkg.__path__ = []
    parent_pkg.__package__ = parent_name
    sys.modules[parent_name] = parent_pkg

    pre_stub = types.ModuleType(f"{parent_name}.pretrained")
    pre_stub.__package__ = parent_name

    class _PreTrainedPolicy(torch.nn.Module):
        def __init__(self, config, **kw):
            super().__init__()
            self.config = config

    pre_stub.PreTrainedPolicy = _PreTrainedPolicy
    pre_stub.T = None
    sys.modules[f"{parent_name}.pretrained"] = pre_stub

    utils_stub = types.ModuleType(f"{parent_name}.utils")
    utils_stub.__package__ = parent_name
    utils_stub.populate_queues = lambda queues, batch, **kw: queues
    sys.modules[f"{parent_name}.utils"] = utils_stub
    setattr(parent_pkg, "utils", utils_stub)

    full_pkg_name = f"{parent_name}.{PACKAGE_NAME}"
    pkg = types.ModuleType(full_pkg_name)
    pkg.__path__ = [SOURCE_DIR]
    pkg.__package__ = full_pkg_name
    pkg.__spec__ = importlib.util.spec_from_file_location(
        full_pkg_name,
        os.path.join(SOURCE_DIR, "__init__.py"),
        submodule_search_locations=[SOURCE_DIR],
    )
    sys.modules[full_pkg_name] = pkg
    sys.modules[PACKAGE_NAME] = pkg
    setattr(parent_pkg, PACKAGE_NAME, pkg)

    py_files = [
        f[:-3] for f in os.listdir(SOURCE_DIR)
        if f.endswith(".py") and f != "__init__.py"
    ]
    sub_specs: dict[str, importlib.machinery.ModuleSpec] = {}
    for mod_name in py_files:
        nested = f"{full_pkg_name}.{mod_name}"
        flat = f"{PACKAGE_NAME}.{mod_name}"
        path = os.path.join(SOURCE_DIR, f"{mod_name}.py")
        spec = importlib.util.spec_from_file_location(nested, path)
        sub = importlib.util.module_from_spec(spec)
        sub.__package__ = full_pkg_name
        sub.__name__ = nested
        sys.modules[nested] = sub
        sys.modules[flat] = sub
        sub_specs[mod_name] = spec

    SKIP_EXEC = {"modeling_xvla", "processor_xvla"}
    load_order = [
        "utils",
        "configuration_florence2",
        "configuration_xvla",
        "action_hub",
        "soft_transformer",
        "modeling_florence2",
    ]
    remaining = [m for m in py_files if m not in load_order and m not in SKIP_EXEC]
    for mod_name in load_order + remaining:
        if mod_name in sub_specs:
            try:
                sub_specs[mod_name].loader.exec_module(
                    sys.modules[f"{PACKAGE_NAME}.{mod_name}"]
                )
            except Exception as exc:
                print(f"  [warn] could not load xvla_src.{mod_name}: {exc}")


_install_lerobot_stubs()
_register_source_package()

from xvla_src.configuration_florence2 import Florence2Config                 # noqa: E402
from xvla_src.modeling_florence2 import Florence2ForConditionalGeneration    # noqa: E402


def resize_with_pad(img: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Letterbox-resize maintaining aspect ratio; padding value = 0."""
    if img.ndim != 4:
        raise ValueError(f"Expected (B,C,H,W), got {img.shape}")
    ch, cw = img.shape[2], img.shape[3]
    if ch == height and cw == width:
        return img
    ratio = max(cw / width, ch / height)
    rh, rw = int(ch / ratio), int(cw / ratio)
    resized = F.interpolate(img, size=(rh, rw), mode="bilinear", align_corners=False)
    ph, pw = max(0, height - rh), max(0, width - rw)
    return F.pad(resized, (pw, 0, ph, 0), value=0.0)


def load_image_pil(path: str):
    """Returns a PIL Image (RGB) from the given path."""
    from PIL import Image
    return Image.open(path).convert("RGB")


def pil_to_tensor(pil_img, device: torch.device) -> torch.Tensor:
    """Converts a PIL RGB image to a normalized, letterboxed (1, 3, 224, 224) float32 tensor."""
    import numpy as np
    arr = torch.from_numpy(np.array(pil_img)).float() / 255.0
    arr = arr.permute(2, 0, 1).unsqueeze(0)
    mean = IMAGENET_MEAN.view(1, 3, 1, 1)
    std = IMAGENET_STD.view(1, 3, 1, 1)
    arr = (arr - mean) / std
    return resize_with_pad(arr, DAVIT_INPUT_SIZE[0], DAVIT_INPUT_SIZE[1]).to(device)


def load_florence2_config() -> Florence2Config:
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        raw = json.load(f)
    return Florence2Config(**raw["florence_config"])


def load_vision_tower(
    device: torch.device,
    keep_language_encoder: bool = False,
) -> Florence2ForConditionalGeneration:
    """
    Loads the Florence-2 model from xvla-pouring-0.1/model.safetensors.

    By default the language decoder is deleted to save VRAM.
    Set keep_language_encoder=True for text-guided attention extraction.
    Returns the model in eval mode on the specified device.
    """
    config = load_florence2_config()
    model = Florence2ForConditionalGeneration(config)

    if not keep_language_encoder and hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "decoder"):
            del lm.model.decoder
        if hasattr(lm, "lm_head"):
            del lm.lm_head

    import safetensors.torch as st
    state_dict = st.load_file(os.path.join(MODEL_DIR, "model.safetensors"))

    prefix = "model.vlm."
    florence_sd = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    enc_key = "language_model.model.encoder.embed_tokens.weight"
    shared_key = "language_model.model.shared.weight"
    if enc_key in florence_sd and shared_key not in florence_sd:
        florence_sd[shared_key] = florence_sd[enc_key]

    missing, _ = model.load_state_dict(florence_sd, strict=False)
    non_decoder_missing = [k for k in missing if "decoder" not in k and "lm_head" not in k]
    if non_decoder_missing:
        print(f"  [WARN] Non-decoder missing keys: {non_decoder_missing[:5]}")

    model.vision_tower = model.vision_tower.to(dtype=torch.float32)
    model.image_projection.data = model.image_projection.data.to(dtype=torch.float32)
    model.image_proj_norm = model.image_proj_norm.to(dtype=torch.float32)

    if keep_language_encoder:
        model.language_model.model.encoder.to(dtype=torch.float32)

    return model.to(device).eval()


def report_vram(device: torch.device, label: str = "") -> None:
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory / (1024 ** 3)
    free, _ = torch.cuda.mem_get_info(device)
    free_gb = free / (1024 ** 3)
    tag = f" [{label}]" if label else ""
    print(f"VRAM{tag}  alloc={alloc:.2f}GB  reserved={reserved:.2f}GB  "
          f"free={free_gb:.2f}GB  total={total:.1f}GB")


def ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR
