#!/usr/bin/env python3
"""
Smoke-test for loading the fine-tuned XVLA model and running a single
forward pass through the DaViT vision encoder.

Prints feature-map shapes at every DaViT stage and basic VRAM statistics.
Does NOT require the LeRobot framework.

Usage (on the Linux server, from the xai/ directory):
    python load_model_test.py
"""

import json
import os
import sys
import time
import importlib.util
import types

import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SOURCE_DIR = os.path.join(PROJECT_DIR, "XVLA original source")
MODEL_DIR = os.path.join(PROJECT_DIR, "xvla-pouring-0.1")
TEST_IMAGE_PATH = os.path.join(PROJECT_DIR, "test_image.jpg")
PACKAGE_NAME = "xvla_src"


def _install_lerobot_stubs() -> None:
    """Registers lightweight stub modules for every lerobot.* import in the XVLA source."""
    def _make(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    _make("lerobot")

    cfg = _make("lerobot.configs")

    class _PreTrainedConfig:
        pass

    class _PolicyFeature:
        pass

    class _FeatureType:
        VISUAL = "VISUAL"
        STATE = "STATE"
        ACTION = "ACTION"

    class _NormalizationMode:
        IDENTITY = "IDENTITY"
        MEAN_STD = "MEAN_STD"

    cfg.PreTrainedConfig = _PreTrainedConfig
    cfg.PolicyFeature = _PolicyFeature
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
    ucc.IMAGENET_STATS = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    ucc.POLICY_PREPROCESSOR_DEFAULT_NAME = "policy_preprocessor"
    ucc.POLICY_POSTPROCESSOR_DEFAULT_NAME = "policy_postprocessor"

    uiu = _make("lerobot.utils.import_utils")
    uiu._transformers_available = True

    def _require_package(pkg, extra=None):
        pass

    uiu.require_package = _require_package

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

    pkg_parent = _make("lerobot.policies")
    pkg_parent.PreTrainedPolicy = _PreTrainedPolicy

    _make("lerobot.utils.utils")

    def _populate_queues(queues, batch, exclude_keys=None):
        return queues

    sys.modules["lerobot.utils"].populate_queues = _populate_queues

    proc = _make("lerobot.processor")

    class _NoOpStep:
        def __init__(self, *a, **kw):
            pass
        def __call__(self, x):
            return x
        def transform_features(self, f):
            return f
        def get_config(self):
            return {}

    class _NoOpPipeline:
        def __init__(self, *a, **kw):
            pass
        def __class_getitem__(cls, item):
            return cls

    class _ProcessorStepRegistry:
        @staticmethod
        def register(name):
            def _wrap(cls):
                return cls
            return _wrap

    _step_classes = [
        "AddBatchDimensionProcessorStep",
        "DeviceProcessorStep",
        "NormalizerProcessorStep",
        "ObservationProcessorStep",
        "PolicyAction",
        "ProcessorStep",
        "RenameObservationsProcessorStep",
        "TokenizerProcessorStep",
        "UnnormalizerProcessorStep",
    ]
    for _cls_name in _step_classes:
        setattr(proc, _cls_name, _NoOpStep)

    proc.PolicyProcessorPipeline = _NoOpPipeline
    proc.ProcessorStepRegistry = _ProcessorStepRegistry

    def _identity(x):
        return x

    proc.policy_action_to_transition = _identity
    proc.transition_to_policy_action = _identity

    typ = _make("lerobot.types")

    class _EnvTransition(dict):
        pass

    class _TransitionKey:
        OBSERVATION = "observation"
        ACTION = "action"
        COMPLEMENTARY_DATA = "complementary_data"

    typ.EnvTransition = _EnvTransition
    typ.TransitionKey = _TransitionKey

    def _register_subclass(key):
        def _wrap(cls):
            return cls
        return _wrap

    cfg.PreTrainedConfig.register_subclass = staticmethod(_register_subclass)


_install_lerobot_stubs()


def _register_source_package() -> None:
    """Registers XVLA original source/ as the package xvla_src."""
    parent_name = "xvla_parent"
    parent_pkg = types.ModuleType(parent_name)
    parent_pkg.__path__ = []
    parent_pkg.__package__ = parent_name
    sys.modules[parent_name] = parent_pkg

    pretrained_stub = types.ModuleType(f"{parent_name}.pretrained")
    pretrained_stub.__package__ = parent_name

    class _PreTrainedPolicy(torch.nn.Module):
        def __init__(self, config, **kw):
            super().__init__()
            self.config = config

    pretrained_stub.PreTrainedPolicy = _PreTrainedPolicy
    pretrained_stub.T = None
    sys.modules[f"{parent_name}.pretrained"] = pretrained_stub

    full_package_name = f"{parent_name}.{PACKAGE_NAME}"
    pkg = types.ModuleType(full_package_name)
    pkg.__path__ = [SOURCE_DIR]
    pkg.__package__ = full_package_name
    pkg.__spec__ = importlib.util.spec_from_file_location(
        full_package_name,
        os.path.join(SOURCE_DIR, "__init__.py"),
        submodule_search_locations=[SOURCE_DIR],
    )
    sys.modules[full_package_name] = pkg
    sys.modules[PACKAGE_NAME] = pkg
    setattr(parent_pkg, PACKAGE_NAME, pkg)

    py_files = [
        f[:-3] for f in os.listdir(SOURCE_DIR)
        if f.endswith(".py") and f != "__init__.py"
    ]
    sub_specs = {}
    for mod_name in py_files:
        nested_full = f"{full_package_name}.{mod_name}"
        flat_full = f"{PACKAGE_NAME}.{mod_name}"
        path = os.path.join(SOURCE_DIR, f"{mod_name}.py")
        spec = importlib.util.spec_from_file_location(nested_full, path)
        sub_mod = importlib.util.module_from_spec(spec)
        sub_mod.__package__ = full_package_name
        sub_mod.__name__ = nested_full
        sys.modules[nested_full] = sub_mod
        sys.modules[flat_full] = sub_mod
        sub_specs[mod_name] = spec

    load_order = [
        "utils",
        "configuration_florence2",
        "configuration_xvla",
        "action_hub",
        "soft_transformer",
        "modeling_florence2",
        "modeling_xvla",
        "processor_xvla",
    ]
    remaining = [m for m in py_files if m not in load_order]
    for mod_name in load_order + remaining:
        if mod_name in sub_specs:
            try:
                sub_specs[mod_name].loader.exec_module(
                    sys.modules[f"{PACKAGE_NAME}.{mod_name}"]
                )
            except Exception as exc:
                print(f"  [warn] could not load xvla_src.{mod_name}: {exc}")


_register_source_package()

from xvla_src.configuration_florence2 import Florence2Config       # noqa: E402
from xvla_src.modeling_florence2 import Florence2ForConditionalGeneration  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
TARGET_SIZE = (224, 224)
INPUT_SIZE = (256, 256)


def resize_with_pad(img: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Letterbox-resize: maintain aspect ratio, pad with zeros."""
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, got {img.shape}")
    ch, cw = img.shape[2], img.shape[3]
    if ch == height and cw == width:
        return img
    ratio = max(cw / width, ch / height)
    rh, rw = int(ch / ratio), int(cw / ratio)
    resized = F.interpolate(img, size=(rh, rw), mode="bilinear", align_corners=False)
    ph, pw = max(0, height - rh), max(0, width - rw)
    return F.pad(resized, (pw, 0, ph, 0), value=0.0)


def load_and_preprocess_image(path: str, device: torch.device) -> torch.Tensor:
    """Loads a JPEG/PNG and returns a normalized (1,3,224,224) tensor."""
    from PIL import Image
    import numpy as np

    img = Image.open(path).convert("RGB")
    arr = torch.from_numpy(np.array(img)).float() / 255.0
    arr = arr.permute(2, 0, 1).unsqueeze(0)

    mean = IMAGENET_MEAN.view(1, 3, 1, 1)
    std = IMAGENET_STD.view(1, 3, 1, 1)
    arr = (arr - mean) / std
    arr = resize_with_pad(arr, TARGET_SIZE[0], TARGET_SIZE[1])

    return arr.to(device)


class _StageHook:
    """Captures the output tensor of a DaViT stage via a forward hook."""

    def __init__(self, name: str):
        self.name = name
        self.output = None
        self._handle = None

    def attach(self, module: torch.nn.Module) -> "_StageHook":
        self._handle = module.register_forward_hook(self._hook)
        return self

    def _hook(self, module, inputs, output):
        if isinstance(output, tuple):
            self.output = output[0].detach()
        else:
            self.output = output.detach()

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def load_florence2_config(model_dir: str) -> Florence2Config:
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        raw = json.load(f)
    return Florence2Config(**raw["florence_config"])


def load_vision_tower(model_dir: str, device: torch.device) -> Florence2ForConditionalGeneration:
    """Loads the vision-tower sub-graph of Florence-2; drops the language decoder."""
    print("\nLoading Florence-2 config …")
    config = load_florence2_config(model_dir)

    print("Instantiating Florence-2 model (no weights yet) …")
    model = Florence2ForConditionalGeneration(config)

    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "decoder"):
            del lm.model.decoder
        if hasattr(lm, "lm_head"):
            del lm.lm_head

    print("Loading safetensors weights …")
    import safetensors.torch as st

    safetensors_path = os.path.join(model_dir, "model.safetensors")
    state_dict = st.load_file(safetensors_path)

    prefix = "model.vlm."
    florence_sd = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    enc_key = "language_model.model.encoder.embed_tokens.weight"
    shared_key = "language_model.model.shared.weight"
    if enc_key in florence_sd and shared_key not in florence_sd:
        florence_sd[shared_key] = florence_sd[enc_key]

    missing, unexpected = model.load_state_dict(florence_sd, strict=False)
    print(f"  Missing keys  : {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if missing:
        non_decoder = [k for k in missing if "decoder" not in k and "lm_head" not in k]
        if non_decoder:
            print(f"  [WARN] Non-decoder missing keys: {non_decoder[:5]}")

    model.vision_tower = model.vision_tower.to(dtype=torch.float32)
    model.image_projection.data = model.image_projection.data.to(dtype=torch.float32)
    model.image_proj_norm = model.image_proj_norm.to(dtype=torch.float32)

    model = model.to(device)
    model.eval()
    return model


def run_forward_pass_test(model: Florence2ForConditionalGeneration,
                          image_tensor: torch.Tensor) -> None:
    """Runs one forward pass through the DaViT vision encoder and reports shapes and timing."""
    vision_tower = model.vision_tower

    hooks = []
    for i, (conv, block) in enumerate(zip(vision_tower.convs, vision_tower.blocks)):
        h_conv = _StageHook(f"conv_embed_{i}")
        h_block = _StageHook(f"block_{i}")
        h_conv.attach(conv)
        h_block.attach(block)
        hooks.extend([h_conv, h_block])

    print("\n--- DaViT forward pass ---")
    print(f"Input image : {image_tensor.shape}  dtype={image_tensor.dtype}")

    t0 = time.perf_counter()
    with torch.no_grad():
        feat = vision_tower.forward_features_unpool(image_tensor)
    torch.cuda.synchronize(image_tensor.device)
    t1 = time.perf_counter()

    print(f"DaViT forward pass completed in {(t1 - t0) * 1000:.1f} ms")
    print(f"\nFinal feature map : {feat.shape}  (B, N_tokens, C)")

    print("\nPer-stage feature shapes:")
    for h in hooks:
        if h.output is not None:
            shape = h.output.shape
            spatial_tokens = shape[1] if len(shape) == 3 else "?"
            n_side = int(spatial_tokens ** 0.5) if isinstance(spatial_tokens, int) else "?"
            print(f"  {h.name:20s}: {tuple(shape)}   "
                  f"≈ {n_side}×{n_side} spatial tokens")

    for h in hooks:
        h.remove()

    print("\n--- Full _encode_image pipeline ---")
    t2 = time.perf_counter()
    with torch.no_grad():
        encoded = model._encode_image(image_tensor)
    torch.cuda.synchronize(image_tensor.device)
    t3 = time.perf_counter()
    print(f"Encoded image tokens : {encoded.shape}  (B, N_tokens, projection_dim=1024)")
    print(f"Completed in {(t3 - t2) * 1000:.1f} ms")


def report_vram(device: torch.device) -> None:
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory / (1024 ** 3)
    free, _ = torch.cuda.mem_get_info(device)
    free_gb = free / (1024 ** 3)
    print(f"\nVRAM  allocated={alloc:.2f} GB  reserved={reserved:.2f} GB  "
          f"free={free_gb:.2f} GB  total={total:.1f} GB")


def main() -> int:
    print("=" * 60)
    print("XVLA XAI — Model Load & Vision Encoder Smoke Test")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[FAIL] CUDA not available. This test requires a GPU.")
        return 1
    device = torch.device("cuda")
    print(f"\nDevice : {torch.cuda.get_device_name(device)}")
    report_vram(device)

    t_start = time.perf_counter()
    try:
        model = load_vision_tower(MODEL_DIR, device)
    except Exception as e:
        print(f"[FAIL] Model load error: {e}")
        raise
    t_loaded = time.perf_counter()
    print(f"\nModel loaded in {t_loaded - t_start:.1f} s")
    report_vram(device)

    total_params = sum(p.numel() for p in model.parameters())
    vision_params = sum(p.numel() for p in model.vision_tower.parameters())
    print(f"\nParameters (vision tower only) : {vision_params / 1e6:.1f} M")
    print(f"Parameters (full Florence-2)   : {total_params / 1e6:.1f} M")

    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"\n[WARN] test_image.jpg not found at {TEST_IMAGE_PATH}")
        print("       Creating a random synthetic image (1, 3, 256, 256) instead.")
        img_tensor = torch.rand(1, 3, 256, 256, device=device)
        img_tensor = resize_with_pad(img_tensor, TARGET_SIZE[0], TARGET_SIZE[1])
    else:
        print(f"\nLoading test image: {TEST_IMAGE_PATH}")
        img_tensor = load_and_preprocess_image(TEST_IMAGE_PATH, device)

    print(f"Image tensor: {img_tensor.shape}  dtype={img_tensor.dtype}")
    print(f"  Value range: [{img_tensor.min().item():.3f}, {img_tensor.max().item():.3f}]")

    run_forward_pass_test(model, img_tensor)
    report_vram(device)

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED — Vision encoder is working correctly.")
    print("Next step: run xai_feature_maps.py")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
