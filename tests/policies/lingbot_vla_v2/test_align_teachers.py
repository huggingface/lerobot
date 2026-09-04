# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Teacher-side wiring for the native-depth / DINO-video distillation branch.

Pins, with stub teachers (no weights, no GPU, no vendored deps installed):

1. ``DepthTeacherBundle.depth_targets`` — the ported ``get_depth_target``:
   first camera only, /255, (B, 256, 1024) bf16, no_grad semantics;
2. ``DepthTeacherBundle.video_targets`` — the ported ``get_video_target``
   return-shape contract (plain tensor / (patch, cls) tuple / dict with
   cls=None when use_cls_loss=false) and the [0,255] scaling;
3. ``LingbotVLAV2Policy._compute_align_targets`` — the trainer-block port:
   depth + future-depth + DINO-video bundle unpack into exactly the forward
   kwargs the model expects, and the actionable RuntimeErrors for the
   mis-wirings users will actually hit.

Real-weight integration (MoGe/MoRGBD/DINO from the HF cache) is gated at the
bottom of this file and normally skipped.
"""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from lerobot.policies.lingbot_vla_v2.teachers.depth_teachers import DepthTeacherBundle  # noqa: E402

B, N, C, H, W = 2, 1, 3, 224, 224


# ---------------------------- stub teachers ----------------------------


class _StubMoGe(torch.nn.Module):
    """Mimics MoGeModel.infer's return dict with a (B, 1, H, W) depth map."""

    def infer(self, images, resolution_level=None, num_tokens=None, apply_mask=None, **kwargs):
        depth = torch.rand(images.shape[0], 1, *images.shape[-2:], device=images.device) * 10.0
        return {"depth": depth}


class _StubMoRGBD(torch.nn.Module):
    """Mimics MDMModel.infer_feat: (feature map [B,C,h,w], cls_token)."""

    def infer_feat(
        self,
        images,
        depth_pred,
        depth_down_scale=None,
        resolution_level=None,
        num_tokens=None,
        enable_depth_mask=None,
    ):
        # 16x16 spatial grid x 1024 channels -> 256 tokens of dim 1024.
        feat = torch.randn(images.shape[0], 1024, 16, 16, device=images.device)
        cls = torch.randn(images.shape[0], 1024, device=images.device)
        return feat, cls


class _StubVideo(torch.nn.Module):
    """Mimics DinoVideoTeacher.get_future_feature's return variants."""

    mode = "patch"

    def get_future_feature(self, video, return_cls=False, return_current=False, current_index=None, fps=None):
        batch = video.shape[0]
        patch = torch.randn(batch, 256, 1024, device=video.device)
        if return_cls and return_current:
            return patch, torch.randn(batch, 1024), torch.randn(batch, 256, 1024), torch.randn(batch, 1024)
        if return_cls:
            return patch, torch.randn(batch, 1024)
        if return_current:
            return patch, torch.randn(batch, 256, 1024)
        return patch


def _bundle(video_mode: str = "patch") -> DepthTeacherBundle:
    video = _StubVideo()
    video.mode = video_mode
    return DepthTeacherBundle(
        moge=_StubMoGe(),
        morgbd=_StubMoRGBD(),
        video=video,
        device=torch.device("cpu"),
    )


def _pil(batch=B, cameras=N, height=H, width=W):
    return torch.rand(batch, cameras, C, height, width) * 255.0


# ---------------------------- depth targets ----------------------------


def test_depth_targets_first_camera_only_bf16_256x1024():
    targets = _bundle().depth_targets(_pil())
    assert targets.shape == (B, 256, 1024)
    assert targets.dtype == torch.bfloat16
    assert not targets.requires_grad


def test_depth_targets_ignore_extra_cameras():
    """Upstream consumes pil_images[:, :1] regardless of camera count."""
    targets = _bundle().depth_targets(_pil(cameras=3))
    assert targets.shape == (B, 256, 1024)


# ---------------------------- video targets ----------------------------


def _video_cfg(**overrides):
    cfg = {"input_size": 256, "num_future_frames": 1, "use_warmup_frame": True, "use_patch_loss": True}
    cfg.update(overrides)
    return cfg


def test_video_targets_plain_tensor_without_cls_or_current():
    out = _bundle().video_targets(_pil(height=256, width=256), _pil(height=256, width=256), _video_cfg())
    assert torch.is_tensor(out)
    assert out.shape == (B, 256, 1024)
    assert out.dtype == torch.bfloat16


def test_video_targets_resizes_to_input_size():
    """256x256 inputs skip interpolation; 224x224 must be bilinear-resized."""
    out = _bundle().video_targets(_pil(), _pil(), _video_cfg())
    assert out.shape == (B, 256, 1024)


def test_video_targets_tuple_when_cls_loss():
    cfg = _video_cfg(use_cls_loss=True)
    patch, cls = _bundle().video_targets(_pil(height=256, width=256), _pil(height=256, width=256), cfg)
    assert patch.shape == (B, 256, 1024)
    assert cls.shape == (B, 1024)


def test_video_targets_dict_with_current_patch_and_cls_none():
    cfg = _video_cfg(use_current_patch_loss=True, use_cls_loss=False)
    out = _bundle().video_targets(_pil(height=256, width=256), _pil(height=256, width=256), cfg)
    assert isinstance(out, dict)
    assert out["patch"].shape == (B, 256, 1024)
    assert out["cls"] is None
    assert out["current_patch"].shape == (B, 256, 1024)


def test_video_targets_dict_with_cls_and_current():
    cfg = _video_cfg(use_cls_loss=True, use_current_patch_loss=True)
    out = _bundle().video_targets(_pil(height=256, width=256), _pil(height=256, width=256), cfg)
    assert out["cls"].shape == (B, 1024)
    assert out["current_patch"].shape == (B, 256, 1024)


def test_video_targets_rejects_no_loss_mode():
    with pytest.raises(ValueError, match="use_patch_loss or use_cls_loss"):
        _bundle().video_targets(_pil(), _pil(), _video_cfg(use_patch_loss=False))


def test_video_targets_passes_effective_fps_to_teacher():
    class _FPSProbe(_StubVideo):
        seen = None

        def get_future_feature(self, video, **kwargs):
            _FPSProbe.seen = kwargs.get("fps")
            return super().get_future_feature(video, **kwargs)

    bundle = DepthTeacherBundle(
        moge=_StubMoGe(), morgbd=_StubMoRGBD(), video=_FPSProbe(), device=torch.device("cpu")
    )
    bundle.video_targets(
        _pil(height=256, width=256), _pil(height=256, width=256), _video_cfg(), effective_fps=30 / 49
    )
    assert _FPSProbe.seen == pytest.approx(30 / 49)


# --------------------- policy._compute_align_targets ---------------------


ALIGN_PARAMS = {
    "mode": "query",
    "num_task_tokens": 8,
    "depth_loss_weight": 0.004,
    "use_future_video": True,
    "depth": {"model_type": "MoRGBD", "use_future_depth": True},
    "video": {
        "input_size": 256,
        "use_patch_loss": True,
        "use_current_patch_loss": True,
        "use_cls_loss": False,
    },
}


def _make_policy_stub(align_params, bundle):
    from lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 import LingbotVLAV2Policy

    stub = LingbotVLAV2Policy.__new__(LingbotVLAV2Policy)
    torch.nn.Module.__init__(stub)
    stub.config = SimpleNamespace(align_params=align_params)
    stub._align_teachers = bundle
    return stub


def test_compute_align_targets_full_recipe_unpack():
    policy = _make_policy_stub(copy.deepcopy(ALIGN_PARAMS), _bundle())
    batch = {"pil_images": _pil(), "future_pil_images": _pil(), "future_video_effective_fps": 30 / 49}
    targets = policy._compute_align_targets(batch)

    assert set(targets) == {
        "depth_targets",
        "future_depth_targets",
        "future_video_targets",
        "future_video_cls_targets",
        "future_video_current_patch",
    }
    for key in (
        "depth_targets",
        "future_depth_targets",
        "future_video_targets",
        "future_video_current_patch",
    ):
        assert targets[key].shape == (B, 256, 1024) and targets[key].dtype == torch.bfloat16
    assert targets["future_video_cls_targets"] is None  # use_cls_loss=False in the recipe


def test_compute_align_targets_depth_only_branch():
    params = copy.deepcopy(ALIGN_PARAMS)
    params["use_future_video"] = False
    params["depth"]["use_future_depth"] = False
    policy = _make_policy_stub(params, _bundle())
    targets = policy._compute_align_targets({"pil_images": _pil()})
    assert set(targets) == {"depth_targets"}


def test_compute_align_targets_tuple_bundle_unpack():
    params = copy.deepcopy(ALIGN_PARAMS)
    params["video"] = {"input_size": 256, "use_patch_loss": True, "use_cls_loss": True}
    policy = _make_policy_stub(params, _bundle())
    batch = {"pil_images": _pil(), "future_pil_images": _pil()}
    targets = policy._compute_align_targets(batch)
    assert targets["future_video_cls_targets"].shape == (B, 1024)
    assert "future_video_current_patch" not in targets


def test_compute_align_targets_missing_pil_images_is_actionable():
    policy = _make_policy_stub(copy.deepcopy(ALIGN_PARAMS), _bundle())
    with pytest.raises(RuntimeError, match="pil_images.*preprocessor.*use_depth_align"):
        policy._compute_align_targets({})


def test_compute_align_targets_missing_future_frames_is_actionable():
    policy = _make_policy_stub(copy.deepcopy(ALIGN_PARAMS), _bundle())
    with pytest.raises(RuntimeError, match="future_pil_images"):
        policy._compute_align_targets({"pil_images": _pil()})


def test_teachers_are_not_module_registered():
    """The bundle must stay outside the module tree or it lands in checkpoints/DDP."""
    policy = _make_policy_stub(copy.deepcopy(ALIGN_PARAMS), _bundle())
    assert "moge" not in dict(policy.named_modules())
    assert "_align_teachers" not in policy.state_dict()


# ----------------- no third-party teacher runtimes -----------------


def test_teacher_build_never_requires_an_upstream_checkout(monkeypatch):
    """The teacher subsystem must not import or sys.path-insert any upstream
    repository: a DINO-video branch without local weights fails on the missing
    weight path — never on a missing checkout."""
    import sys

    import lerobot.policies.lingbot_vla_v2.teachers.depth_teachers as dt

    monkeypatch.delenv("LINGBOT_VLA_V2_UPSTREAM", raising=False)
    from lerobot.policies.lingbot_vla_v2.teachers.depth_teachers import _load_video_teacher

    sys_path_before = list(sys.path)
    with pytest.raises((ValueError, FileNotFoundError), match="ckpt_path"):
        _load_video_teacher({"video": {}}, torch.device("cpu"))
    # The loader must not touch sys.path while failing (other tests may add
    # their own parity-check paths; only our deltas matter).
    assert sys.path == sys_path_before
    # No checkout-resolution helper may exist to regress to.
    assert not hasattr(dt, "_resolve_upstream_root")
    assert "LINGBOT_VLA_V2_UPSTREAM" not in Path(dt.__file__).read_text()


# ----------------- gated real-weight integration -----------------
# Runs only when the real teacher weights are present in the local HF cache.
# Never gated on an upstream checkout: the shipped runtimes must not depend
# on one.

import glob  # noqa: E402
import os  # noqa: E402

_HF_SNAPSHOT = None
try:
    _candidates = glob.glob(
        os.path.expanduser("~/.cache/huggingface/hub/models--robbyant--lingbot-vla-v2-6b/snapshots/*/")
    )
    if _candidates:
        _HF_SNAPSHOT = _candidates[0]
except Exception:
    _HF_SNAPSHOT = None


_MOGE_PATH = None
try:
    _moge_candidates = glob.glob(
        os.path.expanduser("~/.cache/huggingface/hub/models--Ruicheng--moge-2-vitb-normal/snapshots/*/model.pt")
    )
    if _moge_candidates:
        _MOGE_PATH = _moge_candidates[0]
except Exception:
    _MOGE_PATH = None


def _first_party_runtime_available() -> bool:
    from lerobot.policies.lingbot_vla_v2.teachers import depth_teachers

    return depth_teachers.FIRST_PARTY_TEACHERS_READY


_REAL_TEACHERS = pytest.mark.skipif(
    _HF_SNAPSHOT is None or _MOGE_PATH is None or not _first_party_runtime_available(),
    reason="first-party teacher runtimes / real weights not available",
)


def _depth_only_params() -> dict:
    params = copy.deepcopy(ALIGN_PARAMS)
    params["use_future_video"] = False
    params.pop("video")
    params["depth"] = {
        "moge_path": _MOGE_PATH,
        "morgbd_path": os.path.join(_HF_SNAPSHOT, "depth", "model.pt"),
    }
    return params


@_REAL_TEACHERS
def test_real_depth_teacher_end_to_end_first_party_runtime():
    """First-party MoGe/MoRGBD runtimes + downloaded weights; no checkout."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = DepthTeacherBundle.build(_depth_only_params(), device)
    targets = bundle.depth_targets(_pil().to(device))
    assert targets.shape == (B, 256, 1024)
    assert targets.dtype == torch.bfloat16


_DINO_WEIGHTS = None
if _HF_SNAPSHOT is not None:
    _dino_ckpt = os.path.join(_HF_SNAPSHOT, "dino_video", "teacher_step_10000.pth")
    if os.path.isfile(_dino_ckpt) and os.path.isfile(os.path.join(_HF_SNAPSHOT, "dino_video", "config.yaml")):
        _DINO_WEIGHTS = _dino_ckpt

_REAL_DINO = pytest.mark.skipif(
    _DINO_WEIGHTS is None or not _first_party_runtime_available(),
    reason="first-party DINO-video runtime / real weights not available",
)


@_REAL_DINO
def test_real_dino_video_teacher_end_to_end_first_party_runtime():
    """Full official-style recipe: first-party DINO runtime, weights only."""
    params = copy.deepcopy(ALIGN_PARAMS)
    params["depth"] = _depth_only_params()["depth"]
    params["video"]["ckpt_path"] = _DINO_WEIGHTS
    params["video"]["config_path"] = os.path.join(_HF_SNAPSHOT, "dino_video", "config.yaml")
    params["video"]["input_size"] = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = DepthTeacherBundle.build(params, device)
    out = bundle.video_targets(
        _pil(height=256, width=256).to(device), _pil(height=256, width=256).to(device), params["video"]
    )
    assert isinstance(out, dict)
    assert out["patch"].shape == (B, 256, 1024)
    assert out["patch"].dtype == torch.bfloat16
    assert out["current_patch"].shape == (B, 256, 1024)
    assert out["cls"] is None
