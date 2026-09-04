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

"""Interface contract for the first-party DINO-video teacher subpackage.

The runtime is implemented (bit-exact against the upstream SDPA reference on
real weights); these tests pin its boundary behaviour without weights: the two
documented public names, actionable failures for missing weights / unknown
keys, hard rejection of repository/checkout keys, and lazy loading from the
teacher bundle.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from lerobot.policies.lingbot_vla_v2.teachers.dino_video import (  # noqa: E402
    DinoVideoTeacher,
    build_dino_video_teacher,
)

_PKG_DIR = Path(__file__).resolve().parents[3] / "src/lerobot/policies/lingbot_vla_v2/teachers/dino_video"


def test_public_surface_is_exactly_two_names():
    import lerobot.policies.lingbot_vla_v2.teachers.dino_video as pkg

    assert set(pkg.__all__) == {"DinoVideoTeacher", "build_dino_video_teacher"}


def test_missing_checkpoint_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="teacher checkpoint not found"):
        build_dino_video_teacher({"ckpt_path": str(tmp_path / "missing.pth")})


def test_missing_config_is_actionable(tmp_path):
    ckpt = tmp_path / "teacher_step_10000.pth"
    ckpt.write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="teacher config not found"):
        build_dino_video_teacher({"ckpt_path": str(ckpt)})


def test_repository_keys_are_rejected(tmp_path):
    """No checkout/provider fallback may ever be reintroduced."""
    ckpt = tmp_path / "teacher_step_10000.pth"
    ckpt.write_bytes(b"")
    with pytest.raises(ValueError, match="repository/checkout keys.*rejected"):
        build_dino_video_teacher({"ckpt_path": str(ckpt), "upstream_root": "/some/checkout"})
    with pytest.raises(ValueError, match="repository/checkout keys.*rejected"):
        build_dino_video_teacher({"ckpt_path": str(ckpt), "provider_path": "x"})


def test_unknown_keys_are_rejected(tmp_path):
    ckpt = tmp_path / "teacher_step_10000.pth"
    ckpt.write_bytes(b"")
    with pytest.raises(ValueError, match="unknown align_params.video keys"):
        build_dino_video_teacher({"ckpt_path": str(ckpt), "nonsense_key": 1})


def test_recipe_keys_are_tolerated_without_weights_error(tmp_path):
    """The bundle forwards the whole align_params.video dict; loss/head keys
    must be accepted (the failure must be about weights, not schema)."""
    ckpt = tmp_path / "teacher_step_10000.pth"
    with pytest.raises(FileNotFoundError, match="teacher checkpoint not found"):
        build_dino_video_teacher(
            {"ckpt_path": str(ckpt), "use_patch_loss": True, "dim_out": 1024, "cls_pool": "last"}
        )


def test_scaffold_has_no_upstream_or_vendor_markers():
    """The subpackage must stay first-party: no upstream package names, no
    checkout resolvers, no vendor directories."""
    sources = list(_PKG_DIR.glob("*.py"))
    assert len(sources) >= 6, "expected the six modules"
    for source in sources:
        text = source.read_text()
        for marker in ("LINGBOT_VLA_V2_UPSTREAM", "sys.path.insert", "import lumos_dinov3"):
            assert marker not in text, f"{source.name} contains forbidden runtime hook {marker!r}"
    assert not (_PKG_DIR / "vendor").exists()
    assert not (_PKG_DIR / "third_party").exists()


def test_bundle_does_not_import_the_subpackage_until_needed():
    """DINO stays lazily imported: importing the teacher bundle alone must not
    pull the dino_video subpackage."""
    import subprocess
    import sys

    probe = (
        "import sys; "
        "import lerobot.policies.lingbot_vla_v2.teachers.depth_teachers; "
        "print([m for m in sys.modules if 'teachers.dino_video' in m])"
    )
    out = subprocess.run(  # noqa: S603 - fixed interpreter, fixed argv
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "[]", f"dino_video imported eagerly: {out.stdout.strip()}"


def test_dino_video_teacher_is_a_frozen_module_by_construction():
    """from_pretrained's freeze contract is part of the public API surface."""
    assert issubclass(DinoVideoTeacher, torch.nn.Module)
