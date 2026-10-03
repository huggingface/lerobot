#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""The GPU-backend augmentation step: when it acts, what it touches, and how the trainer wires it."""

from types import SimpleNamespace

import pytest
import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ImageAugmentationProcessorStep, PolicyProcessorPipeline
from lerobot.transforms import ImageTransformConfig, ImageTransformsConfig
from lerobot.utils.constants import OBS_IMAGE, OBS_IMAGES, OBS_STATE
from tests.utils import DEVICE

CAM_A, CAM_B = f"{OBS_IMAGES}.a", f"{OBS_IMAGES}.b"


def _config() -> ImageTransformsConfig:
    return ImageTransformsConfig(
        enable=True,
        max_num_transforms=1,
        tfs={"brightness": ImageTransformConfig(type="ColorJitter", kwargs={"brightness": (0.5, 0.5)})},
    )


def _batch() -> dict:
    return {
        CAM_A: torch.full((2, 2, 3, 8, 8), 0.5),
        CAM_B: torch.full((2, 3, 8, 8), 0.5),
        OBS_IMAGE: torch.full((2, 3, 8, 8), 0.5),
        OBS_STATE: torch.ones(2, 4),
        "action": torch.ones(2, 4),
    }


def _features() -> dict:
    return {
        PipelineFeatureType.OBSERVATION: {
            CAM_A: PolicyFeature(FeatureType.VISUAL, (3, 8, 8)),
            OBS_STATE: PolicyFeature(FeatureType.STATE, (4,)),
        },
        PipelineFeatureType.ACTION: {"action": PolicyFeature(FeatureType.ACTION, (4,))},
    }


def _pipeline(step: ImageAugmentationProcessorStep | None = None) -> PolicyProcessorPipeline:
    steps = [step] if step is not None else []
    return PolicyProcessorPipeline(steps=steps, name="test")


def test_step_augments_every_image_key_and_nothing_else():
    step = ImageAugmentationProcessorStep(config=_config())
    batch = _batch()
    out = _pipeline(step)(batch)
    for key in (CAM_A, CAM_B, OBS_IMAGE):
        assert out[key].shape == batch[key].shape
        torch.testing.assert_close(out[key], torch.full_like(batch[key], 0.25))
    assert torch.equal(out[OBS_STATE], batch[OBS_STATE])
    assert torch.equal(out["action"], batch["action"])


def test_explicit_image_keys_limit_the_augmentation():
    step = ImageAugmentationProcessorStep(config=_config(), image_keys=[CAM_B, "missing"])
    batch = _batch()
    out = _pipeline(step)(batch)
    torch.testing.assert_close(out[CAM_B], torch.full_like(batch[CAM_B], 0.25))
    assert torch.equal(out[CAM_A], batch[CAM_A])


def test_config_round_trips(tmp_path):
    step = ImageAugmentationProcessorStep(
        config=_config(), image_keys=[CAM_A], device="cpu", chunk_size=4, seed=11
    )
    config = step.get_config()
    assert config["config"]["tfs"]["brightness"]["kwargs"]["brightness"] == (0.5, 0.5)
    reloaded = ImageAugmentationProcessorStep(**config)
    assert isinstance(reloaded.config, ImageTransformsConfig)
    assert reloaded.config == step.config and reloaded.image_keys == [CAM_A] and reloaded.chunk_size == 4
    assert reloaded.compile_model is False and reloaded.seed == 11
    _pipeline(step).save_pretrained(tmp_path)
    loaded = PolicyProcessorPipeline.from_pretrained(tmp_path, config_filename="test.json")
    assert isinstance(loaded.steps[0], ImageAugmentationProcessorStep) and loaded.steps[0].seed == 11


def test_seeded_step_reproduces_its_augmentation_and_spares_the_default_rng():
    """Same seed, same augmentation; and the policy's own random draws are not shifted by it."""
    config = ImageTransformsConfig(enable=True)
    batch = {CAM_A: torch.rand(4, 2, 3, 8, 8, generator=torch.Generator().manual_seed(0))}
    outputs = []
    for _ in range(2):
        torch.manual_seed(3)
        outputs.append(_pipeline(ImageAugmentationProcessorStep(config=config, seed=1))(dict(batch)))
        outputs.append(torch.rand(2))
    assert torch.equal(outputs[0][CAM_A], outputs[2][CAM_A])
    assert not torch.equal(outputs[0][CAM_A], batch[CAM_A])
    torch.manual_seed(3)
    assert torch.equal(outputs[1], torch.rand(2)), "augmentation consumed the default generator"
    other = _pipeline(ImageAugmentationProcessorStep(config=config, seed=2))(dict(batch))
    assert not torch.equal(other[CAM_A], outputs[0][CAM_A])


def test_transform_features_is_identity():
    features = _features()
    assert ImageAugmentationProcessorStep(config=_config()).transform_features(features) == features


# --- How the trainer wires it -------------------------------------------------------------------


@pytest.fixture
def cpu_generator(monkeypatch):
    """Let a step built for an accelerator device be constructed on a CPU-only runner.

    The step builds its generator on `device` at construction. The wiring tests pass `device="cuda"` as
    the trainer would, on runners that have no CUDA; the generator itself is covered by the seeding tests.
    """
    monkeypatch.setattr(
        ImageAugmentationProcessorStep, "_make_generator", lambda self, device: torch.Generator()
    )


def _train_cfg(backend: str = "gpu", seed: int | None = 1000):
    config = _config()
    config.backend = backend
    config.gpu_compile = False
    config.gpu_chunk_size = 8
    return SimpleNamespace(dataset=SimpleNamespace(image_transforms=config), seed=seed)


def _dataset(image_transforms=None):
    meta = SimpleNamespace(
        camera_keys=[CAM_A, CAM_B, "observation.images.depth"], depth_keys=["observation.images.depth"]
    )
    return SimpleNamespace(meta=meta, image_transforms=image_transforms)


def test_trainer_builds_the_step_for_the_gpu_backend(cpu_generator):
    pytest.importorskip("datasets")  # the trainer and the dataset factory need the [dataset] extra
    from lerobot.scripts.lerobot_train import _make_image_augmentation

    step = _make_image_augmentation(_train_cfg(), _dataset(), torch.device("cuda"), process_index=2)
    assert isinstance(step, ImageAugmentationProcessorStep)
    assert step.image_keys == [CAM_A, CAM_B]  # camera keys, depth left out, no rename applied
    assert step.device == "cuda" and step.chunk_size == 8 and step.compile_model is False
    assert step.seed == 1002  # cfg.seed + process_index, so ranks augment differently


def test_trainer_builds_nothing_for_the_dataloader_backend():
    pytest.importorskip("datasets")  # the trainer and the dataset factory need the [dataset] extra
    from lerobot.scripts.lerobot_train import _make_image_augmentation

    assert _make_image_augmentation(_train_cfg("dataloader"), _dataset(), torch.device("cuda"), 0) is None
    cfg = _train_cfg()
    cfg.dataset.image_transforms.enable = False
    assert _make_image_augmentation(cfg, _dataset(), torch.device("cuda"), 0) is None


def test_trainer_leaves_the_seed_unset_when_the_run_is_unseeded(cpu_generator):
    pytest.importorskip("datasets")  # the trainer and the dataset factory need the [dataset] extra
    from lerobot.scripts.lerobot_train import _make_image_augmentation

    step = _make_image_augmentation(_train_cfg(seed=None), _dataset(), torch.device("cuda"), 3)
    assert step.seed is None


def test_trainer_refuses_a_dataset_that_already_carries_worker_transforms():
    """A hand-built dataset with its own `image_transforms` plus the gpu backend would augment twice."""
    pytest.importorskip("datasets")  # the trainer and the dataset factory need the [dataset] extra
    from lerobot.scripts.lerobot_train import _make_image_augmentation

    with pytest.raises(ValueError, match="already carries"):
        _make_image_augmentation(
            _train_cfg(), _dataset(image_transforms=lambda x: x), torch.device("cuda"), 0
        )


def test_only_training_batches_are_augmented():
    """The augmentation step runs before `_preprocess_dataset_batch` on the training path only."""
    pytest.importorskip("datasets")  # the trainer and the dataset factory need the [dataset] extra
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch

    augmentation = ImageAugmentationProcessorStep(config=_config())
    identity_preprocessor = lambda batch: batch  # noqa: E731
    train_batch = _preprocess_dataset_batch(
        augmentation.observation(_batch()), [CAM_A, CAM_B], {}, identity_preprocessor
    )
    eval_batch = _preprocess_dataset_batch(_batch(), [CAM_A, CAM_B], {}, identity_preprocessor)
    torch.testing.assert_close(train_batch[CAM_A], torch.full_like(train_batch[CAM_A], 0.25))
    torch.testing.assert_close(eval_batch[CAM_A], torch.full_like(eval_batch[CAM_A], 0.5))


def test_trainer_path_keeps_unprefixed_camera_keys_and_custom_columns(cpu_generator):
    """The step is applied to the batch dict directly, so nothing `batch_to_transition` would drop is lost.

    `camera_keys` selects by dtype, so a camera can be named `image` with no `observation.` prefix, and a
    dataset can carry columns of its own. Going through `PolicyProcessorPipeline` keeps `observation.*`
    keys and a fixed list of others only, and this runs before `rename_map` could rename them.
    """
    pytest.importorskip("datasets")  # the trainer and the dataset factory need the [dataset] extra
    from lerobot.scripts.lerobot_train import _make_image_augmentation

    meta = SimpleNamespace(camera_keys=["image", CAM_A], depth_keys=[])
    step = _make_image_augmentation(
        _train_cfg(), SimpleNamespace(meta=meta, image_transforms=None), torch.device("cuda"), 0
    )
    step._device = None  # keep the frames on the CPU runner; the device move is covered elsewhere
    batch = {
        "image": torch.full((2, 3, 8, 8), 0.5),
        CAM_A: torch.full((2, 2, 3, 8, 8), 0.5),
        OBS_STATE: torch.ones(2, 4),
        "action": torch.ones(2, 4),
        "custom.column": torch.arange(2),
        "task": ["pick", "place"],
    }
    out = step.observation(dict(batch))
    assert set(out) == set(batch)
    torch.testing.assert_close(out["image"], torch.full_like(batch["image"], 0.25))
    torch.testing.assert_close(out[CAM_A], torch.full_like(batch[CAM_A], 0.25))
    for key in (OBS_STATE, "action", "custom.column"):
        assert torch.equal(out[key], batch[key]), key
    assert out["task"] == batch["task"]


def test_pipeline_wrapper_would_drop_those_keys():
    """Why the trainer does not wrap the step in a pipeline: the conversion loses un-prefixed keys."""
    batch = {
        "image": torch.full((2, 3, 8, 8), 0.5),
        CAM_A: torch.full((2, 2, 3, 8, 8), 0.5),
        "custom.column": torch.arange(2),
        "action": torch.ones(2, 4),
    }
    out = _pipeline(ImageAugmentationProcessorStep(config=_config(), image_keys=["image", CAM_A]))(batch)
    assert CAM_A in out and "action" in out
    assert "image" not in out and "custom.column" not in out


def test_train_config_rejects_gpu_backend_on_cpu_policy(tmp_path):
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.act.configuration_act import ACTConfig

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="lerobot/test", image_transforms=_config()),
        policy=ACTConfig(device="cpu", push_to_hub=False),
        output_dir=tmp_path / "out",
    )
    cfg.dataset.image_transforms.backend = "gpu"
    with pytest.raises(ValueError, match="backend='gpu'"):
        cfg.validate()
    cfg.dataset.image_transforms.enable = False
    cfg.validate()  # a disabled configuration is fine on any device


def test_every_worker_transform_site_honours_the_backend():
    """Each place the dataset factory builds worker transforms must check `backend`, not just `enable`.

    `make_train_eval_datasets` once checked `enable` alone, so a run with `eval_split > 0` augmented on
    the workers *and* on the device: up to twice the transforms and none of the throughput the gpu
    backend exists for. This walks the factory's source and requires every `ImageTransforms(...)`
    construction to sit under a condition that reads `backend`.
    """
    pytest.importorskip("datasets")  # the trainer and the dataset factory need the [dataset] extra
    import ast
    import inspect

    from lerobot.datasets import factory

    tree = ast.parse(inspect.getsource(factory))
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    sites = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ImageTransforms"
        ):
            guard, function = None, None
            scope = node
            while scope in parents:
                scope = parents[scope]
                if guard is None and isinstance(scope, ast.If | ast.IfExp):
                    guard = scope.test
                if isinstance(scope, ast.FunctionDef):
                    function = scope.name
                    break
            sites[function] = guard is not None and "backend" in ast.dump(guard)
    assert sites == {"make_dataset": True, "make_train_eval_datasets": True}, (
        f"ImageTransforms construction sites and whether each checks image_transforms.backend: {sites}"
    )


@pytest.mark.skipif(DEVICE == "cpu", reason="moves frames to the accelerator")
def test_step_moves_frames_to_its_device():
    step = ImageAugmentationProcessorStep(config=_config(), device=DEVICE)
    out = _pipeline(step)(_batch())
    assert out[CAM_A].device.type == torch.device(DEVICE).type
    torch.testing.assert_close(out[CAM_A].cpu(), torch.full((2, 2, 3, 8, 8), 0.25))


@pytest.mark.skipif(DEVICE == "cpu", reason="compiles the transforms on the accelerator")
def test_compiled_step_matches_the_eager_step():
    config = ImageTransformsConfig(enable=True)
    batch = {CAM_A: torch.rand(4, 2, 3, 8, 8, generator=torch.Generator().manual_seed(0))}
    outputs = [
        _pipeline(
            ImageAugmentationProcessorStep(config=config, device=DEVICE, compile_model=compile_model, seed=1)
        )(dict(batch))[CAM_A]
        for compile_model in (False, True)
    ]
    torch.testing.assert_close(outputs[0], outputs[1], atol=1e-5, rtol=0)
