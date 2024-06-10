from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F  # noqa: N812

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.transforms import RandomSubsetApply, SharpnessJitter, get_image_transforms
from lerobot.common.utils.utils import init_hydra_config, seeded_context
from tests.utils import DEFAULT_CONFIG_PATH

ARTIFACT_DIR = Path("tests/data/save_image_transforms")
REPO_ID = "lerobot/aloha_mobile_shrimp"


def load_png_to_tensor(path: Path):
    return torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1)


@pytest.fixture
def img():
    dataset = LeRobotDataset(REPO_ID)
    return dataset[0][dataset.camera_keys[0]]


@pytest.fixture
def img_random():
    return torch.rand(3, 480, 640)


@pytest.fixture
def color_jitters():
    return [
        v2.ColorJitter(brightness=0.5),
        v2.ColorJitter(contrast=0.5),
        v2.ColorJitter(saturation=0.5),
    ]


@pytest.fixture
def single_transforms():
    return load_file(ARTIFACT_DIR / "single_transforms.safetensors")


@pytest.fixture
def default_transforms():
    return load_file(ARTIFACT_DIR / "default_transforms.safetensors")


def test_get_image_transforms_no_transform(img):
    tf_actual = get_image_transforms(brightness_min_max=(0.5, 0.5), max_num_transforms=0)
    torch.testing.assert_close(tf_actual(img), img)


def test_get_image_transforms_brightness(img):
    brightness_min_max = (0.5, 0.5)
    tf_actual = get_image_transforms(brightness_weight=1.0, brightness_min_max=brightness_min_max)
    tf_expected = v2.ColorJitter(brightness=brightness_min_max)
    torch.testing.assert_close(tf_actual(img), tf_expected(img))


def test_get_image_transforms_contrast(img):
    contrast_min_max = (0.5, 0.5)
    tf_actual = get_image_transforms(contrast_weight=1.0, contrast_min_max=contrast_min_max)
    tf_expected = v2.ColorJitter(contrast=contrast_min_max)
    torch.testing.assert_close(tf_actual(img), tf_expected(img))


def test_get_image_transforms_saturation(img):
    saturation_min_max = (0.5, 0.5)
    tf_actual = get_image_transforms(saturation_weight=1.0, saturation_min_max=saturation_min_max)
    tf_expected = v2.ColorJitter(saturation=saturation_min_max)
    torch.testing.assert_close(tf_actual(img), tf_expected(img))


def test_get_image_transforms_hue(img):
    hue_min_max = (0.5, 0.5)
    tf_actual = get_image_transforms(hue_weight=1.0, hue_min_max=hue_min_max)
    tf_expected = v2.ColorJitter(hue=hue_min_max)
    torch.testing.assert_close(tf_actual(img), tf_expected(img))


def test_get_image_transforms_sharpness(img):
    sharpness_min_max = (0.5, 0.5)
    tf_actual = get_image_transforms(sharpness_weight=1.0, sharpness_min_max=sharpness_min_max)
    tf_expected = SharpnessJitter(sharpness=sharpness_min_max)
    torch.testing.assert_close(tf_actual(img), tf_expected(img))


def test_get_image_transforms_max_num_transforms(img):
    tf_actual = get_image_transforms(
        brightness_min_max=(0.5, 0.5),
        contrast_min_max=(0.5, 0.5),
        saturation_min_max=(0.5, 0.5),
        hue_min_max=(0.5, 0.5),
        sharpness_min_max=(0.5, 0.5),
        random_order=False,
    )
    tf_expected = v2.Compose(
        [
            v2.ColorJitter(brightness=(0.5, 0.5)),
            v2.ColorJitter(contrast=(0.5, 0.5)),
            v2.ColorJitter(saturation=(0.5, 0.5)),
            v2.ColorJitter(hue=(0.5, 0.5)),
            SharpnessJitter(sharpness=(0.5, 0.5)),
        ]
    )
    torch.testing.assert_close(tf_actual(img), tf_expected(img))


def test_get_image_transforms_random_order(img):
    out_imgs = []
    tf = get_image_transforms(
        brightness_min_max=(0.5, 0.5),
        contrast_min_max=(0.5, 0.5),
        saturation_min_max=(0.5, 0.5),
        hue_min_max=(0.5, 0.5),
        sharpness_min_max=(0.5, 0.5),
        random_order=True,
    )
    with seeded_context(1335):
        for _ in range(20):
            out_imgs.append(tf(img))

    for i in range(1, len(out_imgs)):
        with pytest.raises(AssertionError):
            torch.testing.assert_close(out_imgs[0], out_imgs[i])


@pytest.mark.parametrize(
    "transform",
    [
        "brightness",
        "contrast",
        "saturation",
        "hue",
        "sharpness",
    ],
)
def test_backward_compatibility_torchvision(transform, img, single_transforms):
    kwargs = {
        f"{transform}_weight": 1.0,
        f"{transform}_min_max": (0.5, 0.5),
    }
    tf = get_image_transforms(**kwargs)
    actual = tf(img)

    expected = single_transforms[transform]

    torch.testing.assert_close(actual, expected)


def test_backward_compatibility_default_yaml(img, default_transforms):
    cfg = init_hydra_config(DEFAULT_CONFIG_PATH)
    cfg_tf = cfg.image_transforms
    default_tf = get_image_transforms(
        brightness_weight=cfg_tf.brightness.weight,
        brightness_min_max=cfg_tf.brightness.min_max,
        contrast_weight=cfg_tf.contrast.weight,
        contrast_min_max=cfg_tf.contrast.min_max,
        saturation_weight=cfg_tf.saturation.weight,
        saturation_min_max=cfg_tf.saturation.min_max,
        hue_weight=cfg_tf.hue.weight,
        hue_min_max=cfg_tf.hue.min_max,
        sharpness_weight=cfg_tf.sharpness.weight,
        sharpness_min_max=cfg_tf.sharpness.min_max,
        max_num_transforms=cfg_tf.max_num_transforms,
        random_order=cfg_tf.random_order,
    )

    with seeded_context(1337):
        actual = default_tf(img)

    expected = default_transforms["default"]

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("p", [[0, 1], [1, 0]])
def test_random_subset_apply_single_choice(p, img):
    flips = [v2.RandomHorizontalFlip(p=1), v2.RandomVerticalFlip(p=1)]
    random_choice = RandomSubsetApply(flips, p=p, n_subset=1, random_order=False)
    actual = random_choice(img)

    p_horz, _ = p
    if p_horz:
        torch.testing.assert_close(actual, F.horizontal_flip(img))
    else:
        torch.testing.assert_close(actual, F.vertical_flip(img))


def test_random_subset_apply_random_order(img):
    flips = [v2.RandomHorizontalFlip(p=1), v2.RandomVerticalFlip(p=1)]
    random_order = RandomSubsetApply(flips, p=[0.5, 0.5], n_subset=2, random_order=True)
    # We can't really check whether the transforms are actually applied in random order. However,
    # horizontal and vertical flip are commutative. Meaning, even under the assumption that the transform
    # applies them in random order, we can use a fixed order to compute the expected value.
    actual = random_order(img)
    expected = v2.Compose(flips)(img)
    torch.testing.assert_close(actual, expected)


def test_random_subset_apply_valid_transforms(color_jitters, img):
    transform = RandomSubsetApply(color_jitters)
    output = transform(img)
    assert output.shape == img.shape


def test_random_subset_apply_probability_length_mismatch(color_jitters):
    with pytest.raises(ValueError):
        RandomSubsetApply(color_jitters, p=[0.5, 0.5])


@pytest.mark.parametrize("n_subset", [0, 5])
def test_random_subset_apply_invalid_n_subset(color_jitters, n_subset):
    with pytest.raises(ValueError):
        RandomSubsetApply(color_jitters, n_subset=n_subset)


def test_sharpness_jitter_valid_range_tuple(img):
    tf = SharpnessJitter((0.1, 2.0))
    output = tf(img)
    assert output.shape == img.shape


def test_sharpness_jitter_valid_range_float(img):
    tf = SharpnessJitter(0.5)
    output = tf(img)
    assert output.shape == img.shape


def test_sharpness_jitter_invalid_range_min_negative():
    with pytest.raises(ValueError):
        SharpnessJitter((-0.1, 2.0))


def test_sharpness_jitter_invalid_range_max_smaller():
    with pytest.raises(ValueError):
        SharpnessJitter((2.0, 0.1))
