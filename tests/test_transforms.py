import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import v2

from lerobot.common.datasets.transforms import SharpnessJitter, get_image_transforms
from lerobot.common.utils.utils import seeded_context

# test_make_image_transforms
# -

# test backward compatibility torchvision
# - save artifacts

# test backward compatibility default yaml (enable false, enable true)
# - save artifacts


def test_get_image_transforms_no_transform():
    get_image_transforms()
    get_image_transforms(sharpness_weight=0.0)
    get_image_transforms(max_num_transforms=0)


@pytest.fixture
def img():
    # dataset = LeRobotDataset("lerobot/pusht")
    # item = dataset[0]
    # return item["observation.image"]
    path = "tests/data/save_image_transforms/original_frame.png"
    img_chw = torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1)
    return img_chw


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
    with seeded_context(1337):
        for _ in range(20):
            tf = get_image_transforms(
                brightness_min_max=(0.5, 0.5),
                contrast_min_max=(0.5, 0.5),
                saturation_min_max=(0.5, 0.5),
                hue_min_max=(0.5, 0.5),
                sharpness_min_max=(0.5, 0.5),
                random_order=True,
            )
            out_imgs.append(tf(img))

    for i in range(1, 20):
        with pytest.raises(AssertionError):
            torch.testing.assert_close(out_imgs[0], out_imgs[i])


def test_backward_compatibility_torchvision():
    pass


def test_backward_compatibility_default_yaml():
    pass


# class TestRandomSubsetApply:
#     @pytest.fixture(autouse=True)
#     def setup(self):
#         self.jitters = [
#             v2.ColorJitter(brightness=0.5),
#             v2.ColorJitter(contrast=0.5),
#             v2.ColorJitter(saturation=0.5),
#         ]
#         self.flips = [v2.RandomHorizontalFlip(p=1), v2.RandomVerticalFlip(p=1)]
#         self.img = torch.rand(3, 224, 224)

#     @pytest.mark.parametrize("p", [[0, 1], [1, 0]])
#     def test_random_choice(self, p):
#         random_choice = RandomSubsetApply(self.flips, p=p, n_subset=1, random_order=False)
#         output = random_choice(self.img)

#         p_horz, _ = p
#         if p_horz:
#             torch.testing.assert_close(output, F.horizontal_flip(self.img))
#         else:
#             torch.testing.assert_close(output, F.vertical_flip(self.img))

#     def test_transform_all(self):
#         transform = RandomSubsetApply(self.jitters)
#         output = transform(self.img)
#         assert output.shape == self.img.shape

#     def test_transform_subset(self):
#         transform = RandomSubsetApply(self.jitters, n_subset=2)
#         output = transform(self.img)
#         assert output.shape == self.img.shape

#     def test_random_order(self):
#         random_order = RandomSubsetApply(self.flips, p=[0.5, 0.5], n_subset=2, random_order=True)
#         # We can't really check whether the transforms are actually applied in random order. However,
#         # horizontal and vertical flip are commutative. Meaning, even under the assumption that the transform
#         # applies them in random order, we can use a fixed order to compute the expected value.
#         actual = random_order(self.img)
#         expected = v2.Compose(self.flips)(self.img)
#         torch.testing.assert_close(actual, expected)

#     def test_probability_length_mismatch(self):
#         with pytest.raises(ValueError):
#             RandomSubsetApply(self.jitters, p=[0.5, 0.5])

#     def test_invalid_n_subset(self):
#         with pytest.raises(ValueError):
#             RandomSubsetApply(self.jitters, n_subset=5)


# class TestRangeRandomSharpness:
#     @pytest.fixture(autouse=True)
#     def setup(self):
#         self.img = torch.rand(3, 224, 224)

#     def test_valid_range(self):
#         transform = RangeRandomSharpness(0.1, 2.0)
#         output = transform(self.img)
#         assert output.shape == self.img.shape

#     def test_invalid_range_min_negative(self):
#         with pytest.raises(ValueError):
#             RangeRandomSharpness(-0.1, 2.0)

#     def test_invalid_range_max_smaller(self):
#         with pytest.raises(ValueError):
#             RangeRandomSharpness(2.0, 0.1)


# class TestMakeImageTransforms:
#     @pytest.fixture(autouse=True)
#     def setup(self):
#         """Seed should be the same as the one that was used to generate artifacts"""
#         self.config = {
#             "enable": True,
#             "max_num_transforms": 1,
#             "random_order": False,
#             "brightness": {"weight": 0, "min": 2.0, "max": 2.0},
#             "contrast": {
#                 "weight": 0,
#                 "min": 2.0,
#                 "max": 2.0,
#             },
#             "saturation": {
#                 "weight": 0,
#                 "min": 2.0,
#                 "max": 2.0,
#             },
#             "hue": {
#                 "weight": 0,
#                 "min": 0.5,
#                 "max": 0.5,
#             },
#             "sharpness": {
#                 "weight": 0,
#                 "min": 2.0,
#                 "max": 2.0,
#             },
#         }
#         self.path = Path("tests/data/save_image_transforms")
#         self.original_frame = self.load_png_to_tensor(self.path / "original_frame.png")
#         self.transforms = {
#             "brightness": v2.ColorJitter(brightness=(2.0, 2.0)),
#             "contrast": v2.ColorJitter(contrast=(2.0, 2.0)),
#             "saturation": v2.ColorJitter(saturation=(2.0, 2.0)),
#             "hue": v2.ColorJitter(hue=(0.5, 0.5)),
#             "sharpness": RangeRandomSharpness(2.0, 2.0),
#         }

#     @staticmethod
#     def load_png_to_tensor(path: Path):
#         return torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1)

#     @pytest.mark.parametrize(
#         "transform_key, seed",
#         [
#             ("brightness", 1336),
#             ("contrast", 1336),
#             ("saturation", 1336),
#             ("hue", 1336),
#             ("sharpness", 1336),
#         ],
#     )
#     def test_single_transform(self, transform_key, seed):
#         config = self.config
#         config[transform_key]["weight"] = 1
#         cfg = OmegaConf.create(config)

#         actual_t = make_image_transforms(cfg, to_dtype=torch.uint8)
#         with seeded_context(1336):
#             actual = actual_t(self.original_frame)

#         expected_t = self.transforms[transform_key]
#         with seeded_context(1336):
#             expected = expected_t(self.original_frame)

#         torch.testing.assert_close(actual, expected)
