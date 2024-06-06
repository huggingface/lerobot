from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F  # noqa: N812
from PIL import Image
from safetensors.torch import load_file

from lerobot.common.datasets.transforms import RandomSubsetApply, RangeRandomSharpness, make_transforms
from lerobot.common.datasets.utils import flatten_dict
from lerobot.common.utils.utils import init_hydra_config, seeded_context
from tests.utils import DEFAULT_CONFIG_PATH


class TestRandomSubsetApply:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.jitters = [
            v2.ColorJitter(brightness=0.5),
            v2.ColorJitter(contrast=0.5),
            v2.ColorJitter(saturation=0.5),
        ]
        self.flips = [v2.RandomHorizontalFlip(p=1), v2.RandomVerticalFlip(p=1)]
        self.img = torch.rand(3, 224, 224)

    @pytest.mark.parametrize("p", [[0, 1], [1, 0]])
    def test_random_choice(self, p):
        random_choice = RandomSubsetApply(self.flips, p=p, n_subset=1, random_order=False)
        output = random_choice(self.img)

        p_horz, _ = p
        if p_horz:
            torch.testing.assert_close(output, F.horizontal_flip(self.img))
        else:
            torch.testing.assert_close(output, F.vertical_flip(self.img))

    def test_transform_all(self):
        transform = RandomSubsetApply(self.jitters)
        output = transform(self.img)
        assert output.shape == self.img.shape

    def test_transform_subset(self):
        transform = RandomSubsetApply(self.jitters, n_subset=2)
        output = transform(self.img)
        assert output.shape == self.img.shape

    def test_random_order(self):
        random_order = RandomSubsetApply(self.flips, p=[0.5, 0.5], n_subset=2, random_order=True)
        # We can't really check whether the transforms are actually applied in random order. However,
        # horizontal and vertical flip are commutative. Meaning, even under the assumption that the transform
        # applies them in random order, we can use a fixed order to compute the expected value.
        actual = random_order(self.img)
        expected = v2.Compose(self.flips)(self.img)
        torch.testing.assert_close(actual, expected)

    def test_probability_length_mismatch(self):
        with pytest.raises(ValueError):
            RandomSubsetApply(self.jitters, p=[0.5, 0.5])

    def test_invalid_n_subset(self):
        with pytest.raises(ValueError):
            RandomSubsetApply(self.jitters, n_subset=5)


class TestRangeRandomSharpness:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.img = torch.rand(3, 224, 224)

    def test_valid_range(self):
        transform = RangeRandomSharpness(0.1, 2.0)
        output = transform(self.img)
        assert output.shape == self.img.shape

    def test_invalid_range_min_negative(self):
        with pytest.raises(ValueError):
            RangeRandomSharpness(-0.1, 2.0)

    def test_invalid_range_max_smaller(self):
        with pytest.raises(ValueError):
            RangeRandomSharpness(2.0, 0.1)


class TestMakeTransforms:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Seed should be the same as the one that was used to generate artifacts"""
        self.config = {
            "enable": True,
            "max_num_transforms": 1,
            "random_order": False,
            "brightness": {
                "weight": 0,
                "min": 0.0,
                "max": 2.0
            },
            "contrast": {
                "weight": 0,
                "min": 0.0,
                "max": 2.0,
            },
            "saturation": {
                "weight": 0,
                "min": 0.0,
                "max": 2.0,
            },
            "hue": {
                "weight": 0,
                "min": -0.5,
                "max": 0.5,
            },
            "sharpness": {
                "weight": 0,
                "min": 0.0,
                "max": 2.0,
            },
        }
        self.path = Path("tests/data/save_image_transforms")
        # self.expected_frames = load_file(self.path / f"transformed_frames_1336.safetensors")
        self.original_frame = self.load_png_to_tensor(self.path / "original_frame.png")
        # self.original_frame = self.expected_frames["original_frame"]
        self.transforms = {
            "brightness": v2.ColorJitter(brightness=(0.0, 2.0)),
            "contrast": v2.ColorJitter(contrast=(0.0, 2.0)),
            "saturation": v2.ColorJitter(saturation=(0.0, 2.0)),
            "hue": v2.ColorJitter(hue=(-0.5, 0.5)),
            "sharpness": RangeRandomSharpness(0.0, 2.0),
        }

    @staticmethod
    def load_png_to_tensor(path: Path):
        return torch.from_numpy(np.array(Image.open(path).convert('RGB'))).permute(2, 0, 1)

    @pytest.mark.parametrize(
        "transform_key, seed",
        [
            ("brightness", 1336),
            ("contrast", 1336),
            ("saturation", 1336),
            ("hue", 1336),
            ("sharpness", 1336),
        ]
    )
    def test_single_transform(self, transform_key, seed):
        config = self.config
        config[transform_key]["weight"] = 1
        cfg = OmegaConf.create(config)
        transform = make_transforms(cfg, to_dtype=torch.uint8)

        # expected_t = self.transforms[transform_key]
        with seeded_context(seed):
            actual = transform(self.original_frame)
        # torch.manual_seed(42)
        # actual = actual_t(self.original_frame)
        # torch.manual_seed(42)
        # expected = expected_t(self.original_frame)

        # with seeded_context(1336):
        #     expected = expected_t(self.original_frame)

        expected = self.load_png_to_tensor(self.path / f"{seed}_{transform_key}.png")
        # # expected = self.expected_frames[transform_key]
        to_pil = v2.ToPILImage()
        to_pil(actual).save(self.path / f"{seed}_{transform_key}_test.png", quality=100)
        torch.testing.assert_close(actual, expected)
