from typing import Callable

import torch
import torchvision
from robomimic.models.base_nets import SpatialSoftmax
from torch import Tensor, nn
from torchvision.transforms import CenterCrop, RandomCrop


class RgbEncoder(nn.Module):
    """Encoder an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        norm_mean_std: tuple[float, float] = [1.0, 1.0],
        crop_shape: tuple[int, int] | None = None,
        random_crop: bool = False,
        backbone_name: str = "resnet18",
        pretrained_backbone: bool = False,
        use_group_norm: bool = False,
        relu: bool = True,
        num_keypoints: int = 32,
    ):
        """
        Args:
            input_shape: channel-first input shape (C, H, W)
            norm_mean_std: mean and standard deviation used for image normalization. Images are normalized as
                (image - mean) / std.
            crop_shape: (H, W) shape to crop to (must fit within the input shape). If not provided, no
                cropping is done.
            random_crop: Whether the crop should be random at training time (it's always a center crop in
                eval mode).
            backbone_name: The name of one of the available resnet models from torchvision (eg resnet18).
            pretrained_backbone: whether to use timm pretrained weights.
            use_group_norm: Whether to replace batch normalization with group normalization in the backbone.
                The group sizes are set to be about 16 (to be precise, feature_dim // 16).
            relu: whether to use relu as a final step.
            num_keypoints: Number of keypoints for SpatialSoftmax (default value of 32 matches PushT Image).
        """
        super().__init__()
        if input_shape[0] != 3:
            raise ValueError("Only RGB images are handled")
        if not backbone_name.startswith("resnet"):
            raise ValueError(
                "Only resnet is supported for now (because of the assumption that 'layer4' is the output layer)"
            )

        # Set up optional preprocessing.
        if norm_mean_std == [1.0, 1.0]:
            self.normalizer = nn.Identity()
        else:
            self.normalizer = torchvision.transforms.Normalize(mean=norm_mean_std[0], std=norm_mean_std[1])

        if crop_shape is not None:
            self.do_crop = True
            self.center_crop = CenterCrop(crop_shape)  # always use center crop for eval
            if random_crop:
                self.maybe_random_crop = RandomCrop(crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone.
        backbone_model = getattr(torchvision.models, backbone_name)(pretrained=pretrained_backbone)
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if use_group_norm:
            if pretrained_backbone:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # Set up pooling and final layers.
        # Use a dry run to get the feature map shape.
        with torch.inference_mode():
            feat_map_shape = tuple(self.backbone(torch.zeros(size=(1, *input_shape))).shape[1:])
        self.pool = SpatialSoftmax(feat_map_shape, num_kp=num_keypoints)
        self.feature_dim = num_keypoints * 2
        self.out = nn.Linear(num_keypoints * 2, self.feature_dim)
        self.maybe_relu = nn.ReLU() if relu else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Preprocess: normalize and maybe crop (if it was set up in the __init__).
        x = self.normalizer(x)
        if self.do_crop:
            if self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                # Always use center crop for eval.
                x = self.center_crop(x)
        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer.
        x = self.out(x)
        # Maybe a final non-linearity.
        x = self.maybe_relu(x)
        return x


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """
    Args:
        root_module: The module for which the submodules need to be replaced
        predicate: Takes a module as an argument and must return True if the that module is to be replaced.
        func: Takes a module as an argument and returns a new module to replace it with.
    Returns:
        The root module with its submodules replaced.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module
