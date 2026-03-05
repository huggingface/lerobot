# --------------------------------------------------------
# NVIDIA
# Copyright (c) 2025 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# copy from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_onevision/image_processing_llava_onevision_fast.py
from typing import Optional, Union

import numpy as np
from transformers.image_processing_utils import BatchFeature
from transformers.image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)

# These docstrings were added in newer versions of transformers - provide fallbacks for compatibility
try:
    from transformers.image_processing_utils_fast import (
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
    )
except ImportError:
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING = ""
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS = ""
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,  # 0.5, 0.5, 0.5
    IMAGENET_STANDARD_STD,  # 0.5, 0.5, 0.5
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
    make_flat_list_of_images,
    validate_kwargs,
)
from transformers.processing_utils import Unpack
from transformers.utils import (
    TensorType,
    add_start_docstrings,
    is_torch_available,
    is_torchvision_v2_available,
)
from transformers.video_utils import VideoInput

if is_torch_available():
    import torch
if is_torchvision_v2_available():
    from torchvision.transforms.v2 import functional as F  # noqa: N812
    from transformers.image_utils import pil_torch_interpolation_mapping
else:
    from torchvision.transforms import functional as F  # noqa: N812


def crop(img: torch.Tensor, left: int, top: int, right: int, bottom: int) -> torch.Tensor:
    """Crop the given numpy array.

    Args:
        img (torch.Tensor): Image to be cropped. Format should be (C, H, W).
        left (int): The left coordinate of the crop box.
        top (int): The top coordinate of the crop box.
        right (int): The right coordinate of the crop box.
        bottom (int): The bottom coordinate of the crop box.

    Returns:
        torch.Tensor: Cropped image.
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"img should be torch.Tensor. Got {type(img)}")

    if img.ndim not in [2, 3]:
        raise ValueError(f"Image should have 2 or 3 dimensions. Got {img.ndim}")

    img_height = img.shape[1]
    img_width = img.shape[2]
    if top < 0 or left < 0 or bottom > img_height or right > img_width:
        raise ValueError("Crop coordinates out of bounds")

    if top >= bottom or left >= right:
        raise ValueError("Invalid crop coordinates")

    return img[:, top:bottom, left:right]


class Eagle3_VLFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):  # noqa: N801
    do_pad: bool | None


@add_start_docstrings(
    "Constructs a fast ConvNeXT image processor. Based on [`SiglipImageProcessor`] with incorporation of processing each video frame.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    """
        image_grid_pinpoints (`List[List[int]]`, *optional*):
            A list of possible resolutions to use for processing high resolution images. The best resolution is selected
            based on the original size of the image. Can be overridden by `image_grid_pinpoints` in the `preprocess`
            method. Not used for processing videos.
        do_pad (`bool`, *optional*):
            Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
            number of patches in the batch. Padding will be applied to the bottom and right with zeros.
    """,
)
class Eagle3_VLImageProcessorFast(BaseImageProcessorFast):  # noqa: N801
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 448, "width": 448}
    default_to_square = False
    crop_size = None
    do_resize = True
    do_center_crop = None
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_pad = True
    valid_kwargs = Eagle3_VLFastImageProcessorKwargs
    model_input_names = ["pixel_values_videos"]

    def __init__(self, **kwargs: Unpack[Eagle3_VLFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    def _prepare_images_structure(
        self,
        images: ImageInput,
    ) -> ImageInput:
        """
        Prepare the images structure for processing.

        Args:
            images (`ImageInput`):
                The input images to process.

        Returns:
            `ImageInput`: The images with a valid nesting.
        """
        return make_flat_list_of_images(images)

    def _prepare_input_images(
        self,
        images: ImageInput,
        do_convert_rgb: bool = True,
        input_data_format: ChannelDimension | str | None = None,
        device: Union[str, "torch.device"] | None = None,
    ) -> list["torch.Tensor"]:
        """
        Prepare input images for processing by converting them to torch tensors.

        Args:
            images: Input images (PIL Images, numpy arrays, or tensors)
            do_convert_rgb: Whether to convert to RGB
            input_data_format: Input data format (channels first/last)
            device: Device to place tensors on

        Returns:
            List of torch tensors ready for preprocessing
        """
        from PIL import Image as PILImage
        from transformers.image_utils import to_numpy_array

        # Flatten images to a list
        images = make_flat_list_of_images(images)

        # Convert to tensors
        tensor_images = []
        for image in images:
            if isinstance(image, PILImage.Image):
                # Convert PIL to numpy, then to tensor
                if do_convert_rgb and image.mode != "RGB":
                    image = image.convert("RGB")
                np_image = to_numpy_array(image)
                # Convert to channels-first format (C, H, W)
                if len(np_image.shape) == 3:
                    np_image = np_image.transpose(2, 0, 1)  # HWC -> CHW
                tensor_image = torch.from_numpy(np_image)
            elif isinstance(image, torch.Tensor):
                tensor_image = image
                # Ensure channels-first if 3D
                if tensor_image.ndim == 3 and tensor_image.shape[0] != 3:
                    tensor_image = tensor_image.permute(2, 0, 1)  # HWC -> CHW
            elif isinstance(image, (np.ndarray, list)):
                np_image = np.asarray(image)
                if len(np_image.shape) == 3:
                    np_image = np_image.transpose(2, 0, 1)  # HWC -> CHW
                tensor_image = torch.from_numpy(np_image)
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

            # Move to device if specified
            if device is not None:
                tensor_image = tensor_image.to(device)

            tensor_images.append(tensor_image)

        return tensor_images

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool,
        return_tensors: str | TensorType | None,
        disable_grouping: bool | None = None,
    ) -> BatchFeature:
        image_sizes = [get_image_size(image, channel_dim=ChannelDimension.FIRST) for image in images]

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(
            images, disable_grouping=disable_grouping
        )
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images)

        return BatchFeature(
            data={"pixel_values": processed_images, "image_sizes": image_sizes}, tensor_type=return_tensors
        )

    def preprocess(
        self,
        images: ImageInput,
        videos: VideoInput = None,
        **kwargs: Unpack[Eagle3_VLFastImageProcessorKwargs],
    ) -> BatchFeature:
        validate_kwargs(
            captured_kwargs=kwargs.keys(), valid_processor_keys=self.valid_kwargs.__annotations__.keys()
        )
        # Set default kwargs from self. This ensures that if a kwarg is not provided
        # by the user, it gets its default value from the instance, or is set to None.
        for kwarg_name in self.valid_kwargs.__annotations__:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        # Extract parameters that are only used for preparing the input images
        do_convert_rgb = kwargs.pop("do_convert_rgb", self.do_convert_rgb)
        input_data_format = kwargs.pop("input_data_format", None)
        device = kwargs.pop("device", None)
        # Prepare input images
        if images is not None:
            images = self._prepare_input_images(
                images=images,
                do_convert_rgb=do_convert_rgb,
                input_data_format=input_data_format,
                device=device,
            )

        if videos is not None:
            videos = self._prepare_input_images(
                images=videos,
                do_convert_rgb=do_convert_rgb,
                input_data_format=input_data_format,
                device=device,
            )

        # Update kwargs that need further processing before being validated
        kwargs = self._further_process_kwargs(**kwargs)

        # Validate kwargs
        self._validate_preprocess_kwargs(**kwargs)

        # torch resize uses interpolation instead of resample
        resample = kwargs.pop("resample", self.resample)
        kwargs["interpolation"] = (
            pil_torch_interpolation_mapping[resample]
            if isinstance(resample, (PILImageResampling, int))
            else resample
        )

        # Pop kwargs that are not needed in _preprocess
        kwargs.pop("default_to_square", self.default_to_square)
        kwargs.pop("data_format", None)

        # Filter kwargs to only include those accepted by _preprocess
        valid_preprocess_kwargs = {
            "do_resize",
            "size",
            "interpolation",
            "do_center_crop",
            "crop_size",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "do_pad",
            "return_tensors",
            "disable_grouping",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_preprocess_kwargs}

        if images is not None:
            return self._preprocess(images, **filtered_kwargs)
        elif videos is not None:
            return self._preprocess(videos, **filtered_kwargs)


__all__ = ["Eagle3_VLImageProcessorFast"]
