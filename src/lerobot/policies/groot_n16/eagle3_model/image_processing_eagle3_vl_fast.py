# --------------------------------------------------------
# NVIDIA
# Copyright (c) 2025 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# copy from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_onevision/image_processing_llava_onevision_fast.py
from typing import List, Optional, Union

from transformers.image_processing_utils import BatchFeature, get_patch_output_size, select_best_resolution
from transformers.image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    divide_to_patches,
    group_images_by_shape,
    reorder_images,
)
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    IMAGENET_STANDARD_MEAN, # 0.5, 0.5, 0.5
    IMAGENET_STANDARD_STD, # 0.5, 0.5, 0.5
    ChannelDimension,
    ImageInput,
    VideoInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
    make_flat_list_of_images,
    make_batched_videos,
    validate_kwargs
)
from transformers.processing_utils import Unpack
from transformers.utils import TensorType, add_start_docstrings, is_torch_available, is_torchvision_v2_available


if is_torch_available():
    import torch
if is_torchvision_v2_available():
    from transformers.image_utils import pil_torch_interpolation_mapping

    from torchvision.transforms.v2 import functional as F
else:
    from torchvision.transforms import functional as F

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
        raise TypeError('img should be torch.Tensor. Got {}'.format(type(img)))
    
    if img.ndim not in [2, 3]:
        raise ValueError('Image should have 2 or 3 dimensions. Got {}'.format(img.ndim))
    
    img_height = img.shape[1]
    img_width = img.shape[2]
    if top < 0 or left < 0 or bottom > img_height or right > img_width:
        raise ValueError('Crop coordinates out of bounds')
    
    if top >= bottom or left >= right:
        raise ValueError('Invalid crop coordinates')

    return img[:, top:bottom, left:right]


class Eagle3_VLFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    do_pad: Optional[bool]


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
class Eagle3_VLImageProcessorFast(BaseImageProcessorFast):
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

    @add_start_docstrings(
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
        """
            do_pad (`bool`, *optional*):
                    Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
                    number of patches in the batch. Padding will be applied to the bottom and right with zeros.
        """,
    )
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Eagle3_VLFastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

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

    def _preprocess(
        self,
        images: List["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, List[float]]],
        image_std: Optional[Union[float, List[float]]],
        do_pad: bool,
        return_tensors: Optional[Union[str, TensorType]],
    ) -> BatchFeature:

        image_sizes = [get_image_size(image, channel_dim=ChannelDimension.FIRST) for image in images]

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(images)
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


    def preprocess(self, images: ImageInput, videos: VideoInput=None, **kwargs: Unpack[Eagle3_VLFastImageProcessorKwargs]) -> BatchFeature:
        validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self.valid_kwargs.__annotations__.keys())
        # Set default kwargs from self. This ensures that if a kwarg is not provided
        # by the user, it gets its default value from the instance, or is set to None.
        for kwarg_name in self.valid_kwargs.__annotations__:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        # Extract parameters that are only used for preparing the input images
        do_convert_rgb = kwargs.pop("do_convert_rgb")
        input_data_format = kwargs.pop("input_data_format")
        device = kwargs.pop("device")
        # Prepare input images
        if images is not None:
            images = self._prepare_input_images(
                images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
            )

        if videos is not None:
            videos = self._prepare_input_images(
                images=videos, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
            )

        # Update kwargs that need further processing before being validated
        kwargs = self._further_process_kwargs(**kwargs)

        # Validate kwargs
        self._validate_preprocess_kwargs(**kwargs)

        # torch resize uses interpolation instead of resample
        resample = kwargs.pop("resample")
        kwargs["interpolation"] = (
            pil_torch_interpolation_mapping[resample] if isinstance(resample, (PILImageResampling, int)) else resample
        )

        # Pop kwargs that are not needed in _preprocess
        kwargs.pop("default_to_square")
        kwargs.pop("data_format")
        if images is not None:
            return self._preprocess(images, **kwargs)
        elif videos is not None:
            return self._preprocess(videos, **kwargs)
    
__all__ = ["Eagle3_VLImageProcessorFast"]
