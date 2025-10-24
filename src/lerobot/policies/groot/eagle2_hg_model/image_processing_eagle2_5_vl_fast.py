# --------------------------------------------------------
# NVIDIA
# Copyright (c) 2025 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------


# copy from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_onevision/image_processing_llava_onevision_fast.py
from typing import Optional

from transformers.image_processing_utils import (
    BatchFeature,
    get_patch_output_size,
)
from transformers.image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
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


class Eagle25VLFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    max_dynamic_tiles: int | None
    min_dynamic_tiles: int | None
    use_thumbnail: bool | None
    pad_during_tiling: bool | None
    do_pad: bool | None


@add_start_docstrings(
    "Constructs a fast ConvNeXT image processor. Based on [`SiglipImageProcessor`] with incorporation of processing each video frame.",
    # BASE_IMAGE_PROCESSOR_FAST_DOCSTRING, TODO: this was depreciated from transformers remove!
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
class Eagle25VLImageProcessorFast(BaseImageProcessorFast):
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
    max_dynamic_tiles = 12
    min_dynamic_tiles = 1
    use_thumbnail = True
    pad_during_tiling = False
    valid_kwargs = Eagle25VLFastImageProcessorKwargs
    model_input_names = ["pixel_values_videos"]

    def __init__(self, **kwargs: Unpack[Eagle25VLFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    @add_start_docstrings(
        # BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS, TODO: this was depreciated from transformers remove!
        """
            max_dynamic_tiles (`int`, *optional*):
                The maximum number of dynamic tiles to use for processing high resolution images.
            min_dynamic_tiles (`int`, *optional*):
                The minimum number of dynamic tiles to use for processing high resolution images.
            use_thumbnail (`bool`, *optional*):
                Whether to use a thumbnail for processing high resolution images.
            pad_during_tiling (`bool`, *optional*):
                Whether to pad the image during tiling.
            do_pad (`bool`, *optional*):
                    Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
                    number of patches in the batch. Padding will be applied to the bottom and right with zeros.
        """,
    )

    # NOTE(YL): we will overload the preprocess method to add the image_flags
    # def preprocess(
    #     self, images: ImageInput, **kwargs: Unpack[Eagle25VLFastImageProcessorKwargs]
    # ) -> BatchFeature:
    #     return super().preprocess(images, **kwargs)

    def _prepare_images_structure(
        self,
        images: ImageInput,
        expected_ndims: int = 3,
    ) -> ImageInput:
        """
        Prepare the images structure for processing.

        Args:
            images (`ImageInput`):
                The input images to process.
            expected_ndims (`int`, *optional*, defaults to 3):
                Expected number of dimensions for the images (added for transformers >=4.53.0 compatibility).

        Returns:
            `ImageInput`: The images with a valid nesting.
        """
        return make_flat_list_of_images(images)

    def _resize_for_patching(
        self,
        image: "torch.Tensor",
        target_resolution: tuple,
        interpolation: "F.InterpolationMode",
        input_data_format: ChannelDimension,
    ) -> "torch.Tensor":
        """
        Resizes an image to a target resolution while maintaining aspect ratio.

        Args:
            image ("torch.Tensor"):
                The input image.
            target_resolution (tuple):
                The target resolution (height, width) of the image.
            interpolation (`InterpolationMode`):
                Resampling filter to use if resizing the image.
            input_data_format (`ChannelDimension` or `str`):
                The channel dimension format of the input image.

        Returns:
            "torch.Tensor": The resized and padded image.
        """
        new_height, new_width = get_patch_output_size(image, target_resolution, input_data_format)

        # Resize the image
        resized_image = F.resize(image, (new_height, new_width), interpolation=interpolation)

        return resized_image

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """
        previous version mainly focus on ratio.
        We also consider area ratio here.
        """
        best_factor = float("-inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            # ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            # area_ratio = (ratio[0] * ratio[1] * image_size * image_size) / area
            """
            new area > 60% of original image area is enough.
            """
            factor_based_on_area_n_ratio = min(
                (ratio[0] * ratio[1] * image_size * image_size) / area, 0.6
            ) * min(target_aspect_ratio / aspect_ratio, aspect_ratio / target_aspect_ratio)

            if factor_based_on_area_n_ratio > best_factor:
                best_factor = factor_based_on_area_n_ratio
                best_ratio = ratio

        return best_ratio

    def _pad_for_patching(
        self, image: "torch.Tensor", target_resolution: tuple, input_data_format: ChannelDimension
    ) -> "torch.Tensor":
        """
        Pad an image to a target resolution while maintaining aspect ratio.
        """
        target_height, target_width = target_resolution
        new_height, new_width = get_patch_output_size(image, target_resolution, input_data_format)

        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2

        padded_image = F.pad(image, padding=[paste_x, paste_y, paste_x, paste_y])

        return padded_image

    def _get_image_patches(
        self,
        image: "torch.Tensor",
        min_num: int,
        max_num: int,
        size: tuple,
        tile_size: int,
        use_thumbnail: bool,
        interpolation: "F.InterpolationMode",
        pad_during_tiling: bool,
    ) -> list["torch.Tensor"]:
        image_size = get_image_size(image, channel_dim=ChannelDimension.FIRST)
        orig_height, orig_width = image_size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        }
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, tile_size
        )

        # calculate the target width and height
        target_width = tile_size * target_aspect_ratio[0]
        target_height = tile_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        if pad_during_tiling:
            resized_image = self._resize_for_patching(
                image,
                (target_height, target_width),
                interpolation=interpolation,
                input_data_format=ChannelDimension.FIRST,
            )
            padded_image = self._pad_for_patching(
                resized_image,
                (target_height, target_width),
                input_data_format=ChannelDimension.FIRST,
            )
            image_used_to_split = padded_image
        else:
            image_used_to_split = F.resize(image, (target_height, target_width), interpolation=interpolation)

        processed_tiles = []
        for i in range(blocks):
            box = (
                (i % (target_width // tile_size)) * tile_size,
                (i // (target_width // tile_size)) * tile_size,
                ((i % (target_width // tile_size)) + 1) * tile_size,
                ((i // (target_width // tile_size)) + 1) * tile_size,
            )
            # split the image
            split_img = crop(image_used_to_split, box[0], box[1], box[2], box[3])
            processed_tiles.append(split_img)
        assert len(processed_tiles) == blocks

        if use_thumbnail and len(processed_tiles) != 1:
            thumbnail_img = F.resize(image, (tile_size, tile_size), interpolation=interpolation)
            processed_tiles.append(thumbnail_img)

        return processed_tiles

    def _pad_for_batching(
        self,
        pixel_values: list["torch.Tensor"],
    ) -> list["torch.Tensor"]:
        """
        Pads images on the `num_of_patches` dimension with zeros to form a batch of same number of patches.

        Args:
            pixel_values (`List[torch.Tensor]`):
                An array of pixel values of each images of shape (`batch_size`, `num_patches`, `image_in_3D`)

        Returns:
            List[`torch.Tensor`]: The padded images.
        """
        max_patch = max(len(x) for x in pixel_values)
        pixel_values = [
            torch.nn.functional.pad(image, pad=[0, 0, 0, 0, 0, 0, 0, max_patch - image.shape[0]])
            for image in pixel_values
        ]

        return pixel_values

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        max_dynamic_tiles: int,
        min_dynamic_tiles: int,
        use_thumbnail: bool,
        pad_during_tiling: bool,
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
        pad_size: SizeDict | None = None,  # Added for transformers >=4.53.0 compatibility
        disable_grouping: bool | None = None,  # Added for transformers >=4.53.0 compatibility
    ) -> BatchFeature:
        processed_images = []
        image_sizes = []
        # Determine the size tuple
        if size and size.height and size.width:
            size_tuple = (size.height, size.width)
        else:
            size_tuple = (size.shortest_edge, size.shortest_edge)

        # Determine the patch size
        if crop_size and crop_size.height:
            tile_size = crop_size.height
        elif size and size.height:
            tile_size = size.height
        else:
            tile_size = size.shortest_edge

        for image in images:
            image_patches = self._get_image_patches(
                image,
                min_num=min_dynamic_tiles,
                max_num=max_dynamic_tiles,
                size=size_tuple,
                tile_size=tile_size,
                use_thumbnail=use_thumbnail,
                interpolation=interpolation,
                pad_during_tiling=pad_during_tiling,
            )

            # Group images by size for batched processing
            processed_image_patches_grouped = {}
            # Added for transformers >=4.53.0 compatibility
            grouped_image_patches, grouped_image_patches_index = group_images_by_shape(
                image_patches,
                disable_grouping=disable_grouping,
            )

            for shape, stacked_image_patches in grouped_image_patches.items():
                if do_resize:
                    stacked_image_patches = self.resize(
                        image=stacked_image_patches,
                        size=size,
                        interpolation=interpolation,
                    )
                if do_center_crop:
                    stacked_image_patches = self.center_crop(stacked_image_patches, crop_size)
                # Fused rescale and normalize
                stacked_image_patches = self.rescale_and_normalize(
                    stacked_image_patches,
                    do_rescale,
                    rescale_factor,
                    do_normalize,
                    image_mean,
                    image_std,
                )
                processed_image_patches_grouped[shape] = stacked_image_patches
            processed_image_patches = reorder_images(
                processed_image_patches_grouped, grouped_image_patches_index
            )
            processed_image_patches = (
                torch.stack(processed_image_patches, dim=0) if return_tensors else processed_image_patches
            )
            processed_images.append(processed_image_patches)
            image_sizes.append(get_image_size(image, ChannelDimension.FIRST))

        if do_pad:
            processed_images = self._pad_for_batching(processed_images)

        # processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images
        processed_images = torch.cat(processed_images, dim=0) if return_tensors else processed_images
        return BatchFeature(
            data={"pixel_values": processed_images, "image_sizes": image_sizes},
            tensor_type=return_tensors,
        )

    def preprocess(
        self,
        images: ImageInput,
        videos: VideoInput = None,
        **kwargs: Unpack[Eagle25VLFastImageProcessorKwargs],
    ) -> BatchFeature:
        validate_kwargs(
            captured_kwargs=kwargs.keys(),
            valid_processor_keys=self.valid_kwargs.__annotations__.keys(),
        )
        # Set default kwargs from self. This ensures that if a kwarg is not provided
        # by the user, it gets its default value from the instance, or is set to None.
        for kwarg_name in self.valid_kwargs.__annotations__:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        # Extract parameters that are only used for preparing the input images
        do_convert_rgb = kwargs.pop("do_convert_rgb")
        input_data_format = kwargs.pop("input_data_format")
        device = kwargs.pop("device")
        # Prepare input images
        # transformers >= 4.53.0: uses _prepare_image_like_inputs instead of _prepare_input_images
        if images is not None:
            images = self._prepare_image_like_inputs(
                images=images,
                do_convert_rgb=do_convert_rgb,
                input_data_format=input_data_format,
                device=device,
            )

        if videos is not None:
            videos = self._prepare_image_like_inputs(
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
        # Added for transformers >=4.53.0 compatibility
        resample = kwargs.pop("resample", self.resample)
        kwargs["interpolation"] = (
            pil_torch_interpolation_mapping[resample]
            if isinstance(resample, PILImageResampling | int)
            else resample
        )

        # Filter kwargs to only include those accepted by _preprocess
        valid_preprocess_kwargs = {
            "do_resize",
            "size",
            "max_dynamic_tiles",
            "min_dynamic_tiles",
            "use_thumbnail",
            "pad_during_tiling",
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
            "pad_size",
            "disable_grouping",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_preprocess_kwargs}
        if images is not None:
            return self._preprocess(images, **filtered_kwargs)
        elif videos is not None:
            return self._preprocess(videos, **filtered_kwargs)


__all__ = ["Eagle25VLImageProcessorFast"]
