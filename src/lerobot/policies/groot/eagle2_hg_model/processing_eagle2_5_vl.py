# Copyright 2024 The HuggingFace Inc. team.
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
"""
Processor class for Eagle25VL.
copy from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_onevision/processing_llava_onevision.py
"""

import base64
import os
import re
from io import BytesIO

import requests
import torch
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging
from transformers.video_utils import VideoInput

logger = logging.get_logger(__name__)


FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 256


def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == "RGBA":
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")


def fetch_image(ele: dict[str, str | Image.Image]) -> Image.Image:
    image = ele["image"] if "image" in ele else ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        response = requests.get(image, stream=True, timeout=10)
        image_obj = Image.open(BytesIO(response.content))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(
            f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}"
        )
    image = to_rgb(image_obj)
    if "scale_factor" in ele:
        scale_factor = ele["scale_factor"]
        image = image.resize((image.width * scale_factor, image.height * scale_factor), Image.BILINEAR)
    return image


class Eagle25VLProcessorKwargs(ProcessingKwargs, total=False):
    # see processing_utils.ProcessingKwargs documentation for usage.
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {},
        "videos_kwargs": {"max_dynamic_tiles": 1},
    }


class Eagle25VLProcessor(ProcessorMixin):
    r"""
    Constructs a Eagle25VL processor which wraps a Eagle25VL video processor, Eagle25VL image processor and a Eagle25VL tokenizer into a single processor.

    [`Eagle25VLProcessor`] offers all the functionalities of [`Eagle25VLVideoProcessor`], [`Eagle25VLImageProcessor`] and [`Eagle25VLTokenizer`]. See the
    [`~Eagle25VLVideoProcessor.__call__`], [`~Eagle25VLProcessor.__call__`] and [`~Eagle25VLProcessor.decode`] for more information.

    Args:
        image_processor ([`LlavaOnevisionImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        num_image_tokens (`int`, *optional*):
            Number of image tokens for one imagethat will be returned by vision tower.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Should be same as in model's config
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
        video_token (`str`, *optional*, defaults to `"<video>"`):
            Special token used to denote video location.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = [
        "chat_template",
        "num_image_tokens",
        "vision_feature_select_strategy",
        "image_token",
        "video_token",
        "images_kwargs",
        "videos_kwargs",
        "text_kwargs",
    ]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        vision_feature_select_strategy=None,
        chat_template=None,
        image_token="<IMG_CONTEXT>",  # nosec: B107
        video_token="<IMG_CONTEXT>",  # nosec: B107
        tokens_per_tile=256,
        image_placeholder="image",
        video_placeholder="video",
        image_start_token="<img>",
        image_end_token="</img>",
        **kwargs,
    ):
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        self.video_token = tokenizer.video_token if hasattr(tokenizer, "video_token") else video_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        self.image_placeholder = image_placeholder
        self.video_placeholder = video_placeholder
        self.tokens_per_tile = tokens_per_tile
        self.image_start_token = image_start_token
        self.image_end_token = image_end_token
        if "auto_map" in kwargs:
            self.auto_map = kwargs["auto_map"]
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def replace_media_placeholder(
        self, text, image_list, video_list, timestamps_list, fps_list, **output_kwargs
    ):
        num_of_images_in_this_sample = 0
        num_of_videos_in_this_sample = 0
        # Regular expression pattern to match formats like <image-1> or <video-2>
        pattern = re.compile(rf"<({self.image_placeholder}|{self.video_placeholder})-(\d+)>")
        unified_frame_list = []

        # image_min_dynamic_tiles = output_kwargs["images_kwargs"].get(
        #     "min_dynamic_tiles", self.image_processor.min_dynamic_tiles
        # )
        # image_max_dynamic_tiles = output_kwargs["images_kwargs"].get(
        #     "max_dynamic_tiles", self.image_processor.max_dynamic_tiles
        # )
        # image_use_thumbnail = output_kwargs["images_kwargs"].get(
        #     "use_thumbnail", self.image_processor.use_thumbnail
        # )
        video_min_dynamic_tiles = output_kwargs["videos_kwargs"].get(
            "min_dynamic_tiles", self.image_processor.min_dynamic_tiles
        )
        video_max_dynamic_tiles = output_kwargs["videos_kwargs"].get(
            "max_dynamic_tiles", self.image_processor.max_dynamic_tiles
        )
        video_use_thumbnail = output_kwargs["videos_kwargs"].get(
            "use_thumbnail", self.image_processor.use_thumbnail
        )

        tile_size = self.image_processor.size.get("height", 448)

        # Function to replace tags in a single text
        def replace_in_text(text):
            # repl callback function for each match replacement operation
            def repl(match):
                nonlocal unified_frame_list
                nonlocal num_of_images_in_this_sample
                nonlocal num_of_videos_in_this_sample
                media_type = match.group(1)  # 'image' or 'video'
                idx_in_list = int(match.group(2)) - 1  # Convert to list index (0-based)
                # Select the corresponding path based on media type
                idx_mapper = {
                    0: "first",
                    1: "second",
                    2: "third",
                    3: "fourth",
                    4: "fifth",
                    5: "sixth",
                    6: "seventh",
                    7: "eighth",
                    8: "ninth",
                    9: "tenth",
                }
                if media_type == "image":
                    image_inputs = self.image_processor(
                        images=[image_list[idx_in_list]],
                        videos=None,
                        **output_kwargs["images_kwargs"],
                    )
                    num_all_tiles = image_inputs["pixel_values"].shape[0]
                    special_placeholder = f"<image {idx_in_list + 1}>{self.image_start_token}{self.image_token * num_all_tiles * self.tokens_per_tile}{self.image_end_token}"
                    unified_frame_list.append(image_inputs)
                    num_of_images_in_this_sample += 1

                elif media_type == "video":
                    video_inputs = self.image_processor(
                        images=None,
                        videos=[video_list[idx_in_list]],
                        **output_kwargs["videos_kwargs"],
                    )
                    num_all_tiles = video_inputs["pixel_values"].shape[0]
                    image_sizes = video_inputs["image_sizes"]
                    if timestamps_list is not None and -1 not in timestamps_list:
                        frame_timestamps = timestamps_list[idx_in_list]
                    else:
                        frame_timestamps = None
                    sampled_fps = fps_list[idx_in_list] if fps_list is not None else None

                    num_of_tiles_each_frame = [
                        self.get_number_tiles_based_on_image_size(
                            image_size,
                            video_min_dynamic_tiles,
                            video_max_dynamic_tiles,
                            video_use_thumbnail,
                            tile_size,
                        )
                        for image_size in image_sizes
                    ]
                    assert sum(num_of_tiles_each_frame) == num_all_tiles, (
                        f"The number of tiles in each frame is not equal to the total number of tiles: {sum(num_of_tiles_each_frame)} != {num_all_tiles}"
                    )

                    if frame_timestamps is not None:
                        assert len(frame_timestamps) == len(num_of_tiles_each_frame), (
                            f"The number of timestamps is not equal to the number of frames: {len(frame_timestamps)} != {len(num_of_tiles_each_frame)}"
                        )
                        special_placeholder = [
                            f"Frame {i + 1} sample at {frame_timestamps[i]:.2f}s: {self.image_start_token}{self.image_token * num_of_tiles * self.tokens_per_tile}{self.image_end_token}"
                            for i, num_of_tiles in enumerate(num_of_tiles_each_frame)
                        ]
                    else:
                        special_placeholder = [
                            f"Frame {i + 1}: {self.image_start_token}{self.image_token * num_of_tiles * self.tokens_per_tile}{self.image_end_token}"
                            for i, num_of_tiles in enumerate(num_of_tiles_each_frame)
                        ]

                    if sampled_fps is not None:
                        special_placeholder = (
                            f"The {idx_mapper[idx_in_list]} video sampled with {sampled_fps:.2f} fps: "
                            + "".join(special_placeholder)
                        )
                    else:
                        special_placeholder = f"The {idx_mapper[idx_in_list]} video: " + "".join(
                            special_placeholder
                        )
                    unified_frame_list.append(video_inputs)
                    num_of_videos_in_this_sample += 1
                else:
                    raise ValueError(f"Unknown media type: {media_type}")
                return special_placeholder

            return pattern.sub(repl, text)

        text = replace_in_text(text)
        if len(unified_frame_list) > 0:
            pixel_values = torch.cat([frame["pixel_values"] for frame in unified_frame_list])
            image_sizes = torch.cat([frame["image_sizes"] for frame in unified_frame_list])
        else:
            pixel_values = None
            image_sizes = None
        return (
            text,
            pixel_values,
            image_sizes,
            num_of_images_in_this_sample,
            num_of_videos_in_this_sample,
        )

    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        audio=None,
        videos: VideoInput = None,
        **kwargs: Unpack[Eagle25VLProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        LlavaNextImageProcessor's [`~LlavaNextImageProcessor.__call__`] if `images` is not `None`. Please refer to the docstring
        of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of a video input to be fed to a model. Returned when `videos` is not `None`.
            - **image_sizes** -- Size of each image that will be used to unpad an image. Returned when `images` is not `None`.
        """

        output_kwargs = self._merge_kwargs(
            Eagle25VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text_list = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")
        elif isinstance(text, list) and isinstance(text[0], str):
            text_list = text

        if images is None:
            images = []
        if videos is None:
            videos = []

        pixel_values_list = []
        image_sizes_list = []
        new_sample_list = []
        image_start_idx = 0
        video_start_idx = 0
        timestamps_batch = output_kwargs["videos_kwargs"].pop("timestamps", None)
        fps_batch = output_kwargs["videos_kwargs"].pop("fps", None)
        for sample in text_list:
            timestamps_list = timestamps_batch[video_start_idx:] if timestamps_batch is not None else None
            fps_list = fps_batch[video_start_idx:] if fps_batch is not None else None
            (
                sample,
                pixel_values,
                image_sizes,
                num_of_images_in_this_sample,
                num_of_videos_in_this_sample,
            ) = self.replace_media_placeholder(
                sample,
                images[image_start_idx:],
                videos[video_start_idx:],
                timestamps_list,
                fps_list,
                **output_kwargs,
            )
            new_sample_list.append(sample)
            if pixel_values is not None:
                pixel_values_list.append(pixel_values)
                image_sizes_list.append(image_sizes)
            image_start_idx += num_of_images_in_this_sample
            video_start_idx += num_of_videos_in_this_sample

        if len(pixel_values_list) > 0:
            image_inputs = {
                "pixel_values": torch.cat(pixel_values_list),
                "image_sizes": torch.cat(image_sizes_list),
            }
        else:
            image_inputs = {}
        video_inputs = {}
        text_inputs = self.tokenizer(new_sample_list, **output_kwargs["text_kwargs"])
        return BatchFeature(data={**text_inputs, **image_inputs, **video_inputs})

    def get_number_tiles_based_on_image_size(
        self, image_size: tuple, min_num: int, max_num: int, use_thumbnail: bool, tile_size: int
    ) -> int:
        """
        Get the number of tiles based on the image size.
        """
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
        target_aspect_ratio = self.image_processor.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, tile_size
        )
        tiles_num = target_aspect_ratio[0] * target_aspect_ratio[1]
        if use_thumbnail and tiles_num > 1:
            tiles_num += 1
        return tiles_num

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    # override to save video-config in a separate config file
    def save_pretrained(self, save_directory, **kwargs):
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)

        outputs = super().save_pretrained(save_directory, **kwargs)
        return outputs

    # override to load video-config from a separate config file
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # if return_unused_kwargs a tuple is returned where the second element is 'unused_kwargs'
        if isinstance(processor, tuple):
            processor = processor[0]
        return processor

    # Copy from https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
    def process_vision_info(
        self,
        conversations: list[dict] | list[list[dict]],
        return_video_kwargs: bool = False,
    ) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, dict | None]:
        vision_infos = self.extract_vision_info(conversations)
        ## Read images or videos
        image_inputs = []
        video_inputs = []
        video_sample_fps_list = []
        video_timestamps_list = []
        for vision_info in vision_infos:
            if "image" in vision_info or "image_url" in vision_info:
                image_inputs.append(fetch_image(vision_info))
            else:
                raise ValueError("image, image_url or video should in content.")
        if len(image_inputs) == 0:
            image_inputs = None
        if len(video_inputs) == 0:
            video_inputs = None
        if return_video_kwargs:
            return (
                image_inputs,
                video_inputs,
                {"fps": video_sample_fps_list, "timestamps": video_timestamps_list},
            )
        return image_inputs, video_inputs

    def extract_vision_info(self, conversations: list[dict] | list[list[dict]]) -> list[dict]:
        vision_infos = []
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        for conversation in conversations:
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if (
                            "image" in ele
                            or "image_url" in ele
                            or "video" in ele
                            or ele["type"] in ("image", "image_url", "video")
                        ):
                            vision_infos.append(ele)
        return vision_infos


__all__ = ["Eagle25VLProcessor"]
