#!/usr/bin/env python

# Copyright 2025 DexVLA Team and The HuggingFace Inc. team. All rights reserved.
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

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import fetch_image


class Qwen2VLAProcess:
    def __init__(
        self,
        tokenizer=None,
        max_seq_len=512,
        multimodal_processor=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.multimodal_processor = multimodal_processor

    def qwen2_image_preprocess(self, each):
        ele = {}
        each = Image.fromarray(each.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        ele["image"] = each

        ele["resized_height"] = each.height
        ele["resized_width"] = each.width
        each = fetch_image(ele)
        return torch.from_numpy(np.array(each))

    def single_forward_process(self, images, raw_lang, reasoning, eval=False, use_reasoning=True):
        len_views = images.shape[0]
        messages = self.construct_chat_data(len_views, raw_lang)

        data_dict = {"messages": messages}

        image_data = torch.chunk(images, len_views, 0)

        images_list = []

        for _i, each in enumerate(image_data):
            img_pil = self.qwen2_image_preprocess(each)
            images_list.append(img_pil)

        text = self.multimodal_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.multimodal_processor(
            text=text,
            images=images_list,
            videos=None,
            padding=True,
            return_tensors="pt",
        )

        if eval:
            new_dict = {}
            for k, v in model_inputs.items():
                if "image_grid" in k:
                    new_dict["image_grid_spatiotemporal"] = v
                else:
                    new_dict[k] = v
            return new_dict

        input_labels = torch.ones_like(model_inputs["input_ids"]) * -100
        answer = reasoning + " Next action:" + "<|im_end|>" if use_reasoning else "" + "<|im_end|>"

        output_text = self.tokenizer(answer, padding=True, return_tensors="pt")
        output_labels = output_text["input_ids"]
        model_inputs["input_ids"] = torch.cat((model_inputs["input_ids"], output_text["input_ids"]), dim=-1)
        model_inputs["attention_mask"] = torch.cat(
            (model_inputs["attention_mask"], output_text["attention_mask"]), dim=-1
        )
        labels = torch.cat((input_labels, output_labels), dim=-1)

        data_dict["labels"] = labels
        for k, v in model_inputs.items():
            if "image_grid" in k:
                data_dict["image_grid_spatiotemporal"] = v
            else:
                data_dict[k] = v
        return data_dict

    def forward(self, batch, use_reasoning=True):
        """This is the main process function for processing vl data into Qwen2_vl format"""
        all_images = batch["images"]
        all_images = torch.einsum(
            "v b c h w -> b v c h w", all_images
        )  # camera_views, batch_size, channel, height, width

        ret_l = []

        for idx, images in enumerate(all_images):
            raw_lang = batch["raw_langs"][idx]
            reasoning = batch["reasonings"][idx]
            ret_dict = self.single_forward_process(images, raw_lang, reasoning, use_reasoning=use_reasoning)
            ret_l.append(ret_dict)

        return self.post_process(ret_l)

    def post_process(self, instances):
        input_ids = [torch.flip(instance["input_ids"].squeeze(0), dims=[0]) for instance in instances]
        labels = [torch.flip(instance["labels"].squeeze(0), dims=[0]) for instance in instances]

        image_grid_spatiotemporal = torch.stack(
            [instances["image_grid_spatiotemporal"] for instances in instances]
        )
        pixel_values = torch.stack([instances["pixel_values"] for instances in instances])
        pixel_values_videos = None
        video_grid_spatiotemporal = None

        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        labels = torch.flip(labels, dims=[1])
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        input_ids = torch.flip(input_ids, dims=[1])
        b = input_ids.shape[0]

        image_grid_spatiotemporal = image_grid_spatiotemporal.reshape(
            b * image_grid_spatiotemporal.shape[1], image_grid_spatiotemporal.shape[2]
        )
        pixel_values = pixel_values.reshape(b * pixel_values.shape[1], pixel_values.shape[2])

        attention_mask = (input_ids.ne(self.tokenizer.pad_token_id),)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask[0],
            "labels": labels,
            "image_grid_spatiotemporal": image_grid_spatiotemporal,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_spatiotemporal": video_grid_spatiotemporal,
            "pixel_values": pixel_values,
        }

        return batch

    def construct_chat_data(self, len_image, raw_lang):
        messages = [
            {
                "role": "user",
                "content": [],
            },
        ]

        for _i in range(len_image):
            messages[0]["content"].append(
                {
                    "type": "image",
                    "image": None,
                }
            )
        messages[0]["content"].append({"type": "text", "text": ""})
        messages[0]["content"][-1]["text"] = raw_lang

        return messages
