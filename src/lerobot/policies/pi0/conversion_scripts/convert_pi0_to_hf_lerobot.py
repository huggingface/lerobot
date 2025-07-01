# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Convert pi0 parameters from Jax to Pytorch

Follow [README of openpi](https://github.com/Physical-Intelligence/openpi) to create a new environment
and install the required libraries.

```bash
cd ~/code/openpi
source .venv/bin/activate
```

Example downloading parameters:
```bash
python
>>> import openpi.shared.download as download
>>> path='s3://openpi-assets/checkpoints/pi0_base/params'
>>> download.maybe_download(path)
```

Converting pi0_base:
```python
python -m lerobot.policies.pi0.conversion_scripts.convert_pi0_to_hf_lerobot \
    --checkpoint_dir /home/remi_cadene/.cache/openpi/openpi-assets/checkpoints/pi0_base/params \
    --output_path /home/remi_cadene/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch
```

```python
python -m lerobot.policies.pi0.conversion_scripts.convert_pi0_to_hf_lerobot \
    --checkpoint_dir /home/remi_cadene/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim/params \
    --output_path /home/remi_cadene/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim_pytorch
```
"""

import argparse
import pathlib

import jax
import numpy as np
import orbax.checkpoint as ocp
import torch
from jax.sharding import SingleDeviceSharding

from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.conversion_scripts.conversion_utils import (
    get_gemma_config,
    get_paligemma_config,
)
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

PRECISIONS = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}


def slice_paligemma_state_dict(state_dict, config):
    suffix = "/value" if "img/embedding/kernel/value" in state_dict else ""

    # fmt: off
    # patch embeddings
    state_dict["paligemma.vision_tower.vision_model.embeddings.patch_embedding.weight"] = state_dict.pop(f"img/embedding/kernel{suffix}").transpose(
        3, 2, 0, 1
    )
    state_dict["paligemma.vision_tower.vision_model.embeddings.patch_embedding.bias"] = state_dict.pop(f"img/embedding/bias{suffix}")
    # positional embeddings
    state_dict["paligemma.vision_tower.vision_model.embeddings.position_embedding.weight"] = state_dict.pop(f"img/pos_embedding{suffix}").reshape(
        -1, config.vision_config.hidden_size
    )

    # extract vision layers to be sliced at index 0. There are 27 layers in the base model.
    encoderblock_layernorm0_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/scale{suffix}")
    encoderblock_layernorm0_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/bias{suffix}")
    encoderblock_layernorm1_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/scale{suffix}")
    encoderblock_layernorm1_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/bias{suffix}")

    encoderblock_mlp_dense0_kernel= state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel{suffix}")
    encoderblock_mlp_dense0_bias= state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias{suffix}")
    encoderblock_mlp_dense1_kernel= state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel{suffix}")
    encoderblock_mlp_dense1_bias= state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias{suffix}")

    encoderblock_attention_0_key_kernel = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel{suffix}")
    encoderblock_attention_0_key_bias = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias{suffix}")
    encoderblock_attention_0_value_kernel = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel{suffix}")
    encoderblock_attention_0_value_bias = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias{suffix}")
    encoderblock_attention_0_query_kernel = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel{suffix}")
    encoderblock_attention_0_query_bias = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias{suffix}")
    encoderblock_attention_0_out_kernel = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel{suffix}")
    encoderblock_attention_0_out_bias = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias{suffix}")

    for i in range(config.vision_config.num_hidden_layers):
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight"] = encoderblock_layernorm0_scale[i].transpose()
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.bias"] = encoderblock_layernorm0_bias[i]
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight"] = encoderblock_layernorm1_scale[i].transpose()
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.bias"] = encoderblock_layernorm1_bias[i]

        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight"] = encoderblock_mlp_dense0_kernel[i].transpose()
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.bias"] = encoderblock_mlp_dense0_bias[i]
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.weight"] = encoderblock_mlp_dense1_kernel[i].transpose()
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.bias"] = encoderblock_mlp_dense1_bias[i]
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"] = encoderblock_attention_0_key_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"] = encoderblock_attention_0_key_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"] = encoderblock_attention_0_value_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"] = encoderblock_attention_0_value_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"] = encoderblock_attention_0_query_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"] = encoderblock_attention_0_query_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"] = encoderblock_attention_0_out_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[f"paligemma.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"] = encoderblock_attention_0_out_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)

    state_dict["paligemma.vision_tower.vision_model.post_layernorm.weight"] = state_dict.pop(f"img/Transformer/encoder_norm/scale{suffix}").transpose()
    state_dict["paligemma.vision_tower.vision_model.post_layernorm.bias"] = state_dict.pop(f"img/Transformer/encoder_norm/bias{suffix}")

    # multimodal projector

    state_dict['paligemma.multi_modal_projector.linear.weight'] = state_dict.pop(f"img/head/kernel{suffix}").transpose()
    state_dict['paligemma.multi_modal_projector.linear.bias'] = state_dict.pop(f"img/head/bias{suffix}")

    # text decoder (gemma)
    embedding_vector = state_dict.pop(f"llm/embedder/input_embedding{suffix}")
    state_dict["paligemma.language_model.model.embed_tokens.weight"] = embedding_vector

    # pop the einsum attention + mlp representations. There are 18 layers in gemma-2b.

    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum/w{suffix}")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum/w{suffix}")

    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp/linear{suffix}")
    # TODO verify correctness of layer norm loading

    llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm/scale{suffix}")
    llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm/scale{suffix}")

    for i in range(config.text_config.num_hidden_layers):
        # llm_attention_q_einsum[i].shape = (8, 2048, 256)
        q_proj_weight_reshaped = llm_attention_q_einsum[i].transpose(0, 2, 1).reshape(config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size)

        state_dict[f"paligemma.language_model.model.layers.{i}.self_attn.q_proj.weight"] = q_proj_weight_reshaped

        # llm_attention_kv_einsum[i, 0, 0].shape = (2048, 256)
        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"paligemma.language_model.model.layers.{i}.self_attn.k_proj.weight"] = k_proj_weight_reshaped
        # llm_attention_kv_einsum[i, 1, 0].shape = (2048, 256)
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"paligemma.language_model.model.layers.{i}.self_attn.v_proj.weight"] = v_proj_weight_reshaped

        # output projection.

        # llm_attention_attn_vec_einsum[i].shape = (8, 256, 2048)
        o_proj_weight_reshaped = llm_attention_attn_vec_einsum[i].transpose(2, 0, 1).reshape(config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size)

        state_dict[f"paligemma.language_model.model.layers.{i}.self_attn.o_proj.weight"] = o_proj_weight_reshaped
        # mlp layers
        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"paligemma.language_model.model.layers.{i}.mlp.gate_proj.weight"] = gate_proj_weight.transpose()
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"paligemma.language_model.model.layers.{i}.mlp.up_proj.weight"] = up_proj_weight.transpose()
        state_dict[f"paligemma.language_model.model.layers.{i}.mlp.down_proj.weight"] = llm_mlp_linear[i].transpose()
        state_dict[f"paligemma.language_model.model.layers.{i}.input_layernorm.weight"] = llm_input_layernorm[i]
        state_dict[f"paligemma.language_model.model.layers.{i}.post_attention_layernorm.weight"] = llm_post_attention_layernorm[i]

    state_dict["paligemma.language_model.model.norm.weight"] = state_dict.pop(f"llm/final_norm/scale{suffix}")
    state_dict["paligemma.language_model.lm_head.weight"] = embedding_vector # weights are tied.

    # fmt: on
    expert_dict = {}
    final_state_dict = {}
    for key, value in state_dict.items():
        if key not in [
            f"llm/final_norm_1/scale{suffix}",
            f"llm/layers/attn/attn_vec_einsum_1/w{suffix}",
            f"llm/layers/attn/kv_einsum_1/w{suffix}",
            f"llm/layers/attn/q_einsum_1/w{suffix}",
            f"llm/layers/mlp_1/gating_einsum{suffix}",
            f"llm/layers/mlp_1/linear{suffix}",
            f"llm/layers/pre_attention_norm_1/scale{suffix}",
            f"llm/layers/pre_ffw_norm_1/scale{suffix}",
        ]:
            final_state_dict[key] = torch.from_numpy(value)
        else:
            expert_dict[key] = value

    return final_state_dict, expert_dict


def slice_gemma_state_dict(state_dict, config, num_expert=1):
    # fmt: off
    # text decoder (gemma)
    # no embedding vector, the expert just has the decoder layers

    embedding_vector = torch.zeros([config.vocab_size, config.hidden_size])
    state_dict["gemma_expert.model.embed_tokens.weight"] = embedding_vector

    # pop the einsum attention + mlp representations. There are 18 layers in gemma-2b.

    suffix = "/value" if f"llm/layers/attn/attn_vec_einsum_{num_expert}/w/value" in state_dict else ""

    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum_{num_expert}/w{suffix}")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum_{num_expert}/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum_{num_expert}/w{suffix}")

    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp_{num_expert}/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp_{num_expert}/linear{suffix}")
    # TODO verify correctness of layer norm loading

    llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/scale{suffix}")
    llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/scale{suffix}")

    for i in range(config.num_hidden_layers):
        q_proj_weight_reshaped = llm_attention_q_einsum[i].transpose(0, 2, 1).reshape(config.num_attention_heads * config.head_dim, config.hidden_size)

        state_dict[f"gemma_expert.model.layers.{i}.self_attn.q_proj.weight"] = q_proj_weight_reshaped

        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"gemma_expert.model.layers.{i}.self_attn.k_proj.weight"] = k_proj_weight_reshaped
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"gemma_expert.model.layers.{i}.self_attn.v_proj.weight"] = v_proj_weight_reshaped

        # output projection.

        # llm_attention_attn_vec_einsum[i].shape = (8, 256, 1024)
        o_proj_weight_reshaped = llm_attention_attn_vec_einsum[i].reshape(config.num_attention_heads * config.head_dim, config.hidden_size).transpose(1,0)# .transpose(2, 0, 1).reshape(config.num_attention_heads * config.head_dim, config.hidden_size).transpose(1, 0)

        state_dict[f"gemma_expert.model.layers.{i}.self_attn.o_proj.weight"] = o_proj_weight_reshaped
        # mlp layers
        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"gemma_expert.model.layers.{i}.mlp.gate_proj.weight"] = gate_proj_weight.transpose()
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"gemma_expert.model.layers.{i}.mlp.up_proj.weight"] = up_proj_weight.transpose()
        state_dict[f"gemma_expert.model.layers.{i}.mlp.down_proj.weight"] = llm_mlp_linear[i].transpose()
        state_dict[f"gemma_expert.model.layers.{i}.input_layernorm.weight"] = llm_input_layernorm[i]
        state_dict[f"gemma_expert.model.layers.{i}.post_attention_layernorm.weight"] = llm_post_attention_layernorm[i]

    state_dict["gemma_expert.model.norm.weight"] = state_dict.pop(f"llm/final_norm_{num_expert}/scale{suffix}")
    state_dict["gemma_expert.lm_head.weight"] = embedding_vector # weights are tied. (and zeros here)

    # fmt: on
    final_state_dict = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            final_state_dict[key] = torch.from_numpy(value)
        else:
            final_state_dict[key] = value
    return final_state_dict


def flatten_for_memory(tree, parent_key=""):
    out = {}
    for k, v in tree.items():
        new_key = f"{parent_key}/{k}" if parent_key else k
        if isinstance(v, dict):
            out.update(flatten_for_memory(v, new_key))
        else:
            out[new_key] = np.array(v)  # Ensure conversion to np.array for consistency
    return out


def flatten_for_npz(tree, parent_key=""):
    out = {}
    for k, v in tree.items():
        new_key = f"{parent_key}/{k}" if parent_key else k
        if isinstance(v, dict):
            out.update(flatten_for_npz(v, new_key))
        else:
            # bf16/f32 here?
            out[new_key] = np.array(v)
    return out


def slice_initial_orbax_checkpoint(checkpoint_dir: str):
    params_path = pathlib.Path(checkpoint_dir).resolve()
    checkpointer = ocp.PyTreeCheckpointer()

    metadata = checkpointer.metadata(params_path)
    print("Metadata keys:", list(metadata.keys()))

    params_name = "params"

    item = {params_name: metadata[params_name]}
    device = jax.local_devices()[0]  # Use the first local device
    sharding = SingleDeviceSharding(device)
    restored = checkpointer.restore(
        params_path,
        ocp.args.PyTreeRestore(
            item=item,
            restore_args=jax.tree_util.tree_map(
                lambda _: ocp.ArrayRestoreArgs(
                    restore_type=jax.Array,  # or np.ndarray, but bf16 is annoying about it
                    sharding=sharding,
                ),
                item,
            ),
            transforms={},
        ),
    )
    params = restored[params_name]

    # get params for PaliGemma
    pali_params = params["PaliGemma"]
    del params["PaliGemma"]
    pali_params_flat = flatten_for_npz(pali_params)
    return {"paligemma_params": pali_params_flat, "projection_params": params}


def update_keys_with_prefix(d: dict, prefix: str) -> dict:
    """Update dictionary keys by adding a prefix."""
    return {f"{prefix}{key}": value for key, value in d.items()}


def convert_pi0_checkpoint(checkpoint_dir: str, precision: str, tokenizer_id: str, output_path: str):
    # Break down orbax ckpts - they are in OCDBT
    initial_params = slice_initial_orbax_checkpoint(checkpoint_dir=checkpoint_dir)
    # process projection params
    keys = [
        "state_proj",
        "action_in_proj",
        "action_out_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
    ]

    projection_params = {}
    for key in keys:
        kernel_params = initial_params["projection_params"][key]["kernel"]
        bias_params = initial_params["projection_params"][key]["bias"]
        if isinstance(kernel_params, dict):
            weight = kernel_params["value"]
            bias = bias_params["value"]
        else:
            weight = kernel_params
            bias = bias_params
        projection_params[f"{key}.weight"] = torch.from_numpy(np.array(weight)).T
        projection_params[f"{key}.bias"] = torch.from_numpy(np.array(bias))

    # Process PaliGemma weights
    paligemma_config = get_paligemma_config(precision)
    paligemma_params, gemma_raw_dictionary = slice_paligemma_state_dict(
        initial_params["paligemma_params"], paligemma_config
    )

    # Process Gemma  weights (at this stage they are unused)
    gemma_config = get_gemma_config(precision)
    gemma_params = slice_gemma_state_dict(gemma_raw_dictionary, config=gemma_config)

    # Instantiate model from configs

    if "pi0_aloha_sim" in checkpoint_dir:
        pi0_config = PI0Config(
            empty_cameras=2,
            adapt_to_pi_aloha=True,
            use_delta_joint_actions_aloha=False,
        )
    elif "pi0_aloha_towel" in checkpoint_dir:
        pi0_config = PI0Config(
            adapt_to_pi_aloha=True,
            use_delta_joint_actions_aloha=True,
        )
    elif "pi0_base" in checkpoint_dir:
        pi0_config = PI0Config(
            empty_cameras=0,
            adapt_to_pi_aloha=False,
            use_delta_joint_actions_aloha=False,
        )
    else:
        raise ValueError()

    # gemma_config=gemma_config, paligemma_config=paligemma_config)
    pi0_model = PI0Policy(pi0_config)

    paligemma_params = update_keys_with_prefix(paligemma_params, "model.paligemma_with_expert.")
    gemma_params = update_keys_with_prefix(gemma_params, "model.paligemma_with_expert.")
    projection_params = update_keys_with_prefix(projection_params, "model.")

    # load state dict
    torch_dtype = PRECISIONS[precision]
    pi0_model.load_state_dict({**paligemma_params, **gemma_params, **projection_params})
    pi0_model = pi0_model.to(torch_dtype)
    # pi0_tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    pi0_model.save_pretrained(output_path, safe_serialization=True)
    # pi0_tokenizer.save_pretrained(output_path, dtype=torch_dtype)

    # assert that model loads properly
    del pi0_model
    PI0Policy.from_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        default="/raid/pablo/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim/params",
        type=str,
        help="Path to the ocdbt checkpoint",
    )

    parser.add_argument(
        "--precision",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        type=str,
        help="Precision identifier for model conversion - should match the base checkpoint precision.",
    )
    # tokenizer is identical to paligemma, it appears

    parser.add_argument(
        "--tokenizer_hub_id",
        default="google/paligemma-3b-pt-224",
        type=str,
        help="Hub path to the tokenizer to save",
    )

    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="Path to save converted weights to",
    )

    args = parser.parse_args()
    convert_pi0_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        precision=args.precision,
        tokenizer_id=args.tokenizer_hub_id,
        output_path=args.output_path,
    )
