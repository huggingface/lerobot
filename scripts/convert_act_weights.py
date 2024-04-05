import torch

from lerobot.common.policies.factory import make_policy
from lerobot.common.utils import init_hydra_config

cfg = init_hydra_config(
    "/home/alexander/Projects/lerobot/outputs/train/act_aloha_sim_transfer_cube_human/.hydra/config.yaml"
)

policy = make_policy(cfg)

state_dict = torch.load("/home/alexander/Projects/act/outputs/sim_transfer_cube_human_vae/policy_last.ckpt")

# Remove keys based on what they start with.

start_removals = [
    # There is a bug that means the pretrained model doesn't even use the final decoder layers.
    *[f"model.transformer.decoder.layers.{i}" for i in range(1, 7)],
    "model.is_pad_head.",
]

for to_remove in start_removals:
    for k in list(state_dict.keys()):
        if k.startswith(to_remove):
            del state_dict[k]


# Replace keys based on what they start with.

start_replacements = [
    ("model.", ""),
    ("query_embed.weight", "pos_embed.weight"),
    ("pos_table", "vae_encoder_pos_enc"),
    ("pos_embed.weight", "decoder_pos_embed.weight"),
    ("encoder.", "vae_encoder."),
    ("encoder_action_proj.", "vae_encoder_action_input_proj."),
    ("encoder_joint_proj.", "vae_encoder_robot_state_input_proj."),
    ("latent_proj.", "vae_encoder_latent_output_proj."),
    ("latent_proj.", "vae_encoder_latent_output_proj."),
    ("input_proj.", "encoder_img_feat_input_proj."),
    ("input_proj_robot_state", "encoder_robot_state_input_proj"),
    ("latent_out_proj.", "encoder_latent_input_proj."),
    ("transformer.encoder.", "encoder."),
    ("transformer.decoder.", "decoder."),
    ("backbones.0.0.body.", "backbone."),
    ("additional_pos_embed.weight", "encoder_robot_and_latent_pos_embed.weight"),
    ("cls_embed.weight", "vae_encoder_cls_embed.weight"),
]

for to_replace, replace_with in start_replacements:
    for k in list(state_dict.keys()):
        if k.startswith(to_replace):
            k_ = replace_with + k.removeprefix(to_replace)
            state_dict[k_] = state_dict[k]
            del state_dict[k]


missing_keys, unexpected_keys = policy.load_state_dict(state_dict, strict=False)

if len(missing_keys) != 0:
    print("MISSING KEYS")
    print(missing_keys)
if len(unexpected_keys) != 0:
    print("UNEXPECTED KEYS")
    print(unexpected_keys)

# if len(missing_keys) != 0 or len(unexpected_keys) != 0:
#     print("Failed due to mismatch in state dicts.")
#     exit()

policy.save("/tmp/weights.pth")
