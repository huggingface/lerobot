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
    ("model.query_embed.weight", "model.pos_embed.weight"),
    ("model.pos_table", "model.vae_encoder_pos_enc"),
    ("model.pos_embed.weight", "model.decoder_pos_embed.weight"),
    ("model.encoder.", "model.vae_encoder."),
    ("model.encoder_action_proj.", "model.vae_encoder_action_input_proj."),
    ("model.encoder_joint_proj.", "model.vae_encoder_robot_state_input_proj."),
    ("model.latent_proj.", "model.vae_encoder_latent_output_proj."),
    ("model.latent_proj.", "model.vae_encoder_latent_output_proj."),
    ("model.input_proj.", "model.encoder_img_feat_input_proj."),
    ("model.input_proj_robot_state", "model.encoder_robot_state_input_proj"),
    ("model.latent_out_proj.", "model.encoder_latent_input_proj."),
    ("model.transformer.encoder.", "model.encoder."),
    ("model.transformer.decoder.", "model.decoder."),
    ("model.backbones.0.0.body.", "model.backbone."),
    ("model.additional_pos_embed.weight", "model.encoder_robot_and_latent_pos_embed.weight"),
    ("model.cls_embed.weight", "model.vae_encoder_cls_embed.weight"),
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
