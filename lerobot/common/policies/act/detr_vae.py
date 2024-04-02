import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from .backbone import build_backbone
from .transformer import TransformerEncoder, TransformerEncoderLayer, build_transformer


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class ActionChunkingTransformer(nn.Module):
    """
    Action Chunking Transformer as per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
    (https://arxiv.org/abs/2304.13705).

    Note: In this code we use the symbols `vae_encoder`, 'encoder', `decoder`. The meanings are as follows.
        - The `vae_encoder` is, as per the literature around conditional variational auto-encoders (cVAE), the
          part of the model that encodes the target data (here, a sequence of actions), and the condition
          (here, we include the robot joint-space state as an input to the encoder).
        - The `transformer` is the cVAE's decoder. But since we have an option to train this model without the
          variational objective (in which case we drop the `vae_encoder` altogether), we don't call it the
          `vae_decoder`.
        # TODO(now): remove the following
        - The `encoder` is actually a component of the cVAE's "decoder". But we refer to it as an "encoder"
          because, in terms of the transformer with cross-attention that forms the cVAE's decoder, it is the
          "encoder" part. We drop the `vae_` prefix because we have an option to train this model without the
          variational objective (in which case we drop the `vae_encoder` altogether), and nothing about this
          model has anything to do with a VAE).
        - The `decoder` is a building block of the VAE decoder, and is just the "decoder" part of a
          transformer with cross-attention. For the same reasoning behind the naming of `encoder`, we make
          this term agnostic to the option to use a variational objective for training.

    """

    def __init__(
        self, backbones, transformer, vae_encoder, state_dim, action_dim, horizon, camera_names, vae
    ):
        """Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            horizon: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.

        Args:
            state_dim: Robot positional state dimension.
            action_dim: Action dimension.
            horizon: The number of actions to generate in one forward pass.
            vae: Whether to use the variational objective. TODO(now): Give more details.
        """
        super().__init__()
        self.camera_names = camera_names
        self.transformer = transformer
        self.vae_encoder = vae_encoder
        self.vae = vae
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        # Positional embedding to be used as input to the latent vae_encoder (if applicable) and for the
        self.pos_embed = nn.Embedding(horizon, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            # TODO(rcadene): understand what is env_state, and why it needs to be 7
            self.input_proj_env_state = nn.Linear(state_dim // 2, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # vae_encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.vae_encoder_action_proj = nn.Linear(14, hidden_dim)  # project action to embedding
        self.vae_encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(
            hidden_dim, self.latent_dim * 2
        )  # project hidden state to latent std, var
        self.register_buffer(
            "pos_table", get_sinusoid_encoding_table(1 + 1 + horizon, hidden_dim)
        )  # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(
            2, hidden_dim
        )  # learned position embedding for proprio and latent

    def forward(self, qpos, image, env_state, actions=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None  # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if self.vae and is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.vae_encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            qpos_embed = self.vae_encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)  # (bs, 1, hidden_dim)
            vae_encoder_input = torch.cat(
                [cls_embed, qpos_embed, action_embed], axis=1
            )  # (bs, seq+1, hidden_dim)
            vae_encoder_input = vae_encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
            # do not mask cls token
            # cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device)  # False: not a padding
            # is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            vae_encoder_output = self.vae_encoder(
                vae_encoder_input, pos=pos_embed
            )  # , src_key_padding_mask=is_pad)
            vae_encoder_output = vae_encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(vae_encoder_output)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, _ in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id])  # HARDCODED
                features = features[0]  # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self.transformer(
                src,
                None,
                self.pos_embed.weight,
                pos,
                latent_input,
                proprio_input,
                self.additional_pos_embed.weight,
            )[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)  # seq length = 2
            hs = self.transformer(transformer_input, None, self.pos_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_vae_encoder(args):
    d_model = args.hidden_dim  # 256
    dropout = args.dropout  # 0.1
    nhead = args.nheads  # 8
    dim_feedforward = args.dim_feedforward  # 2048
    num_encoder_layers = args.enc_layers  # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm  # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(
        d_model, nhead, dim_feedforward, dropout, activation, normalize_before
    )
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)

    transformer = build_transformer(args)

    vae_encoder = build_vae_encoder(args)

    model = ActionChunkingTransformer(
        backbones,
        transformer,
        vae_encoder,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        horizon=args.num_queries,
        camera_names=args.camera_names,
        vae=args.vae,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: {:.2f}M".format(n_parameters / 1e6))

    return model
