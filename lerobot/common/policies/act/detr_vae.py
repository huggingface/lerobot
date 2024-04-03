import einops
import numpy as np
import torch
from torch import nn

from .backbone import build_backbone
from .transformer import TransformerEncoder, TransformerEncoderLayer, build_transformer


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
    (paper: https://arxiv.org/abs/2304.13705, code: https://github.com/tonyzhaozh/act)

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
        self, backbones, transformer, vae_encoder, state_dim, action_dim, horizon, camera_names, use_vae
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
            use_vae: Whether to use the variational objective. TODO(now): Give more details.
        """
        super().__init__()

        self.camera_names = camera_names
        self.transformer = transformer
        self.vae_encoder = vae_encoder
        self.use_vae = use_vae
        hidden_dim = transformer.d_model

        # BERT style VAE encoder with input [cls, *joint_space_configuration, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        if use_vae:
            self.cls_embed = nn.Embedding(1, hidden_dim)
            # Projection layer for joint-space configuration to hidden dimension.
            self.vae_encoder_robot_state_input_proj = nn.Linear(state_dim, hidden_dim)
            # Projection layer for action (joint-space target) to hidden dimension.
            self.vae_encoder_action_input_proj = nn.Linear(state_dim, hidden_dim)
            # Final size of latent z. TODO(now): Add to hyperparams.
            self.latent_dim = 32
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
            # Fixed sinusoidal positional embedding the whole input to the VAE encoder.
            self.register_buffer(
                "vae_encoder_pos_enc", get_sinusoid_encoding_table(1 + 1 + horizon, hidden_dim)
            )

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, robot_state, image_feature_map_pixels].
        self.backbones = nn.ModuleList(backbones)
        self.encoder_img_feat_input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
        self.encoder_robot_state_input_proj = nn.Linear(state_dim, hidden_dim)
        self.encoder_latent_input_proj = nn.Linear(self.latent_dim, hidden_dim)
        # TODO(now): Fix this nonsense. One positional embedding is needed. We should extract the image
        # feature dimension with a dry run.
        self.additional_pos_embed = nn.Embedding(
            2, hidden_dim
        )  # learned position embedding for proprio and latent

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(horizon, hidden_dim)
        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, robot_state, image, actions=None):
        """
        Args:
            robot_state: (B, J) batch of robot joint configurations.
            image: (B, N, C, H, W) batch of N camera frames.
            actions: (B, S, A) batch of actions from the target dataset which must be provided if the
                VAE is enabled and the model is in training mode.
        """
        if self.use_vae and self.training:
            assert (
                actions is not None
            ), "actions must be provided when using the variational objective in training mode."

        batch_size, _ = robot_state.shape

        # Prepare the latent for input to the transformer.
        if self.use_vae and actions is not None:
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = einops.repeat(self.cls_embed.weight, "1 d -> b 1 d", b=batch_size)  # (B, 1, D)
            robot_state_embed = self.vae_encoder_robot_state_input_proj(robot_state).unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(actions)  # (B, S, D)
            vae_encoder_input = torch.cat([cls_embed, robot_state_embed, action_embed], axis=1)  # (B, S+2, D)
            vae_encoder_input = vae_encoder_input.permute(1, 0, 2)  # (S+2, B, D)
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            # Prepare fixed positional embedding.
            pos_embed = self.vae_encoder_pos_enc.clone().detach().permute(1, 0, 2)  # (S+2, 1, D)
            # Forward pass through VAE encoder and sample the latent with the reparameterization trick.
            vae_encoder_output = self.vae_encoder(
                vae_encoder_input, pos=pos_embed
            )  # , src_key_padding_mask=is_pad)  # TODO(now)
            vae_encoder_output = vae_encoder_output[0]  # take cls output only
            latent_pdf_params = self.vae_encoder_latent_output_proj(vae_encoder_output)
            mu = latent_pdf_params[:, : self.latent_dim]
            logvar = latent_pdf_params[:, self.latent_dim :]
            # Use reparameterization trick to sample from the latent's PDF.
            latent_sample = mu + logvar.div(2).exp() * torch.randn_like(mu)
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = logvar = None
            latent_sample = torch.zeros([batch_size, self.latent_dim], dtype=robot_state.dtype).to(
                robot_state.device
            )

        # Prepare all other transformer inputs.
        # Image observation features and position embeddings.
        all_cam_features = []
        all_cam_pos = []
        for cam_id, _ in enumerate(self.camera_names):
            # TODO(now): remove the positional embedding from the backbones.
            features, pos = self.backbones[0](image[:, cam_id])  # HARDCODED
            features = features[0]  # take the last layer feature
            pos = pos[0]
            all_cam_features.append(self.encoder_img_feat_input_proj(features))
            all_cam_pos.append(pos)
        # Concatenate image observation feature maps along the width dimension.
        transformer_input = torch.cat(all_cam_features, axis=3)
        # TODO(now): remove the positional embedding from the backbones.
        pos = torch.cat(all_cam_pos, axis=3)
        robot_state_embed = self.encoder_robot_state_input_proj(robot_state)
        latent_embed = self.encoder_latent_input_proj(latent_sample)

        # Run the transformer and project the outputs to the action space.
        transformer_output = self.transformer(
            transformer_input,
            query_embed=self.decoder_pos_embed.weight,
            pos_embed=pos,
            latent_input=latent_embed,
            proprio_input=robot_state_embed,
            additional_pos_embed=self.additional_pos_embed.weight,
        )
        a_hat = self.action_head(transformer_output)

        return a_hat, [mu, logvar]


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
        use_vae=args.vae,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: {:.2f}M".format(n_parameters / 1e6))

    return model
