import torch
import torch.nn as nn
import warnings
from typing import Dict, Optional, Any, Tuple
from .vq import VQ, NSVQ, EMAVQ, AEQuantizer
from .vjepa_encoder import build_vision_encoder, CosmosAutoencoder, VJEPAEncoder

import yaml
from lerobot.policies.lawam.lam_core.data_loader.video_aug import LAM_IMAGE_HW, LAM_PATCH_SIZE
from .utils.lam_encoder import LAMEncoder
from .utils.lam_decoder import LAMDecoder_v2, StatePredictor
from .utils.modules import PatchEmbed





class LatentLAMModel(nn.Module):
    """LAM model with a shared vision encoder, latent action bottleneck, and state decoder."""

    def __init__(
        self,
        dim: int=1024,
        num_heads: int = 16,
        ffn_expansion_factor: int = 2,
        enc_layers: int = 6,
        codebook_size: int = 16,
        code_dim: int = 256,
        max_state_dim: int = 32,
        num_frames: int = 5,
        num_queries: int = 1,
        vq_kwargs: Optional[Dict[str, Any]] = None,
        dec_layers: int = 6,
        dropout: float = 0.1,
        vq_type: str = "nsvq",
        disable_vq: bool = False,
        norm_latents: bool = False,
        norm_latents_type: str = "l2",
        vision_model_id: str = "facebook/vjepa2-vitl-fpc64-256",
        enc_add_state: bool = False,
        enc_modal_mask: bool = False,
        latent_layer_to_use: Any = 23,
        multi_input: bool = False,
        num_embodiments: int = 32,
        image_hw: Tuple[int, int] = LAM_IMAGE_HW,
        patch_size: int = LAM_PATCH_SIZE,
        decoder_last_ln: bool = True,
        **kwargs
    ):
        super().__init__()
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.patch_size = int(patch_size)
        if self.patch_size != LAM_PATCH_SIZE:
            raise ValueError(
                f"Unsupported patch_size={self.patch_size}. Only {LAM_PATCH_SIZE} is supported."
            )
        if self.image_hw[0] % self.patch_size != 0 or self.image_hw[1] % self.patch_size != 0:
            raise ValueError(
                f"image_hw={self.image_hw} must be divisible by patch_size={self.patch_size}."
            )
        self.grid_height = self.image_hw[0] // self.patch_size
        self.grid_width = self.image_hw[1] // self.patch_size
        # Allocate one normalization layer per requested latent layer.
        if isinstance(latent_layer_to_use, (list, tuple)):
            num_latent_layers = len(latent_layer_to_use)
        else:
            num_latent_layers = 1
        encoder_obj, input_dim = build_vision_encoder(
            vision_model_id,
            num_latent_layers=num_latent_layers,
            norm_layer_type=norm_latents_type,
            enable_norm=norm_latents,
        )
        self.feature_dim = dim
        self.code_dim = code_dim
        if encoder_obj is None:
            # Fallback path for training without a frozen pretrained visual encoder.
            self.vision_encoder = PatchEmbed(patch_size=self.patch_size, embed_dim=dim, in_chans=3)
            self.input_dim = self.vision_encoder.feature_dim
            self.train_in_latent = False
        else:
            self.vision_encoder = encoder_obj
            self.input_dim = input_dim
            self.train_in_latent = True
            if isinstance(self.vision_encoder, CosmosAutoencoder):
                if isinstance(latent_layer_to_use, (list, tuple)):
                    raise ValueError(
                        "CosmosAutoencoder branch supports only a single latent_layer_to_use value."
                    )
                if self.patch_size != int(getattr(self.vision_encoder, "patch_size", self.patch_size)):
                    raise ValueError(
                        "CosmosAutoencoder branch requires LAM patch_size to match the tokenizer spatial compression ratio."
                    )
            encoder_image_size = getattr(self.vision_encoder, "image_size", None)
            if encoder_image_size is not None:
                expected_hw = (int(encoder_image_size), int(encoder_image_size))
                if self.image_hw != expected_hw:
                    warnings.warn(
                        f"LAM image_hw={self.image_hw} does not match encoder-native size {expected_hw} "
                        f"for `{vision_model_id}`. Make sure this is intentional and that the token grid stays aligned.",
                        stacklevel=2,
                    )
        self.num_frames = num_frames
        self.num_queries = num_queries
        self.num_embodiments = int(num_embodiments)
        self.feature_decoder = None
        self.state_decoder = None
        self.frame_to_pre = 1
        self.decoder = LAMDecoder_v2(
            context_dim=dim,
            input_dim=self.input_dim,
            num_queries=num_queries,
            num_layers=dec_layers,
            num_heads=num_heads,
            dropout=dropout,
            train_in_latent=self.train_in_latent,
            ffn_expansion_factor=ffn_expansion_factor,
            num_embodiments=self.num_embodiments,
            code_dim=code_dim,
            grid_hw=(self.grid_height, self.grid_width),
            last_ln=decoder_last_ln,
        )
        self.state_decoder = StatePredictor(
            latent_dim=dim,
            dropout=dropout,
            num_embodiments=self.num_embodiments,
            num_queries=num_queries,
            max_state_dim=max_state_dim,
            code_dim=code_dim,
        )
        self.norm_latents = norm_latents
        # print("norm_latents:", self.norm_latents)
        self.norm_latents_type = norm_latents_type
        self.vq_type = vq_type
        self.encoder = LAMEncoder(
            context_dim=dim,
            input_dim=2 * self.input_dim if multi_input else self.input_dim,
            add_state=enc_add_state,
            modal_mask=enc_modal_mask,
            num_layers=enc_layers,
            num_heads=num_heads,
            dropout=dropout,
            ffn_expansion_factor=ffn_expansion_factor,
            num_frames=self.num_frames,
            grid_hw=(self.grid_height, self.grid_width),
            num_queries=num_queries,
            max_state_dim=max_state_dim,
            num_embodiments=self.num_embodiments,
            code_dim=code_dim,
        )
        self.latent_layer_to_use = latent_layer_to_use
        self.multi_input = multi_input
        vq_kwargs = vq_kwargs or {}

        if self.vq_type == "nsvq":
            self.vq = NSVQ(
                codebook_size=codebook_size,
                code_dim=code_dim,
                use_diveq=False,
                **vq_kwargs
            )
        elif self.vq_type in ("ema", "ema_vq"):
            self.vq = EMAVQ(
                codebook_size=codebook_size,
                code_dim=code_dim,
                **vq_kwargs
            )
        elif self.vq_type == "vq":
            self.vq = VQ(
                codebook_size=codebook_size,
                code_dim=code_dim,
                **vq_kwargs
            )
        elif self.vq_type in ("vae", "beta_vae"):
            from .vq import VAEQuantizer
            self.vq = VAEQuantizer(
                code_dim=code_dim,
                **vq_kwargs,
            )
        elif self.vq_type == "ae":
            self.vq = AEQuantizer(
                code_dim=code_dim,
                codebook_size=codebook_size,
                **vq_kwargs,
            )
        else:
            print(f"Unsupported vq_type='{vq_type}', falling back to NSVQ.")
            self.vq = NSVQ(
                codebook_size=codebook_size,
                code_dim=code_dim,
                **vq_kwargs
            )
        self.disable_vq = disable_vq
        # self.state_delta_predictor = StatePredictor(
        #         latent_dim=code_dim,
        #         dropout=dropout
        #     )
        self.codebook_size = codebook_size

    def _uses_vjepa_split_features(self) -> bool:
        return self.train_in_latent and isinstance(self.vision_encoder, VJEPAEncoder)
    def _resolve_states(
        self,
        *,
        videos: torch.Tensor,
        states: Optional[torch.Tensor],
        state_mask: Optional[torch.Tensor],
        require_states: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        expected_dim = int(getattr(getattr(self, "encoder", None), "max_state_dim", 0) or 0)
        if expected_dim <= 0:
            raise ValueError("[LatentLAMModel] encoder.max_state_dim must be > 0.")

        if states is None:
            if require_states:
                raise ValueError("[LatentLAMModel] `states` is required for training/inference.")
            if videos.ndim < 2:
                raise ValueError(f"[LatentLAMModel] invalid videos shape: {tuple(videos.shape)}")
            fallback_t = int(videos.shape[1])
            if fallback_t <= 0:
                raise ValueError("[LatentLAMModel] videos must have positive temporal length.")
            state_dtype = videos.dtype if torch.is_floating_point(videos) else torch.float32
            states = torch.zeros(
                videos.shape[0],
                fallback_t,
                expected_dim,
                dtype=state_dtype,
                device=videos.device,
            )
        else:
            if not isinstance(states, torch.Tensor):
                states = torch.as_tensor(states)
            states = states.to(device=videos.device)
            if states.ndim == 4 and states.size(-2) == 1:
                states = states.squeeze(-2)
            if states.ndim != 3:
                raise ValueError(
                    f"[LatentLAMModel] `states` must be [B,T,D] or [B,T,1,D], got {tuple(states.shape)}"
                )
            if states.shape[0] != videos.shape[0]:
                raise ValueError(
                    f"[LatentLAMModel] batch mismatch: videos B={videos.shape[0]} vs states B={states.shape[0]}"
                )
            if require_states and states.shape[1] != 2:
                raise ValueError(
                    f"[LatentLAMModel] training/inference expects endpoint states with T=2, got T={states.shape[1]}"
                )
            if states.shape[-1] != expected_dim:
                raise ValueError(
                    f"[LatentLAMModel] states dim mismatch: got D={states.shape[-1]}, expected {expected_dim}"
                )

        if state_mask is not None:
            if not isinstance(state_mask, torch.Tensor):
                state_mask = torch.as_tensor(state_mask)
            state_mask = state_mask.to(device=videos.device)
            if state_mask.ndim == 2:
                if state_mask.shape != (states.shape[0], states.shape[-1]):
                    raise ValueError(
                        f"[LatentLAMModel] 2D state_mask must be [B,D], got {tuple(state_mask.shape)}"
                    )
                state_mask = state_mask[:, None, :].expand(-1, states.shape[1], -1)
            elif state_mask.ndim == 3:
                if state_mask.shape != states.shape:
                    raise ValueError(
                        f"[LatentLAMModel] state_mask shape mismatch: mask={tuple(state_mask.shape)} vs states={tuple(states.shape)}"
                    )
            else:
                raise ValueError(
                    f"[LatentLAMModel] state_mask must be [B,D] or [B,T,D], got {tuple(state_mask.shape)}"
                )
            state_mask = state_mask.to(torch.bool)

        return states, state_mask

    def _resolve_embodiment_ids(
        self,
        embodiment_ids: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
        require_embodiment_ids: bool,
    ) -> torch.Tensor:
        if embodiment_ids is None:
            if require_embodiment_ids:
                raise KeyError("[LatentLAMModel] `embodiment_ids` is required for training.")
            return torch.zeros(batch_size, device=device, dtype=torch.long)

        if not isinstance(embodiment_ids, torch.Tensor):
            raise TypeError(
                f"[LatentLAMModel] `embodiment_ids` must be torch.Tensor or None, got {type(embodiment_ids).__name__}"
            )
        emb = embodiment_ids
        if emb.ndim == 2 and emb.size(1) == 1:
            emb = emb.squeeze(1)
        elif emb.ndim != 1:
            raise ValueError(
                f"[LatentLAMModel] `embodiment_ids` must be [B] or [B,1], got {tuple(emb.shape)}"
            )
        if emb.shape[0] != batch_size:
            raise ValueError(
                f"[LatentLAMModel] embodiment_ids batch mismatch: got {emb.shape[0]}, expected {batch_size}"
            )
        return emb.to(device=device, dtype=torch.long)

    def forward(
        self,
        videos: torch.Tensor,
        states: Optional[torch.Tensor],
        dec_videos: torch.Tensor,
        state_mask: Optional[torch.Tensor] = None,
        embodiment_ids: Optional[torch.Tensor] = None,
    ):
        return self._run(
            videos=videos,
            states=states,
            dec_videos=dec_videos,
            state_mask=state_mask,
            embodiment_ids=embodiment_ids,
            vq_training=True,
            require_states=True,
            require_embodiment_ids=True,
        )


    def _run(
        self,
        videos: torch.Tensor,   #[B, T, C,H,W]
        states: Optional[torch.Tensor],  #[B,T,D]
        dec_videos: torch.Tensor,  #[B,T,C,H,W]
        state_mask: Optional[torch.Tensor] = None,
        embodiment_ids: Optional[torch.Tensor] = None,
        user_specific: Optional[int] = None,
        vq_training: bool = True,
        require_states: bool = False,
        require_embodiment_ids: bool = False,
        predict_future_frame: bool = True,
    ):
        states, state_mask = self._resolve_states(
            videos=videos,
            states=states,
            state_mask=state_mask,
            require_states=require_states,
        )
        emb_tensor = self._resolve_embodiment_ids(
            embodiment_ids=embodiment_ids,
            batch_size=int(states.shape[0]),
            device=videos.device,
            require_embodiment_ids=require_embodiment_ids,
        )
        state_mask_for_model = state_mask
        if state_mask is not None:
            states_for_model = states * state_mask.to(dtype=states.dtype)
        else:
            states_for_model = states

        T = videos.shape[1]

        if self.train_in_latent:
            if self._uses_vjepa_split_features():
                if T != 2:
                    raise ValueError(
                        f"V-JEPA LAM expects endpoint clips with T=2, got T={T}."
                    )
                if self.encoder.add_state:
                    raise NotImplementedError(
                        "V-JEPA split-feature path does not support enc_add_state=True. "
                        "Current configs should keep enc_add_state=false."
                    )
                enc_in = self.vision_encoder.encode_video(videos)  # [B, 1, K, D]
                dec_feats = self.vision_encoder.encode(dec_videos)  # [B, 2, K, D]
                dec_in = dec_feats[:, :1]
                tgt = dec_feats[:, -1:]
                dec_states = states_for_model[:, :1]
            else:
                cat_videos = torch.cat([videos, dec_videos], dim=1)
                all_features = self.vision_encoder.encode(
                    cat_videos,
                    n=self.latent_layer_to_use,
                )

                if isinstance(self.latent_layer_to_use, (list, tuple)) and isinstance(
                    all_features, (list, tuple)
                ):
                    dec_feats = all_features[-1]
                    if self.multi_input and len(self.latent_layer_to_use) >= 2:
                        enc_feats = torch.cat([all_features[0], all_features[-1]], dim=-1)
                    else:
                        enc_feats = all_features[0]
                else:
                    enc_feats = dec_feats = all_features

                enc_in = enc_feats[:, :T]  # [B, T, K, D]
                dec_in = dec_feats[:, T : T + 1]  # [B, 1, K, D]
                tgt = dec_feats[:, -1:]  # [B, 1, K, D]
                dec_states = states_for_model[:, :1]
            # print(f"dec_in norm mean:{dec_in.norm(dim=-1).mean()}", "\n")
            # print(f"dec_in norm std:{dec_in.norm(dim=-1).std()}", "\n")
            if vq_training:
                vision_features = None
            else:
                vision_features = torch.stack([dec_in, tgt], dim=1)
        else:
            cat_videos = torch.cat([videos, dec_videos], dim=1)
            patches = self.vision_encoder.encode(cat_videos)
            enc_in = patches[:, :T]  # [B, T, K, D]
            dec_in = patches[:, T : T + 1]  # [B, 1, K, D]
            tgt = patches[:, -1:]  # [B, 1, K, D]
            dec_states = states_for_model[:, :1]
            vision_features = None if vq_training else torch.stack([dec_in, tgt], dim=1)

        nodes = self.encoder(enc_in, states_for_model, embodiment_id=emb_tensor)  # [B, num_queries, code_dim]
        # nodes = self.encoder(video_feature)
        zero_tensor = torch.tensor(0.0, device=nodes.device)
        autocast_device = "cuda" if nodes.is_cuda else "cpu"
        with torch.amp.autocast(device_type=autocast_device, enabled=False):
            if self.vq is not None:
                if vq_training:
                    out = self.vq(nodes.float())
                    quantized, perplexity, indices, entropy_loss, vq_loss = out[:5]
                else:
                    out = self.vq.inference(nodes.float(), user_specific=user_specific)
                    quantized, indices = out[:2]
                    perplexity = zero_tensor
                    entropy_loss = zero_tensor
                    vq_loss = zero_tensor
            else:
                # Fallback: should not happen, but keep behavior consistent
                quantized = nodes.float()
                indices = torch.zeros(
                    (nodes.shape[0], nodes.shape[1]), device=nodes.device, dtype=torch.long
                )
                perplexity = zero_tensor
                entropy_loss = zero_tensor
                vq_loss = zero_tensor

        # Ensure indices exist when quantizer returns None (e.g., VAE/AE paths)
        if indices is None:
            indices = torch.zeros((nodes.shape[0], nodes.shape[1]), device=nodes.device, dtype=torch.long)
        if perplexity is None:
            perplexity = zero_tensor
        if entropy_loss is None:
            entropy_loss = zero_tensor
        if vq_loss is None:
            vq_loss = zero_tensor
        if self.disable_vq:
            # quantized = torch.zeros_like(nodes)
            quantized = nodes + 0.0 * quantized
        recon = None
        s_pred = None
        # delta_s_pred = self.state_delta_predictor(quantized, state_0=states[:,0])
        if predict_future_frame:
            recon = self.decoder(features=dec_in, actions=quantized)
            if self.state_decoder is not None:
                s_pred = self.state_decoder(z_t=quantized, state_0=dec_states, embodiment_id=emb_tensor)
                if state_mask_for_model is not None:
                    if s_pred.ndim == 2:
                        pred_mask = state_mask_for_model[:, 0, :].to(dtype=s_pred.dtype)
                    elif s_pred.ndim == 3 and s_pred.shape[1] == 1:
                        pred_mask = state_mask_for_model[:, :1, :].to(dtype=s_pred.dtype)
                    else:
                        pred_mask = None
                    if pred_mask is not None:
                        s_pred = s_pred * pred_mask
        # with torch.no_grad():
        #     print(tgt.mean(), tgt.std())
        #     delta = tgt-dec_in
        #     print(delta.mean(), delta.std())
        return (
            recon,
            dec_in,
            tgt,
            perplexity,
            indices,
            s_pred,
            vision_features,
            quantized,
            entropy_loss,
            vq_loss,
        )

    @torch.inference_mode()
    def inference(
        self,
        videos: torch.Tensor,
        states: Optional[torch.Tensor],
        dec_videos: torch.Tensor,
        state_mask: Optional[torch.Tensor] = None,
        embodiment_ids: Optional[torch.Tensor] = None,
    ):
        return self._run(
            videos=videos,
            states=states,
            dec_videos=dec_videos,
            state_mask=state_mask,
            embodiment_ids=embodiment_ids,
            vq_training=False,
            require_states=True,
            require_embodiment_ids=False,
            predict_future_frame=True,
        )

    @torch.inference_mode()
    def get_latent_action(
        self,
        videos: torch.Tensor,
        states: Optional[torch.Tensor],
        dec_videos: Optional[torch.Tensor] = None,
        state_mask: Optional[torch.Tensor] = None,
        predict_future_frame: bool = False,
        user_specific=None,
        embodiment_ids: Optional[torch.Tensor] = None,
    ):
        if dec_videos is None:
            dec_videos = videos
        (
            recon,
            dec_in,
            tgt,
            perplexity,
            indices,
            s_pred,
            features,
            quantized,
            entropy_loss,
            vq_loss,
        ) = self._run(
            videos=videos,
            states=states,
            dec_videos=dec_videos,
            state_mask=state_mask,
            user_specific=user_specific,
            vq_training=False,
            require_states=False,
            predict_future_frame=predict_future_frame,
            embodiment_ids=embodiment_ids,
            require_embodiment_ids=False,
        )
        vq_distances = None
        vq_logits = None
        vq_probs = None
        latent_mu = None
        latent_logvar = None
        return {
            "recon": recon,
            "dec_in": dec_in,
            "tgt": tgt,
            "perplexity": perplexity,
            "indices": indices,
            "s_pred": s_pred,
            "features": features,
            "quantized": quantized,
            "vq_distances": vq_distances,
            "vq_logits": vq_logits,
            "vq_probs": vq_probs,
            # Optional VAE stats (may be None)
            "latent_mu": latent_mu,
            "latent_logvar": latent_logvar,
        }

    @torch.no_grad()
    def extract_vision_features(self, videos: torch.Tensor, *, n: Optional[Any] = -2) -> torch.Tensor:
        n_used = n if n is not None else self.latent_layer_to_use
        try:
            feats = self.vision_encoder.encode(videos, n=n_used)
        except TypeError:
            feats = self.vision_encoder.encode(videos)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        return feats

def load_latent_action_model(ckpt_path, yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg.get('model', cfg) or {}

    if "image_hw" not in model_cfg:
        raise ValueError("LAM config must provide `model.image_hw`.")
    if "patch_size" not in model_cfg:
        raise ValueError("LAM config must provide `model.patch_size`.")

    init_kwargs = dict(model_cfg)
    init_kwargs.pop("ar_prediction", None)  # Backward compatibility with older configs.

    # Build on CPU first so checkpoint loading works across GPU and non-GPU hosts.
    latent_action_model = LatentLAMModel(**init_kwargs).to("cpu")

    lam_ckpt = torch.load(ckpt_path, map_location="cpu")['state_dict']
    new_ckpt = {}
    model_state = latent_action_model.state_dict()
    for key in lam_ckpt.keys():
        # Remove the Lightning module prefix saved during LAM training.
        renamed = key.replace("lam.", "")
        if renamed.startswith("vision_encoder.model.layer."):
            candidate = renamed.replace("vision_encoder.model.layer.", "vision_encoder.model.model.layer.", 1)
            if candidate in model_state:
                renamed = candidate
        new_ckpt[renamed] = lam_ckpt[key]
    model_keys = set(model_state.keys())
    ckpt_keys = set(new_ckpt.keys())

    missing_keys = sorted(list(model_keys - ckpt_keys))
    unexpected_keys = sorted(list(ckpt_keys - model_keys))
    shape_mismatches = []
    for k in sorted(model_keys & ckpt_keys):
        if model_state[k].shape != new_ckpt[k].shape:
            shape_mismatches.append((k, tuple(model_state[k].shape), tuple(new_ckpt[k].shape)))

    if missing_keys or unexpected_keys or shape_mismatches:
        error_lines = ["Failed to load LAM weights:"]
        if missing_keys:
            error_lines.append(f"Missing keys required by the model but absent from the checkpoint ({len(missing_keys)}):")
            error_lines += [f"  - {k}" for k in missing_keys]
        if unexpected_keys:
            error_lines.append(f"Unexpected keys present in the checkpoint but unused by the model ({len(unexpected_keys)}):")
            error_lines += [f"  - {k}" for k in unexpected_keys]
        if shape_mismatches:
            error_lines.append(f"Keys with shape mismatches ({len(shape_mismatches)}):")
            error_lines += [f"  - {k}: model{ms} vs checkpoint{cs}" for k, ms, cs in shape_mismatches]
        raise RuntimeError("\n".join(error_lines))

    latent_action_model.load_state_dict(new_ckpt, strict=True)
    for p in latent_action_model.parameters():
        p.requires_grad = False
    return latent_action_model.eval()
