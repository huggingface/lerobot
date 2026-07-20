import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.sfpuel_utils.model import ImageEncoder, PolarEncoder
from .models.sdm_utils import model as sdm

# from lerobot.policies import PI05Config, PI05Policy
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
from lerobot_policy_pi05_polarization import PI05PolarizationConfig

from lerobot.policies.pi05 import (  # noqa: E402
    PI05Config,
    PI05Policy,
    make_pi05_pre_post_processors,  # noqa: E402
)

class PolFEM(nn.Module):

    def __init__(self, img_ch=4, stokes_ch=6, canonical_resolution=256):
        super().__init__()
        self.polfe = ImageEncoder(img_ch)
        self.polar_encoder = PolarEncoder(stokes_ch, self.polfe.out_channels[0])
        self.polfe_fusion = sdm.ImageFeatureFusion(self.polfe.out_channels)
        self.canonical_resolution = canonical_resolution

    def train(self, mode=True):
        return super().train(False)   # always frozen, ignore requests to switch to train mode

    @torch.no_grad()
    def forward(self, image_batch, stokes6, n_img_array):
        # image_batch: (B*4, 4, H, W); stokes6: (B, 6, H, W); n_img_array: (B,)
        res = self.canonical_resolution
        image_batch = F.interpolate(image_batch, size=(res, res), mode='bilinear', align_corners=True)
        stokes6 = F.interpolate(stokes6, size=(res, res), mode='bilinear', align_corners=True)

        control = self.polar_encoder(stokes6)          # (B, 96, res/4, res/4) roughly
        polfe_feats = self.polfe([image_batch, control])   # tuple of 4 stage tensors

        feat = self.polfe_fusion(polfe_feats, n_img_array)   # (B*N, 256, H/4, W/4)
        B = n_img_array.shape[0]
        N = n_img_array[0].item()
        feat = feat.reshape(B, N, *feat.shape[1:]).mean(dim=1)   # -> (B, 256, H/4, W/4)
        return feat


class PolarizationTokenizer(nn.Module):
    def __init__(self, in_channels=256, embed_dim=2048, patch=4, grid=16):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, grid * grid, embed_dim))
        self.modality_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.modality_embed, std=0.02)

    def forward(self, polfem_features):          # (B, 256, 64, 64)
        x = self.proj(polfem_features)            # (B, 2048, 16, 16)
        x = x.flatten(2).transpose(1, 2)           # (B, 256, 2048)
        x = self.norm(x)
        return x + self.pos_embed + self.modality_embed


def build_polfem_inputs(pol000, pol045, pol090, pol135, mask, eps=1e-8):
    """
    pol000..pol135: (B, 3, H, W) tensors, RGB, values in [0,1]
                     (i.e. after: img.float()/255.0, and after normalize_augment-style
                     per-scene max-normalization within the mask -- see note below)
    mask: (B, 1, H, W), values in {0,1}
    Returns:
        image_batch: (B*4, 4, H, W)   # 4 angle images, each with mask appended -> img_ch=4
        stokes6:     (B, 6, H, W)     # [S0, S1, S2, sin(2ψ), cos(2ψ), DoLP]
        n_img_array: (B,) int tensor, all values = 4
    """
    B = pol000.shape[0]

    #rescale to 256 if not already
    if pol000.shape[2] != 256:
        pol000 = F.interpolate(pol000, size=(256, 256), mode='bilinear', align_corners=True)
        pol045 = F.interpolate(pol045, size=(256, 256), mode='bilinear', align_corners=True)
        pol090 = F.interpolate(pol090, size=(256, 256), mode='bilinear', align_corners=True)
        pol135 = F.interpolate(pol135, size=(256, 256), mode='bilinear', align_corners=True)
        mask = F.interpolate(mask, size=(256, 256), mode='nearest')
        
        print("Rescaled polar images to 256x256")
        print("polar000.shape", pol000.shape)
    
    polar = torch.stack([pol000, pol045, pol090, pol135], dim=1)   # (B,4,3,H,W)

    s0 = polar.mean(dim=1)                       # (B,3,H,W)
    s1 = (polar[:, 0] - polar[:, 2]) / 2.        # (B,3,H,W)  -- I0 - I90
    s2 = (polar[:, 1] - polar[:, 3]) / 2.        # (B,3,H,W)  -- I45 - I135

    stokes = torch.stack([s0, s1, s2], dim=1).mean(dim=2)   # (B,3,H,W), color-averaged
    aolp = 0.5 * torch.atan2(stokes[:, 2:3], stokes[:, 1:2] + eps)
    aolp_embed = torch.cat([torch.sin(aolp * 2), torch.cos(aolp * 2)], dim=1)   # (B,2,H,W)
    dolp = torch.sqrt(stokes[:, 1:2] ** 2 + stokes[:, 2:3] ** 2) / (stokes[:, :1] + eps)
    stokes6 = torch.cat([stokes, aolp_embed, dolp], dim=1)   # (B,6,H,W)

    # 1. Force mask to be strictly 4D (B, 1, H, W) to clear interpolation artifacts
    mask = mask.view(B, 1, *polar.shape[-2:])

    # 2. Expand mask to match B*4 sequence
    mask_per_angle = mask.repeat(1, 4, 1, 1).reshape(-1, 1, *mask.shape[-2:])
    
    # 3. polar is (B, 4, 3, H, W). mask.unsqueeze(1) becomes (B, 1, 1, H, W) 
    # This safely broadcasts across both angle (4) and color channels (3)
    image_batch = (polar * mask.unsqueeze(1)).reshape(-1, 3, *polar.shape[-2:])
    image_batch = torch.cat([image_batch, mask_per_angle], dim=1)   # (B*4, 4, H, W)

    n_img_array = torch.full((B,), 4, dtype=torch.long)
    return image_batch, stokes6, n_img_array


def pa_to_pol_angles(pa_array, s0, eps=1e-8):
    """
    Exactly inverts the forward polarimetric representation back to polar angle intensities.
    Handles dynamic data types (float, uint8, int16, etc.) with proper normalization.
    
    Args:
        pa_array: (B, C, H, W) tensor containing [_, cos(2ψ), sin(2ψ), DoLP, ...]
        s0: (B, 3, H, W) tensor, the original per-channel total intensity S0 (values in [0,1])
    Returns:
        pol000, pol045, pol090, pol135: (B, 3, H, W) tensors with values in [0,1]
    """
    # 1. Cast to float for math operations
    pa_float = pa_array.float()
    
    # 2. Normalize based on dtype to bring values to the target range
    if not pa_array.is_floating_point():
        if pa_array.dtype == torch.uint8:
            # Map [0, 255] -> [-1.0, 1.0]
            pa_float = (pa_float / 127.5) - 1.0
        elif pa_array.dtype == torch.int8:
            # Map [-128, 127] -> [-1.0, 1.0]
            pa_float = (pa_float + 0.5) / 127.5
        elif pa_array.dtype == torch.int16:
            # Map [-32768, 32767] -> [-1.0, 1.0]
            pa_float = (pa_float + 0.5) / 32767.5
        elif pa_array.dtype == torch.int32:
            # Map [-2147483648, 2147483647] -> [-1.0, 1.0]
            pa_float = (pa_float + 0.5) / 2147483647.5

    # 3. Extract components from normalized float array
    two_cos_psi = pa_float[:, 0:1, :, :]
    two_sin_psi = pa_float[:, 1:2, :, :]
    dolp = pa_float[:, 2:3, :, :]
    
    # If DoLP was also mapped into [-1, 1] during quantization, shift it back to [0, 1]
    # Note: If your quantization pipeline keeps DoLP in [0, 1] or [0, 255], adjust accordingly.
    if not pa_array.is_floating_point() and pa_array.dtype == torch.uint8:
        # Assuming DoLP used the same [0, 255] -> [-1, 1] shift, we restore it to [0, 1]
        dolp = (dolp + 1.0) / 2.0
    
    # 4. Reconstruct Stokes components
    s1 = s0 * dolp * two_cos_psi
    s2 = s0 * dolp * two_sin_psi
    
    # 5. Invert the Stokes relations
    pol000 = s0 + s1
    pol045 = s0 + s2
    pol090 = s0 - s1
    pol135 = s0 - s2
    
    # 6. Clamp to ensure safety against floating-point precision leaks
    pol000 = torch.clamp(pol000, 0.0, 1.0)
    pol045 = torch.clamp(pol045, 0.0, 1.0)
    pol090 = torch.clamp(pol090, 0.0, 1.0)
    pol135 = torch.clamp(pol135, 0.0, 1.0)
    
    return pol000, pol045, pol090, pol135

    

    


class PI05PytorchWithPolarization(PI05Pytorch):
    def __init__(self, config, rtc_processor=None):
        super().__init__(config, rtc_processor)
        self.polfem = PolFEM(img_ch=4, stokes_ch=6, canonical_resolution=256)
        self.polar_tokenizer = PolarizationTokenizer(in_channels=256, embed_dim=2048)  # match PaliGemma width
        self._polar_inputs = None   # side-channel set just before each call

    def embed_prefix(self, images, img_masks, tokens, masks):
        embs, pad_masks, att_masks = super().embed_prefix(images, img_masks, tokens, masks)
        if self._polar_inputs is not None:
            image_batch, stokes6, n_img_array = self._polar_inputs
            feat = self.polfem(image_batch, stokes6, n_img_array)   # frozen, no_grad internally
            polar_tokens = self.polar_tokenizer(feat)                 # (B, 256, hidden)
            bsize, n_polar = polar_tokens.shape[:2]
            embs = torch.cat([embs, polar_tokens], dim=1)
            pad_masks = torch.cat([pad_masks, torch.ones(bsize, n_polar, dtype=pad_masks.dtype, device=pad_masks.device)], dim=1)
            att_masks = torch.cat([att_masks, torch.zeros(bsize, n_polar, dtype=att_masks.dtype, device=att_masks.device)], dim=1)
        return embs, pad_masks, att_masks


class PI05WithPolarization(PI05Policy):
    config_class = PI05PolarizationConfig
    name = "pi05_polarization"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = PI05PytorchWithPolarization(config, rtc_processor=self.rtc_processor)

    def predict_action_chunk(self, batch, **kwargs):
        if "observation.polar000" not in batch:
            if "observation.images.virtual_polar" in batch and batch["observation.images.base_0_rgb"].shape[1] == 3:
                pol000, pol045, pol090, pol135 = pa_to_pol_angles(batch["observation.images.virtual_polar"], batch["observation.images.base_0_rgb"])
                batch["observation.polar000"] = pol000
                batch["observation.polar045"] = pol045
                batch["observation.polar090"] = pol090
                batch["observation.polar135"] = pol135


        image_batch, stokes6, n_img_array = build_polfem_inputs(
            batch["observation.polar000"], batch["observation.polar045"],
            batch["observation.polar090"], batch["observation.polar135"],
            batch.get("observation.polar_mask", torch.ones_like(batch["observation.polar000"][:, :1])),
        )
        self.model._polar_inputs = (image_batch, stokes6, n_img_array)
        return super().predict_action_chunk(batch, **kwargs)

    def get_optim_params(self) -> dict:
        # only ever hand the optimizer parameters that are actually trainable —
        # matters once you start toggling which submodules are frozen per training stage
        return {"params": [p for p in self.parameters() if p.requires_grad]}


if __name__ == "__main__":

    ckpt = torch.load("data/checkpoints/ckpt.pth", map_location="cpu")
    full_state_dict = ckpt["state_dict"]     # confirmed key name, per model_utils.load_checkpoint

    polfem = PolFEM(img_ch=4, stokes_ch=6, canonical_resolution=256)

    wanted_prefixes = ("feature_extractor.polfe.", "feature_extractor.polar_encoder.", "feature_extractor.polfe_fusion.")
    filtered = {k: v for k, v in full_state_dict.items() if k.startswith(wanted_prefixes)}

    #remove feature_extractor prefix
    filtered = {k.replace("feature_extractor.", ""):v for k, v in filtered.items()}
    

    # for k, v in filtered.items():
    #     print(k)

    missing, unexpected = polfem.load_state_dict(filtered, strict=False)
    print("missing:", missing)       # expect empty
    print("unexpected:", unexpected) # expect empty -- if not, print full_state_dict.keys() and re-check prefixes