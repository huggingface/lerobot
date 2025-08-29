#!/usr/bin/env python
"""
Benchmark script to compare forward pass speed between RLearn and ReWiND implementations.

This script compares the inference speed of:
1. RLearn model (lerobot implementation)
2. ReWiND model (reference implementation)

Both models use the same backbone architectures (DINOv2 + sentence-transformers)
and implement similar reward modeling approaches.
"""

import time
from itertools import chain
from random import random

import einx
import torch
import torch.nn.functional as F
from einops import pack, rearrange, repeat, unpack
from hl_gauss_pytorch import HLGaussLayer
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.nn.utils.rnn import pad_sequence

# ReWiND implementation (copied from user's context)
from transformers import AutoImageProcessor, AutoModel
from vit_pytorch.accept_video_wrapper import AcceptVideoWrapper
from x_mlps_pytorch import Feedforwards
from x_transformers import Decoder

from lerobot.constants import OBS_IMAGES, OBS_LANGUAGE
from lerobot.policies.rlearn.configuration_rlearn import RLearNConfig

# RLearn implementation
from lerobot.policies.rlearn.modeling_rlearn import RLearNPolicy


# ReWiND helper functions
def exists(v):
    return v is not None


def satisfy_prob(prob):
    return random() < prob


def mask_from_lens(lens):
    seq = torch.arange(lens.amax().item(), device=lens.device)
    mask = einx.less("n, b -> b n", seq, lens)
    return mask


def randint(min_value: int, max_value: torch.Tensor):
    value_range = (max_value - min_value).float()
    return ((value_range * torch.rand_like(value_range)) + min_value).round().clamp(min=min_value).long()


# ReWiND DinoImageEmbedder
class DinoImageEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.image_model = AutoModel.from_pretrained("facebook/dinov2-base")

    def forward(self, images):
        model_inputs = self.image_processor(images, return_tensors="pt")
        outputs = self.image_model(**model_inputs)
        last_hidden_states = outputs[0]
        return last_hidden_states[:, 0]  # cls


# ReWiND RewardModel
class RewardModel(nn.Module):
    def __init__(
        self,
        decoder: dict | Decoder = dict(dim=768, depth=4, heads=8, attn_dim_head=64),
        image_model: nn.Module | None = None,
        mlp_predictor_depth=3,
        reward_bins=10,
        max_video_frames=16,
        dim_image_embed=768,
        num_register_tokens=4,
        lang_per_token_embed=True,
        sentence_transformer_path="sentence-transformers/all-MiniLM-L12-v2",
        categorical_rewards=False,
        use_hl_gauss_loss=True,
        reward_min_value=0.0,
        reward_max_value=1.0,
        reward_hl_gauss_loss_num_bins=20,
    ):
        super().__init__()

        self.lang_per_token_embed = lang_per_token_embed
        self.mini_lm = SentenceTransformer(sentence_transformer_path)
        mini_lm_dim = self.mini_lm.encode(["__"]).shape[-1]

        if not exists(image_model):
            image_model = DinoImageEmbedder()

        self.video_embed = AcceptVideoWrapper(image_model)
        self.decoder = Decoder(**decoder)
        dim = self.decoder.dim

        self.first_pos_emb = nn.Parameter(torch.randn(dim) * 1e-2)
        self.to_lang_tokens = nn.Linear(mini_lm_dim, dim)
        self.to_video_tokens = nn.Linear(dim_image_embed, dim)

        self.mlp_predictor = Feedforwards(
            dim=dim, dim_out=reward_bins if categorical_rewards else None, depth=mlp_predictor_depth
        )

        self.num_register_tokens = num_register_tokens
        self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim) * 1e-2)
        self.categorical_rewards = categorical_rewards

        self.hl_gauss_layer = HLGaussLayer(
            dim=dim,
            use_regression=not use_hl_gauss_loss,
            hl_gauss_loss=dict(
                min_value=reward_min_value,
                max_value=reward_max_value,
                num_bins=reward_hl_gauss_loss_num_bins,
            ),
        )

    def parameters(self):
        return chain(
            self.decoder.parameters(),
            iter((self.video_embed.pos_emb,)),
            self.to_lang_tokens.parameters(),
            self.to_video_tokens.parameters(),
            self.mlp_predictor.parameters(),
            self.hl_gauss_layer.parameters(),
        )

    def forward(
        self,
        commands: list[str],
        video,  # (b c t h w)
        extra_embed_tokens=None,  # (b n d)
        rewards=None,
        video_lens=None,
    ):
        batch = video.shape[0]
        assert len(commands) == batch

        device = video.device
        mask = None

        # register tokens
        register_tokens = repeat(self.register_tokens, "n d -> b n d", b=batch)

        # language embed
        lang_embeds = self.mini_lm.encode(
            commands,
            output_value="token_embeddings" if self.lang_per_token_embed else "sentence_embedding",
            convert_to_numpy=False,
        )
        lang_embeds = pad_sequence(lang_embeds, batch_first=True).to(device)

        if self.lang_per_token_embed:
            lens = torch.tensor([t.shape[0] for t in lang_embeds], device=device)
            mask = mask_from_lens(lens)

        # extra embeds
        if not exists(extra_embed_tokens):
            extra_embed_tokens = register_tokens[:, 0:0]

        elif exists(extra_embed_tokens) and exists(mask):
            mask = F.pad(mask, (0, extra_embed_tokens.shape[-2]), value=True)

        # video embeds
        video_embeds = self.video_embed(video, eval_with_no_grad=True)

        if self.lang_per_token_embed:
            mask = F.pad(mask, (0, video_embeds.shape[1] + self.num_register_tokens), value=True)

        # linear projections
        lang_tokens = self.to_lang_tokens(lang_embeds)
        video_tokens = self.to_video_tokens(video_embeds)

        # add video start positional embedding
        first_video_token, rest_video_tokens = video_tokens[:, :1], video_tokens[:, 1:]
        first_video_token = first_video_token + repeat(self.first_pos_emb, "d -> b 1 d", b=batch)
        video_tokens = torch.cat((first_video_token, rest_video_tokens), dim=1)

        # pack all tokens for attention
        tokens, lang_video_packed_shape = pack(
            (lang_tokens, register_tokens, extra_embed_tokens, video_tokens), "b * d"
        )

        # attention
        attended = self.decoder(tokens, mask=mask)

        # unpack and project the video tokens to logits to train reward predictor
        _, _, _, attended_video_tokens = unpack(attended, lang_video_packed_shape, "b * d")

        video_frame_embed_or_logits = self.mlp_predictor(attended_video_tokens)

        # determine video masking for loss
        video_mask = None
        if exists(video_lens):
            video_mask = mask_from_lens(video_lens)
            max_video_len = video_lens.amax().item()
            video_frame_embed_or_logits = video_frame_embed_or_logits[:, :max_video_len]
            if exists(rewards):
                rewards = rewards[:, :max_video_len]
                rewards = einx.where("b t, b t,", video_mask, rewards, -1)

        # return raw prediction or loss
        return_loss = exists(rewards)
        if not return_loss:
            if self.categorical_rewards:
                return video_frame_embed_or_logits
            else:
                return self.hl_gauss_layer(video_frame_embed_or_logits)

        # calculate loss
        if self.categorical_rewards:
            assert rewards.dtype in (torch.long, torch.int)
            loss = F.cross_entropy(
                rearrange(video_frame_embed_or_logits, "b t l -> b l t"), rewards, ignore_index=-1
            )
        else:
            assert rewards.dtype == torch.float
            loss = self.hl_gauss_layer(video_frame_embed_or_logits, rewards, mask=video_mask)

        return loss


def benchmark_models():
    """Benchmark forward pass speed of RLearn vs ReWiND models."""

    print("Setting up models and test data...")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test data
    batch_size = 2
    num_frames = 16
    height, width = 224, 224

    commands = [
        "pick up the blue ball and put it in the red tray",
        "pick up the red cube and put it in the green bin",
    ]

    # Create video tensor (B, C, T, H, W) for ReWiND
    video_rewind = torch.rand(batch_size, 3, num_frames, height, width, device=device)

    # Create video tensor (B, T, C, H, W) for RLearn
    video_rlearn = video_rewind.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)

    # Create batch dict for RLearn
    batch = {OBS_IMAGES: video_rlearn, OBS_LANGUAGE: commands}

    # Initialize RLearn model
    print("Initializing RLearn model...")
    rlearn_config = RLearNConfig()
    rlearn_model = RLearNPolicy(rlearn_config).to(device)
    rlearn_model.eval()

    # Initialize ReWiND model
    print("Initializing ReWiND model...")
    rewind_model = RewardModel().to(device)
    rewind_model.eval()

    # Warm up both models
    print("Warming up models...")
    with torch.no_grad():
        for _ in range(3):
            _ = rlearn_model.predict_rewards(batch)
            _ = rewind_model(commands, video_rewind)

    # Benchmark RLearn
    print("\nBenchmarking RLearn model...")
    rlearn_times = []
    with torch.no_grad():
        for i in range(100):
            start_time = time.perf_counter()
            rewards = rlearn_model.predict_rewards(batch)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.perf_counter()
            rlearn_times.append(end_time - start_time)

    # Benchmark ReWiND
    print("Benchmarking ReWiND model...")
    rewind_times = []
    with torch.no_grad():
        for i in range(100):
            start_time = time.perf_counter()
            rewards = rewind_model(commands, video_rewind)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.perf_counter()
            rewind_times.append(end_time - start_time)

    # Calculate statistics
    rlearn_avg = sum(rlearn_times) / len(rlearn_times) * 1000  # Convert to ms
    rlearn_std = torch.tensor(rlearn_times).std().item() * 1000
    rlearn_min = min(rlearn_times) * 1000
    rlearn_max = max(rlearn_times) * 1000

    rewind_avg = sum(rewind_times) / len(rewind_times) * 1000
    rewind_std = torch.tensor(rewind_times).std().item() * 1000
    rewind_min = min(rewind_times) * 1000
    rewind_max = max(rewind_times) * 1000

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS (100 runs, inference only)")
    print("=" * 60)
    print(f"RLearn avg: {rlearn_avg:.2f} ms")
    print(f"RLearn std: {rlearn_std:.2f} ms")
    print(f"RLearn min: {rlearn_min:.2f} ms")
    print(f"RLearn max: {rlearn_max:.2f} ms")
    print(f"ReWiND avg: {rewind_avg:.2f} ms")
    print(f"ReWiND std: {rewind_std:.2f} ms")
    print(f"ReWiND min: {rewind_min:.2f} ms")
    print(f"ReWiND max: {rewind_max:.2f} ms")

    speedup = rlearn_avg / rewind_avg if rewind_avg > 0 else float("inf")
    print(f"Speedup (RLearn/ReWiND): {speedup:.2f}x")
    print(f"{'RLearn is faster!' if speedup > 1 else 'ReWiND is faster!'}")
    # Verify outputs are similar in shape
    print("\nOutput shapes:")
    with torch.no_grad():
        rlearn_output = rlearn_model.predict_rewards(batch)
        rewind_output = rewind_model(commands, video_rewind)

    print(f"RLearn output shape: {rlearn_output.shape}")
    print(f"ReWiND output shape: {rewind_output.shape}")

    if rlearn_output.shape == rewind_output.shape:
        print("✓ Output shapes match")
    else:
        print("⚠ Output shapes differ - this may indicate implementation differences")

    print("\nBenchmark completed successfully!")


if __name__ == "__main__":
    benchmark_models()
