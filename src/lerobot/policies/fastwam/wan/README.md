# FastWAM `wan` package

This package holds FastWAM's model implementation. It mixes a small **vendored
subset of the official Wan2.2 source tree** with FastWAM's own code, kept flat in
a single directory.

## Vendored from Wan2.2

- Upstream repository: https://github.com/Wan-Video/Wan2.2
- Upstream commit: `42bf4cfaa384bc21833865abc2f9e6c0e67233dc`
- License: Apache-2.0, matching the license in `LICENSE.txt` from the upstream repository

Copied files:

- `model.py` (was `wan/modules/model.py`), trimmed: the flash-attention path
  (the vendored `attention.py` and the block/model `forward`s) was removed.
  FastWAM's DiT uses SDPA instead (see `video_dit.py`).
- `get_sampling_sigmas` in `video_dit.py` (was `wan/utils/fm_solvers.py`), inlined
  next to its only caller.

This subset only backs FastWAM's **custom MoT video DiT**. The Wan2.2 VAE,
UMT5 text encoder, and tokenizer are no longer vendored - they come from
`diffusers.AutoencoderKLWan`, `transformers.UMT5EncoderModel`, and
`transformers.AutoTokenizer` (see `components.py` and `adapters.py`).

## FastWAM's own code

- `video_dit.py` builds on `model` (`sinusoidal_embedding_1d`, `rope_params`,
  `rope_apply`, …) and computes attention with SDPA (`fastwam_masked_attention`). Its
  `WanContinuousFlowMatchScheduler` uses `get_sampling_sigmas` for Wan-compatible
  inference timesteps.
- `components.py` / `adapters.py` load the VAE, text encoder, tokenizer, and the
  custom DiT weights.
- `modular.py` defines the FastWAM model (`ActionDiT`, `MoT`, `FastWAM`, …).
