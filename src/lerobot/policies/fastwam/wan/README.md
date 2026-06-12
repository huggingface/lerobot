# Wan2.2 Upstream Subset

This directory contains the trimmed subset of the official Wan2.2 source tree used by FastWAM.

- Upstream repository: https://github.com/Wan-Video/Wan2.2
- Upstream commit: `42bf4cfaa384bc21833865abc2f9e6c0e67233dc`
- License: Apache-2.0, matching the license in `LICENSE.txt` from the upstream repository

Copied files:

- `wan/modules/attention.py`
- `wan/modules/model.py`
- `wan/modules/__init__.py`
- `wan/utils/fm_solvers.py`
- `wan/utils/__init__.py`

This subset now only backs FastWAM's **custom MoT video DiT**. The Wan2.2 VAE,
UMT5 text encoder, and tokenizer are no longer vendored — they come from
`diffusers.AutoencoderKLWan`, `transformers.UMT5EncoderModel`, and
`transformers.AutoTokenizer` (see `../wan_adapters.py` and `../wan_components.py`).

Current FastWAM adapters that directly reuse this vendored subset:

- `../wan_video_dit.py` builds on `wan.modules.model` (`sinusoidal_embedding_1d`, `rope_params`, `rope_apply`, …) and `wan.modules.attention.flash_attention`.
- `../modular_fastwam.py` reuses `wan.utils.fm_solvers.get_sampling_sigmas` for Wan-compatible inference timesteps.
