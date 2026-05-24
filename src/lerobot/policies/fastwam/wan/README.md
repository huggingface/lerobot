# Wan2.2 Upstream Subset

This directory contains an unmodified subset of the official Wan2.2 source tree.

- Upstream repository: https://github.com/Wan-Video/Wan2.2
- Upstream commit: `42bf4cfaa384bc21833865abc2f9e6c0e67233dc`
- License: Apache-2.0, matching the license in `LICENSE.txt` from the upstream repository

Copied files:

- `wan/modules/attention.py`
- `wan/modules/model.py`
- `wan/modules/t5.py`
- `wan/modules/tokenizers.py`
- `wan/modules/vae2_1.py`
- `wan/modules/vae2_2.py`
- `wan/modules/__init__.py`
- `wan/utils/fm_solvers.py`
- `wan/utils/fm_solvers_unipc.py`
- `wan/utils/__init__.py`

FastWAM-specific model glue and any code adapted from these modules live outside this directory. This keeps the upstream Wan2.2 code reviewable as a vendored reference subset and makes it straightforward to replace this directory with an external Wan2.2 dependency by changing import paths.

Current FastWAM adapters that directly reuse this vendored subset:

- `../wan_components.py` instantiates the upstream `wan.modules.t5.umt5_xxl` encoder factory and uses `wan.modules.tokenizers.HuggingfaceTokenizer`.
- `../wan_adapters.py` wraps `wan.modules.vae2_2.Wan2_2_VAE` with the FastWAM tensor-batch encode/decode API.
- `../modular_fastwam.py` reuses `wan.utils.fm_solvers.get_sampling_sigmas` for Wan-compatible inference timesteps.
