"""Validate the decomposed-ONNX pipeline on a SHRUNK pi0 (random weights).

Runs with no download and negligible disk by monkeypatching the gemma configs to
tiny dimensions and replacing the SigLIP vision tower with a cheap deterministic
stand-in. It checks two equivalences against the real `sample_actions` reference:

  1. PyTorch decomposition (PrefixEncoder + host Euler loop + DenoiseStep) == reference
  2. ONNX decomposition (exported graphs + onnxruntime host loop)       == reference

If this passes, the identical code in reference.py / export_*.py / run_onnx.py will
produce a faithful ONNX conversion of the real checkpoint once disk is freed.
"""

import tempfile
from pathlib import Path

import numpy as np
import torch

import lerobot.policies.pi0.modeling_pi0 as m0

# ---- shrink the architecture BEFORE building the policy -------------------------
_TINY = m0.GemmaConfig(width=256, depth=2, mlp_dim=512, num_heads=8, num_kv_heads=1, head_dim=32)
m0.get_gemma_config = lambda variant: _TINY


def _tiny_embed_image(self, image):
    # Deterministic stand-in for SigLIP: 4 image tokens derived from the pixels.
    b = image.shape[0]
    w = self.paligemma.config.text_config.hidden_size
    feat = image.float().mean(dim=(2, 3))[:, :1]  # [B,1]
    return feat[:, :, None].expand(b, 4, w).contiguous()


m0.PaliGemmaWithExpertModel.embed_image = _tiny_embed_image
# --------------------------------------------------------------------------------

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.pi0.configuration_pi0 import PI0Config  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from pi0_onnx.common import (  # noqa: E402
    DenoiseStep,
    PrefixEncoder,
    euler_loop_numpy,
    export_to_onnx,
    make_ort_denoise_callable,
    patch_sincos_float32,
    prefix_arg_names,
)


def build_tiny_policy():
    cams = ["observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb"]
    input_features = {c: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)) for c in cams}
    input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(32,))
    cfg = PI0Config(
        dtype="float32",
        device="cpu",
        chunk_size=4,
        n_action_steps=4,
        num_inference_steps=3,
        input_features=input_features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
    )
    policy = m0.PI0Policy(cfg)
    policy.eval().to("cpu", dtype=torch.float32)
    return policy, cams


def main():
    torch.manual_seed(0)
    patch_sincos_float32()
    policy, cams = build_tiny_policy()
    cfg = policy.config
    g = torch.Generator().manual_seed(1234)
    B = 1

    images = [torch.rand(B, 3, 224, 224, generator=g) * 2 - 1 for _ in cams]
    img_masks = [torch.ones(B, dtype=torch.bool) for _ in cams]
    lang_tokens = torch.arange(cfg.tokenizer_max_length).remainder(50).add(10)[None].to(torch.long)
    lang_masks = torch.ones(B, cfg.tokenizer_max_length, dtype=torch.bool)
    state = torch.randn(B, cfg.max_state_dim, generator=g)
    noise = torch.randn(B, cfg.chunk_size, cfg.max_action_dim, generator=g)

    # ---- reference: the real sample_actions with fixed noise --------------------
    with torch.no_grad():
        ref = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone()
        )
    ref = ref[:, :, : cfg.output_features[ACTION].shape[0]].numpy()

    # ---- PyTorch decomposition --------------------------------------------------
    prefix = PrefixEncoder(policy)
    denoise = DenoiseStep(policy)
    with torch.no_grad():
        keys, values, ppm = prefix(*images, *img_masks, lang_tokens, lang_masks)

    def torch_denoise(state_np, x_t_np, t_np, k_np, v_np, ppm_np):
        with torch.no_grad():
            v = denoise(
                torch.from_numpy(state_np),
                torch.from_numpy(x_t_np),
                torch.from_numpy(t_np),
                torch.from_numpy(k_np),
                torch.from_numpy(v_np),
                torch.from_numpy(ppm_np),
            )
        return v.numpy()

    prefix_out = (keys.numpy(), values.numpy(), ppm.numpy(), state.numpy())
    dec = euler_loop_numpy(prefix_out, torch_denoise, noise.numpy(), cfg.num_inference_steps)
    dec = dec[:, :, : cfg.output_features[ACTION].shape[0]]
    d_torch = np.abs(dec - ref).max()
    print(f"[torch decomposition]  max|dec - ref| = {d_torch:.3e}")
    assert d_torch < 1e-4, "PyTorch decomposition does not match reference!"

    # ---- ONNX export + onnxruntime loop ----------------------------------------
    import onnxruntime as ort

    tmp = Path(tempfile.mkdtemp(prefix="pi0_onnx_tiny_"))
    prefix_path = str(tmp / "prefix.onnx")
    denoise_path = str(tmp / "denoise.onnx")

    p_args = (*images, *img_masks, lang_tokens, lang_masks)
    used = export_to_onnx(
        prefix, p_args, prefix_arg_names(len(cams)), ["keys", "values", "prefix_pad_masks"], prefix_path
    )
    print(f"[export] prefix.onnx via {used}")

    t0 = torch.full((B,), 1.0, dtype=torch.float32)
    d_args = (state, noise.clone(), t0, keys, values, ppm)
    used = export_to_onnx(
        denoise,
        d_args,
        ["state", "x_t", "timestep", "keys", "values", "prefix_pad_masks"],
        ["v_t"],
        denoise_path,
    )
    print(f"[export] denoise.onnx via {used}")

    psess = ort.InferenceSession(prefix_path, providers=["CPUExecutionProvider"])
    dsess = ort.InferenceSession(denoise_path, providers=["CPUExecutionProvider"])

    pfeeds = {}
    pin = [i.name for i in psess.get_inputs()]
    for i in range(len(cams)):
        pfeeds[pin[i]] = images[i].numpy()
        pfeeds[pin[len(cams) + i]] = img_masks[i].numpy()
    pfeeds[pin[-2]] = lang_tokens.numpy()
    pfeeds[pin[-1]] = lang_masks.numpy()
    k_o, v_o, ppm_o = psess.run(None, pfeeds)

    ort_call = make_ort_denoise_callable(dsess)
    onnx_dec = euler_loop_numpy((k_o, v_o, ppm_o, state.numpy()), ort_call, noise.numpy(), cfg.num_inference_steps)
    onnx_dec = onnx_dec[:, :, : cfg.output_features[ACTION].shape[0]]
    d_onnx = np.abs(onnx_dec - ref).max()
    print(f"[onnx  decomposition]  max|onnx - ref| = {d_onnx:.3e}")
    assert d_onnx < 1e-3, "ONNX decomposition does not match reference!"

    for f in tmp.glob("*"):
        f.unlink()
    tmp.rmdir()
    print("\nVALIDATION PASSED: decomposition + ONNX export logic is correct.")


if __name__ == "__main__":
    main()
