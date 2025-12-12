from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from types import MethodType
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from transformers.models.gemma import modeling_gemma  # noqa: F401

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.modeling_smolvla import pad_tensor


@dataclass
class AttentionSample:
    camera_key: str
    overlay_bgr: np.ndarray
    attention_resized: np.ndarray
    attention_patches: np.ndarray
    attention_raw_patches: np.ndarray
    original_bgr: np.ndarray
    raw_max: float
    raw_mean: float
    raw_sum: float


_ATTN_DEBUG = os.getenv("LEROBOT_ATTN_DEBUG", "0").lower() not in ("0", "false", "no", "off", "")
# 画像への attention 確率質量が小さすぎるときの抑制 (必要なら env で上げ下げ)
_ATTN_MIN_IMAGE_MASS = float(os.getenv("LEROBOT_ATTN_MIN_IMAGE_MASS", "0.0"))
# suffix の denoise attention を何ステップ平均するか
_ATTN_SUFFIX_AVG_STEPS = int(os.getenv("LEROBOT_ATTN_SUFFIX_AVG_STEPS", "4"))
# 1step の query だけ見るか, 周辺も平均するか (奇数推奨, 1ならそのstepのみ)
_ATTN_Q_WINDOW = max(1, int(os.getenv("LEROBOT_ATTN_Q_WINDOW", "1")))


def resolve_attention_context(
    policy: PreTrainedPolicy,
) -> "SmolVlaAttentionContext | Pi05AttentionContext | None":
    if getattr(policy, "name", None) == "smolvla":
        ctx = SmolVlaAttentionContext(policy)
        ctx.enable()
        return ctx
    if getattr(policy, "name", None) == "pi05":
        ctx = Pi05AttentionContext(policy)
        ctx.enable()
        return ctx
    return None


class SmolVlaAttentionContext:
    """
    SmolVLA 用のアテンション保存＋可視化ヘルパー。
    """

    def __init__(self, policy: PreTrainedPolicy):
        self.policy = policy
        self._enabled = False

    def enable(self) -> None:
        if self._enabled:
            return

        model = getattr(self.policy, "model", None)
        vlm = getattr(model, "vlm_with_expert", None)
        if model is None or vlm is None:
            return

        vlm.save_attn = True
        vlm.last_attn = None

        def eager_attention_forward_with_save(
            self,
            attention_mask: torch.Tensor,
            batch_size: int,
            head_dim: int,
            query_states: torch.Tensor,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
        ) -> torch.Tensor:
            num_att_heads = self.num_attention_heads
            num_key_value_heads = self.num_key_value_heads
            num_key_value_groups = num_att_heads // num_key_value_heads

            sequence_length = key_states.shape[1]

            key_states_expanded = key_states[:, :, :, None, :].expand(
                batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
            )
            key_states_expanded = key_states_expanded.reshape(
                batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
            )

            value_states_expanded = value_states[:, :, :, None, :].expand(
                batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
            )
            value_states_expanded = value_states_expanded.reshape(
                batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
            )

            query_states = query_states.to(dtype=torch.float32)
            key_states_cast = key_states_expanded.to(dtype=torch.float32)

            query_states = query_states.transpose(1, 2)
            key_states_cast = key_states_cast.transpose(1, 2)

            att_weights = torch.matmul(query_states, key_states_cast.transpose(2, 3))
            att_weights *= head_dim**-0.5

            att_weights = att_weights.to(dtype=torch.float32)
            big_neg = torch.finfo(att_weights.dtype).min
            masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
            probs = torch.nn.functional.softmax(masked_att_weights, dim=-1)
            probs = probs.to(dtype=value_states_expanded.dtype)

            if getattr(self, "save_attn", False):
                self.last_attn = probs.detach()

            att_output = torch.matmul(probs, value_states_expanded.permute(0, 2, 1, 3))

            att_output = att_output.permute(0, 2, 1, 3)
            att_output = att_output.reshape(
                batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim
            )

            return att_output

        vlm.eager_attention_forward = MethodType(eager_attention_forward_with_save, vlm)

        original_prepare_images = self.policy.prepare_images

        def prepare_images_with_names(
            self, batch: dict[str, Any]
        ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
            images, masks = original_prepare_images(batch)
            present_img_keys = [key for key in self.config.image_features if key in batch]
            setattr(self.model, "_attn_present_img_keys", present_img_keys[: len(images)])
            return images, masks

        self.policy.prepare_images = MethodType(prepare_images_with_names, self.policy)

        original_embed_prefix = model.embed_prefix

        def embed_prefix_with_ranges(
            self,
            images: Iterable[torch.Tensor],
            img_masks: Iterable[torch.Tensor],
            lang_tokens: torch.Tensor,
            lang_masks: torch.Tensor,
            state: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            images = list(images)
            img_masks = list(img_masks)
            embs = []
            pad_masks = []
            att_masks: list[int] = []

            present_img_keys = getattr(
                self, "_attn_present_img_keys", list(self.config.image_features.keys())[: len(images)]
            )
            token_cursor = 0
            img_patch_ranges: list[tuple[int, int]] = []

            for img, img_mask in zip(images, img_masks, strict=False):
                if self.add_image_special_tokens:
                    image_start_token = (
                        self.vlm_with_expert.embed_language_tokens(
                            self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                        )
                        .unsqueeze(0)
                        .expand(img.shape[0], -1, -1)
                    )
                    image_start_mask = torch.ones_like(
                        image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                    )
                    att_masks += [0] * image_start_mask.shape[-1]
                    embs.append(image_start_token)
                    pad_masks.append(image_start_mask)
                    token_cursor += image_start_token.shape[1]

                img_patch_start = token_cursor

                img_emb = self.vlm_with_expert.embed_image(img)

                img_emb_dim = img_emb.shape[-1]
                img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

                bsize, num_img_embs = img_emb.shape[:2]
                expanded_img_mask = img_mask[:, None].expand(bsize, num_img_embs)

                embs.append(img_emb)
                pad_masks.append(expanded_img_mask)

                att_masks += [0] * num_img_embs
                token_cursor += num_img_embs
                img_patch_end = token_cursor
                img_patch_ranges.append((int(img_patch_start), int(img_patch_end)))

                if self.add_image_special_tokens:
                    image_end_token = (
                        self.vlm_with_expert.embed_language_tokens(
                            self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                        )
                        .unsqueeze(0)
                        .expand(img.shape[0], -1, -1)
                    )
                    image_end_mask = torch.ones_like(
                        image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                    )
                    embs.append(image_end_token)
                    pad_masks.append(image_end_mask)
                    att_masks += [0] * image_end_token.shape[1]
                    token_cursor += image_end_token.shape[1]

            lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            lang_emb = lang_emb * math.sqrt(lang_emb_dim)

            embs.append(lang_emb)
            pad_masks.append(lang_masks)

            num_lang_embs = lang_emb.shape[1]
            att_masks += [0] * num_lang_embs
            token_cursor += num_lang_embs

            state_emb = self.state_proj(state)
            state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
            embs.append(state_emb)
            bsize = state_emb.shape[0]
            device = state_emb.device

            states_seq_len = state_emb.shape[1]
            state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            att_masks += [1] * states_seq_len
            embs = torch.cat(embs, dim=1)
            pad_masks = torch.cat(pad_masks, dim=1)
            att_masks_tensor = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)[None, :]

            seq_len = pad_masks.shape[1]
            if seq_len < self.prefix_length:
                embs = pad_tensor(embs, self.prefix_length, pad_value=0)
                pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
                att_masks_tensor = pad_tensor(att_masks_tensor, self.prefix_length, pad_value=0)

            att_masks_tensor = att_masks_tensor.expand(bsize, -1)

            self.last_image_patch_ranges = img_patch_ranges if img_patch_ranges else None
            self.last_image_patch_range = img_patch_ranges[0] if img_patch_ranges else None
            self.last_image_patch_names = present_img_keys if img_patch_ranges else None

            return embs, pad_masks, att_masks_tensor

        model.embed_prefix = MethodType(embed_prefix_with_ranges, model)

        self._enabled = True

    @staticmethod
    def _compute_attention_map(
        attn: torch.Tensor, img_range: tuple[int, int]
    ) -> tuple[np.ndarray, float, float, float] | tuple[None, None, None, None]:
        if not isinstance(attn, torch.Tensor) or attn.ndim != 4 or attn.shape[0] < 1:
            return None, None, None, None
        attn_b = attn[0]
        attn_mean_heads = attn_b.mean(0)
        last_q = attn_mean_heads[-1]
        img_start, img_end = img_range
        if img_end > last_q.shape[-1]:
            return None, None, None, None
        img_attn = last_q[img_start:img_end]
        n_patches = img_attn.shape[0]
        if n_patches <= 0:
            return None, None, None, None
        grid_size = int(round(float(n_patches) ** 0.5))
        if grid_size * grid_size != n_patches:
            return None, None, None, None
        img_attn = img_attn.to(dtype=torch.float32)
        raw_max = float(img_attn.max().item())
        raw_mean = float(img_attn.mean().item())
        raw_sum = float(img_attn.sum().item())
        attn_map_raw = img_attn.detach().cpu().numpy().reshape(grid_size, grid_size)
        return attn_map_raw, raw_max, raw_mean, raw_sum

    @staticmethod
    def _render_overlay(img_bgr: np.ndarray, attn_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = img_bgr.shape[:2]
        attn_resized = cv2.resize(attn_map, (w, h), interpolation=cv2.INTER_LINEAR)
        attn_uint8 = (attn_resized * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(attn_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_bgr, 0.5, heatmap, 0.5, 0.0)
        return overlay, attn_resized

    def collect_attentions(
        self, policy: PreTrainedPolicy, images_bgr: dict[str, np.ndarray]
    ) -> list[AttentionSample]:
        model = getattr(policy, "model", None)
        vlm = getattr(model, "vlm_with_expert", None)
        if model is None or vlm is None:
            return []
        attn = getattr(vlm, "last_attn", None)
        ranges = getattr(model, "last_image_patch_ranges", None)
        names = getattr(model, "last_image_patch_names", None)
        if attn is None or ranges is None or names is None:
            return []

        samples: list[AttentionSample] = []
        for cam_key, img_range in zip(names, ranges, strict=False):
            img_bgr = images_bgr.get(cam_key)
            if img_bgr is None:
                continue
            attn_map_raw, raw_max, raw_mean, raw_sum = self._compute_attention_map(attn, img_range)
            if attn_map_raw is None:
                continue
            attn_map = attn_map_raw - attn_map_raw.min()
            maxv = attn_map.max()
            if maxv > 0:
                attn_map = attn_map / maxv
            overlay, attn_resized = self._render_overlay(img_bgr, attn_map)
            samples.append(
                AttentionSample(
                    camera_key=cam_key,
                    overlay_bgr=overlay,
                    attention_resized=attn_resized,
                    attention_patches=attn_map,
                    attention_raw_patches=attn_map_raw,
                    original_bgr=img_bgr,
                    raw_max=raw_max if raw_max is not None else 0.0,
                    raw_mean=raw_mean if raw_mean is not None else 0.0,
                    raw_sum=raw_sum if raw_sum is not None else 0.0,
                )
            )
        return samples


class Pi05AttentionContext:
    """
    PI0.5 用のアテンション保存＋可視化ヘルパー。

    改善点:
    - query を全平均せず, chunk 内 step に対応する query 行を見る
    - out.attentions の最終層固定をやめ, “画像を見ている層” を自動選択する
    """

    def __init__(self, policy: PreTrainedPolicy):
        self.policy = policy
        self._enabled = False
        self._vis_step_in_chunk = 0
        self._last_seen_chunk_id: int | None = None

    def enable(self) -> None:
        if self._enabled:
            return

        model = getattr(self.policy, "model", None)
        paligemma = getattr(model, "paligemma_with_expert", None)
        if model is None or paligemma is None:
            return

        # 画像キーを覚えておくために _preprocess_images をラップ
        original_preprocess_images = self.policy._preprocess_images

        def preprocess_images_with_names(self, batch: dict[str, Any]):
            images, masks = original_preprocess_images(batch)
            present_img_keys = [key for key in self.config.image_features if key in batch]
            setattr(model, "_attn_present_img_keys", present_img_keys)
            return images, masks

        self.policy._preprocess_images = MethodType(preprocess_images_with_names, self.policy)

        # embed_image をフックして “画像トークン長” と “processed_hw” を取る
        original_embed_image = paligemma.embed_image

        def embed_image_with_capture(img: torch.Tensor):
            out = original_embed_image(img)
            if getattr(model, "_attn_vis_capturing", False):
                lst = getattr(model, "_attn_vis_img_token_lens", None)
                if lst is None:
                    lst = []
                lst.append(int(out.shape[1]))
                setattr(model, "_attn_vis_img_token_lens", lst)
                if isinstance(img, torch.Tensor) and img.ndim == 4:
                    setattr(model, "_attn_vis_processed_hw", (int(img.shape[-2]), int(img.shape[-1])))
            return out

        paligemma.embed_image = embed_image_with_capture

        # embed_prefix をラップして画像パッチ範囲を記録 (original を呼ぶだけ)
        original_embed_prefix = model.embed_prefix

        def embed_prefix_with_ranges(
            self_model, images: Iterable[torch.Tensor], img_masks: Iterable[torch.Tensor], tokens, masks
        ):
            setattr(self_model, "_attn_vis_capturing", True)
            setattr(self_model, "_attn_vis_img_token_lens", [])
            setattr(self_model, "_attn_vis_processed_hw", None)
            setattr(self_model, "_attn_vis_chunk_id", int(getattr(self_model, "_attn_vis_chunk_id", 0)) + 1)
            try:
                out = original_embed_prefix(images, img_masks, tokens, masks)
            finally:
                setattr(self_model, "_attn_vis_capturing", False)

            if not isinstance(out, (tuple, list)) or len(out) != 3:
                return out

            embs, pad_masks, att_masks_tensor = out
            setattr(self_model, "_attn_vis_prefix_len", int(embs.shape[1]))

            img_token_lens: list[int] = list(getattr(self_model, "_attn_vis_img_token_lens", []))
            present_img_keys: list[str] = list(getattr(self_model, "_attn_present_img_keys", []))

            patch_ranges: list[tuple[int, int]] = []
            patch_names: list[str] = []
            cursor = 0

            images = list(images)
            img_masks = list(img_masks)
            n = min(len(img_token_lens), len(images))
            for idx in range(n):
                num_img_tokens = int(img_token_lens[idx])
                img_mask = img_masks[idx]
                if bool(torch.any(img_mask)):
                    patch_ranges.append((cursor, cursor + num_img_tokens))
                    if idx < len(present_img_keys):
                        patch_names.append(present_img_keys[idx])
                cursor += num_img_tokens

            self_model.last_image_patch_ranges = patch_ranges
            self_model.last_image_patch_names = patch_names

            return embs, pad_masks, att_masks_tensor

        model.embed_prefix = MethodType(embed_prefix_with_ranges, model)

        # language_model forward で prefix attentions を保存
        lm = paligemma.paligemma.language_model
        try:
            lm.config.attn_implementation = "eager"
            lm.config._attn_implementation = "eager"
        except Exception:
            pass
        lm.config.output_attentions = True
        original_lm_forward = lm.forward

        def lm_forward_with_attn(self_lm, *args, **kwargs):
            kwargs["output_attentions"] = True
            out = original_lm_forward(*args, **kwargs)
            attn = getattr(out, "attentions", None)
            setattr(model, "last_attn_prefix", attn if attn is not None else None)
            return out

        lm.forward = MethodType(lm_forward_with_attn, lm)

        # gemma_expert forward で suffix attentions を保存 (best layer を選ぶ)
        gemma_expert = getattr(paligemma, "gemma_expert", None)
        gemma_model = getattr(gemma_expert, "model", None) if gemma_expert is not None else None
        if gemma_model is not None:
            try:
                gemma_model.config.attn_implementation = "eager"
                gemma_model.config._attn_implementation = "eager"
            except Exception:
                pass
            gemma_model.config.output_attentions = True
            original_gemma_forward = gemma_model.forward

            def _select_best_layer(attentions: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> int:
                ranges = getattr(model, "last_image_patch_ranges", None)
                if not ranges:
                    return int(len(attentions) - 1)
                img0 = min(s for s, _ in ranges)
                img1 = max(e for _, e in ranges)

                # “実行に効く”側を優先したいなら最初の exec_horizon 分だけで選ぶ
                exec_h = None
                rtc = getattr(self.policy.config, "rtc_config", None)
                if rtc is not None:
                    exec_h = getattr(rtc, "execution_horizon", None)
                if exec_h is None:
                    exec_h = getattr(self.policy.config, "n_action_steps", None)
                exec_h = int(exec_h) if exec_h is not None else 0

                best_i = int(len(attentions) - 1)
                best_mass = -1.0
                for i, a in enumerate(attentions):
                    if not isinstance(a, torch.Tensor) or a.ndim != 4 or a.shape[0] < 1:
                        continue
                    # (heads, q, k) -> (q, k)
                    m = a[0].to(dtype=torch.float32).mean(0)
                    q_len, k_len = m.shape
                    q1 = min(q_len, max(1, exec_h)) if exec_h > 0 else q_len
                    if img1 > k_len:
                        continue
                    mass = float(m[:q1, img0:img1].sum(dim=-1).mean().item())
                    if mass > best_mass:
                        best_mass = mass
                        best_i = int(i)
                setattr(model, "_attn_best_layer_idx", best_i)
                setattr(model, "_attn_best_layer_mass", float(best_mass))
                return best_i

            def gemma_forward_with_attn(self_gemma, *args, **kwargs):
                kwargs["output_attentions"] = True
                out = original_gemma_forward(*args, **kwargs)
                attn_list = getattr(out, "attentions", None)

                if attn_list is None:
                    setattr(model, "last_attn_suffix", None)
                    return out

                # best layer を選んで保存
                best_i = _select_best_layer(attn_list)
                picked = attn_list[best_i].detach()
                setattr(model, "last_attn_suffix", picked)

                # denoise step を蓄積
                steps = getattr(model, "_attn_suffix_steps", None)
                if steps is None:
                    steps = []
                steps.append(picked)
                # メモリ暴走防止
                if len(steps) > 32:
                    steps = steps[-32:]
                setattr(model, "_attn_suffix_steps", steps)

                if _ATTN_DEBUG:
                    chunk_id = int(getattr(model, "_attn_vis_chunk_id", 0) or 0)
                    last_logged = int(getattr(model, "_attn_best_layer_logged_chunk_id", -1))
                    if chunk_id != last_logged:
                        setattr(model, "_attn_best_layer_logged_chunk_id", chunk_id)
                        logging.info(
                            "[attn][pi05] chunk_id=%d processed_hw=%s best_layer=%d best_layer_mass=%.6f",
                            chunk_id,
                            str(getattr(model, "_attn_vis_processed_hw", None)),
                            int(getattr(model, "_attn_best_layer_idx", -1)),
                            float(getattr(model, "_attn_best_layer_mass", -1.0)),
                        )

                return out

            gemma_model.forward = MethodType(gemma_forward_with_attn, gemma_model)

        self._enabled = True

    @staticmethod
    def _resize_with_pad_params(
        orig_hw: tuple[int, int], target_hw: tuple[int, int]
    ) -> tuple[float, int, int, int, int]:
        oh, ow = orig_hw
        th, tw = target_hw
        scale = min(tw / ow, th / oh)
        new_w = int(round(ow * scale))
        new_h = int(round(oh * scale))
        pad_w = tw - new_w
        pad_h = th - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        return scale, new_h, new_w, pad_top, pad_left

    @classmethod
    def _map_grid_to_original_image(
        cls,
        grid: np.ndarray,
        *,
        orig_hw: tuple[int, int],
        processed_hw: tuple[int, int] | None,
    ) -> np.ndarray:
        oh, ow = orig_hw
        if processed_hw is None:
            return cv2.resize(grid, (ow, oh), interpolation=cv2.INTER_LINEAR)

        ph, pw = processed_hw
        proc = cv2.resize(grid, (pw, ph), interpolation=cv2.INTER_LINEAR)

        _, new_h, new_w, pad_top, pad_left = cls._resize_with_pad_params((oh, ow), (ph, pw))
        y0 = max(0, min(ph, pad_top))
        y1 = max(0, min(ph, pad_top + new_h))
        x0 = max(0, min(pw, pad_left))
        x1 = max(0, min(pw, pad_left + new_w))
        cropped = proc[y0:y1, x0:x1]
        if cropped.size == 0:
            return cv2.resize(proc, (ow, oh), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(cropped, (ow, oh), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _render_overlay(img_bgr: np.ndarray, attn_map_01: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = img_bgr.shape[:2]
        attn_resized = cv2.resize(attn_map_01, (w, h), interpolation=cv2.INTER_LINEAR)
        attn_uint8 = (attn_resized * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(attn_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_bgr, 0.5, heatmap, 0.5, 0.0)
        return overlay, attn_resized

    @classmethod
    def _compute_attention_map(
        cls,
        attn: torch.Tensor,
        img_range: tuple[int, int],
        *,
        processed_hw: tuple[int, int] | None,
        orig_hw: tuple[int, int],
        q_index: int | None,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float, dict[str, float]] | tuple[None, None, None, None, None, None]:
        if not isinstance(attn, torch.Tensor) or attn.ndim != 4 or attn.shape[0] < 1:
            return None, None, None, None, None, None

        attn_b = attn[0].to(dtype=torch.float32)  # (heads, q, k)
        attn_mean = attn_b.mean(0)  # (q, k)

        q_len, k_len = attn_mean.shape
        img_start, img_end = img_range
        if img_end > k_len or img_start < 0 or img_start >= img_end:
            return None, None, None, None, None, None

        # query slice: plan_step (=chunk_step) に対応する 1行 (or window)
        if q_index is None:
            q0, q1 = 0, q_len
        else:
            qi = int(max(0, min(q_len - 1, q_index)))
            win = int(_ATTN_Q_WINDOW)
            half = win // 2
            q0 = max(0, qi - half)
            q1 = min(q_len, qi + half + 1)

        q_to_img = attn_mean[q0:q1, img_start:img_end]  # (nq, n_patches)
        if q_to_img.numel() == 0:
            return None, None, None, None, None, None

        img_mass_per_q = q_to_img.sum(dim=-1)  # (nq,)
        img_mass = float(img_mass_per_q.mean().item())

        raw_vec = q_to_img.mean(0)  # (n_patches,)
        raw_max = float(raw_vec.max().item())
        raw_mean = float(raw_vec.mean().item())
        raw_sum = float(raw_vec.sum().item())

        n_patches = int(raw_vec.shape[0])
        grid_size = int(round(float(n_patches) ** 0.5))
        if grid_size * grid_size != n_patches:
            return None, None, None, None, None, None

        raw_grid = raw_vec.detach().cpu().numpy().reshape(grid_size, grid_size)

        # conditioned: 画像パッチ内で再正規化
        denom = img_mass_per_q[:, None].clamp_min(1e-9)
        cond = q_to_img / denom
        w = img_mass_per_q.clamp_min(0)
        if float(w.sum().item()) > 0:
            w = w / w.sum()
            cond_vec = (cond * w[:, None]).sum(0)
        else:
            cond_vec = cond.mean(0)

        cond_vec_np = cond_vec.detach().cpu().numpy().reshape(grid_size, grid_size)
        vis_grid = cond_vec_np - cond_vec_np.min()
        vmax = float(vis_grid.max())
        if vmax > 0:
            vis_grid = vis_grid / vmax

        suppressed = False
        if img_mass < float(_ATTN_MIN_IMAGE_MASS):
            suppressed = True
            vis_grid = np.zeros_like(vis_grid, dtype=np.float32)
            raw_grid = np.zeros_like(raw_grid, dtype=np.float32)

        vis_map_01 = cls._map_grid_to_original_image(
            vis_grid.astype(np.float32),
            orig_hw=orig_hw,
            processed_hw=processed_hw,
        )
        raw_map = cls._map_grid_to_original_image(
            raw_grid.astype(np.float32),
            orig_hw=orig_hw,
            processed_hw=processed_hw,
        )

        argmax = int(np.argmax(cond_vec_np))
        argmax_rc = (int(argmax // grid_size), int(argmax % grid_size))

        diag = {
            "q_len": float(q_len),
            "k_len": float(k_len),
            "q0": float(q0),
            "q1": float(q1),
            "img_mass": float(img_mass),
            "suppressed": float(1.0 if suppressed else 0.0),
            "argmax_r": float(argmax_rc[0]),
            "argmax_c": float(argmax_rc[1]),
        }
        return vis_map_01, raw_map, raw_max, raw_mean, raw_sum, diag

    def collect_attentions(
        self, policy: PreTrainedPolicy, images_bgr: dict[str, np.ndarray]
    ) -> list[AttentionSample]:
        model = getattr(policy, "model", None)
        if model is None:
            return []

        ranges = getattr(model, "last_image_patch_ranges", None)
        names = getattr(model, "last_image_patch_names", None)
        if ranges is None or names is None:
            return []

        processed_hw = getattr(model, "_attn_vis_processed_hw", None)

        # chunk_id が進んだら step を 0 に戻す (RTC chunk 更新に同期)
        chunk_id = int(getattr(model, "_attn_vis_chunk_id", 0) or 0)
        if self._last_seen_chunk_id is None or chunk_id != self._last_seen_chunk_id:
            self._last_seen_chunk_id = chunk_id
            self._vis_step_in_chunk = 0

        chunk_size = int(getattr(policy.config, "n_action_steps", 1) or 1)
        q_index = int(self._vis_step_in_chunk)
        self._vis_step_in_chunk = (self._vis_step_in_chunk + 1) % max(1, chunk_size)

        # suffix を優先 (denoise steps の平均を使う)
        steps = getattr(model, "_attn_suffix_steps", None)
        attn = None
        if isinstance(steps, list) and len(steps) > 0:
            tail = steps[-max(1, _ATTN_SUFFIX_AVG_STEPS) :]
            try:
                attn = torch.stack(tail, dim=0).mean(0)
            except Exception:
                attn = tail[-1]
        if attn is None:
            attn = getattr(model, "last_attn_suffix", None)

        if attn is None:
            # prefix フォールバック (ただし意味は弱め)
            attn_list = getattr(model, "last_attn_prefix", None)
            if isinstance(attn_list, (list, tuple)) and len(attn_list) > 0:
                attn = attn_list[-1]
            else:
                attn = None

        if attn is None:
            return []

        samples: list[AttentionSample] = []
        for img_range, cam_key in zip(ranges, names, strict=False):
            img_bgr = images_bgr.get(cam_key)
            if img_bgr is None:
                continue
            orig_hw = (int(img_bgr.shape[0]), int(img_bgr.shape[1]))

            vis_map_01, raw_map, raw_max, raw_mean, raw_sum, diag = self._compute_attention_map(
                attn,
                img_range,
                processed_hw=processed_hw if isinstance(processed_hw, tuple) else None,
                orig_hw=orig_hw,
                q_index=q_index,
            )
            if vis_map_01 is None or raw_map is None:
                continue

            if _ATTN_DEBUG and diag is not None:
                logging.info(
                    "[attn][pi05] cam=%s chunk_id=%d step=%d q=(%d..%d)/%d k=%d img_range=%s img_mass=%.6f suppressed=%d argmax(rc)=(%d,%d)",
                    cam_key,
                    chunk_id,
                    int(q_index),
                    int(diag["q0"]),
                    int(diag["q1"]),
                    int(diag["q_len"]),
                    int(diag["k_len"]),
                    str(img_range),
                    float(diag["img_mass"]),
                    int(diag["suppressed"]),
                    int(diag["argmax_r"]),
                    int(diag["argmax_c"]),
                )

            overlay, attn_resized = self._render_overlay(img_bgr, vis_map_01)

            samples.append(
                AttentionSample(
                    camera_key=cam_key,
                    overlay_bgr=overlay,
                    attention_resized=attn_resized,
                    attention_patches=vis_map_01,
                    attention_raw_patches=raw_map,
                    original_bgr=img_bgr,
                    raw_max=float(raw_max or 0.0),
                    raw_mean=float(raw_mean or 0.0),
                    raw_sum=float(raw_sum or 0.0),
                )
            )

        return samples
