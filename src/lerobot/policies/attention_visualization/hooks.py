from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from types import MethodType
from typing import Any, Iterable

import cv2
import numpy as np
import torch

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


def _attn_debug_enabled() -> bool:
    v = os.environ.get("LEROBOT_ATTN_DEBUG", "")
    return v not in ("", "0", "false", "False", "no", "No")


def resolve_attention_context(policy: PreTrainedPolicy) -> "SmolVlaAttentionContext | PiAttentionContext | None":
    name = getattr(policy, "name", None)
    if name == "smolvla":
        ctx = SmolVlaAttentionContext(policy)
        ctx.enable()
        return ctx
    if name in ("pi05", "pi0"):
        ctx = PiAttentionContext(policy)
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
        self._call_id = 0

    def enable(self) -> None:
        if self._enabled:
            return

        model = getattr(self.policy, "model", None)
        vlm = getattr(model, "vlm_with_expert", None)
        if model is None or vlm is None:
            return

        vlm.save_attn = True
        vlm.last_attn = None

        original_eager_forward = getattr(vlm, "eager_attention_forward", None)
        if original_eager_forward is None:
            return

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

            query_states_f = query_states.to(dtype=torch.float32)
            key_states_f = key_states_expanded.to(dtype=torch.float32)

            query_states_f = query_states_f.transpose(1, 2)
            key_states_f = key_states_f.transpose(1, 2)

            att_weights = torch.matmul(query_states_f, key_states_f.transpose(2, 3))
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
            embs: list[torch.Tensor] = []
            pad_masks: list[torch.Tensor] = []
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
            embs_cat = torch.cat(embs, dim=1)
            pad_masks_cat = torch.cat(pad_masks, dim=1)
            att_masks_tensor = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks_cat.device)[None, :]

            seq_len = pad_masks_cat.shape[1]
            if seq_len < self.prefix_length:
                embs_cat = pad_tensor(embs_cat, self.prefix_length, pad_value=0)
                pad_masks_cat = pad_tensor(pad_masks_cat, self.prefix_length, pad_value=0)
                att_masks_tensor = pad_tensor(att_masks_tensor, self.prefix_length, pad_value=0)

            att_masks_tensor = att_masks_tensor.expand(bsize, -1)

            self.last_image_patch_ranges = img_patch_ranges if img_patch_ranges else None
            self.last_image_patch_names = present_img_keys if img_patch_ranges else None
            self._attn_query_mask = att_masks_tensor.detach()

            return embs_cat, pad_masks_cat, att_masks_tensor

        model.embed_prefix = MethodType(embed_prefix_with_ranges, model)

        self._enabled = True

    @staticmethod
    def _render_overlay(img_bgr: np.ndarray, attn_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = img_bgr.shape[:2]
        attn_resized = cv2.resize(attn_map, (w, h), interpolation=cv2.INTER_LINEAR)
        attn_uint8 = (attn_resized * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(attn_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_bgr, 0.5, heatmap, 0.5, 0.0)
        return overlay, attn_resized

    def collect_attentions(self, policy: PreTrainedPolicy, images_bgr: dict[str, np.ndarray]) -> list[AttentionSample]:
        model = getattr(policy, "model", None)
        vlm = getattr(model, "vlm_with_expert", None)
        if model is None or vlm is None:
            return []

        attn = getattr(vlm, "last_attn", None)
        ranges = getattr(model, "last_image_patch_ranges", None)
        names = getattr(model, "last_image_patch_names", None)
        if attn is None or ranges is None or names is None:
            return []

        if not isinstance(attn, torch.Tensor) or attn.ndim != 4 or attn.shape[0] < 1:
            return []
        if not torch.isfinite(attn).all():
            return []

        self._call_id += 1
        attn_b = attn[0].to(dtype=torch.float32)
        attn_mean = attn_b.mean(0)

        q_mask = getattr(model, "_attn_query_mask", None)
        q_idxs: np.ndarray | None = None
        if isinstance(q_mask, torch.Tensor) and q_mask.ndim == 2 and q_mask.shape[0] >= 1:
            q_idxs_np = torch.where(q_mask[0].to(dtype=torch.bool))[0].detach().cpu().numpy()
            if q_idxs_np.size > 0:
                q_idxs = q_idxs_np

        q_idx = int(q_idxs[-1]) if q_idxs is not None else int(attn_mean.shape[0] - 1)
        q_idx = max(0, min(q_idx, attn_mean.shape[0] - 1))
        q_row = attn_mean[q_idx]

        samples: list[AttentionSample] = []
        if _attn_debug_enabled():
            logging.info("[attn][smolvla] call_id=%d q_idx=%d n_imgs=%d", self._call_id, q_idx, len(ranges))

        for cam_key, img_range in zip(names, ranges, strict=False):
            img_bgr = images_bgr.get(cam_key)
            if img_bgr is None:
                continue

            img_start, img_end = img_range
            if img_end > q_row.shape[-1]:
                continue

            img_attn_raw_1d = q_row[img_start:img_end]
            n_patches = int(img_attn_raw_1d.shape[0])
            if n_patches <= 0:
                continue

            grid_size = int(round(float(n_patches) ** 0.5))
            if grid_size * grid_size != n_patches:
                continue

            raw_max = float(img_attn_raw_1d.max().item())
            raw_mean = float(img_attn_raw_1d.mean().item())
            raw_sum = float(img_attn_raw_1d.sum().item())

            denom = float(max(raw_sum, 1e-9))
            img_attn_norm_1d = (img_attn_raw_1d / denom).detach().cpu().numpy()
            img_attn_raw_np = img_attn_raw_1d.detach().cpu().numpy()

            attn_map = img_attn_norm_1d.reshape(grid_size, grid_size)
            attn_map_raw = img_attn_raw_np.reshape(grid_size, grid_size)

            attn_vis = attn_map - attn_map.min()
            maxv = float(attn_vis.max())
            if maxv > 0:
                attn_vis = attn_vis / maxv

            overlay, attn_resized = self._render_overlay(img_bgr, attn_vis)

            if _attn_debug_enabled():
                argmax = np.unravel_index(int(attn_map.argmax()), attn_map.shape)
                logging.info(
                    "[attn][smolvla] call_id=%d cam=%s img_range=%s img_mass=%.6f argmax(rc)=(%d,%d)",
                    self._call_id,
                    cam_key,
                    str(img_range),
                    raw_sum,
                    int(argmax[0]),
                    int(argmax[1]),
                )

            samples.append(
                AttentionSample(
                    camera_key=cam_key,
                    overlay_bgr=overlay,
                    attention_resized=attn_resized,
                    attention_patches=attn_vis,
                    attention_raw_patches=attn_map_raw,
                    original_bgr=img_bgr,
                    raw_max=raw_max,
                    raw_mean=raw_mean,
                    raw_sum=raw_sum,
                )
            )

        return samples


class PiAttentionContext:
    """
    π0 / π0.5 用のアテンション保存＋可視化ヘルパー。

    方針:
      - suffix(denoise側, gemma_expert)の attentions のみを使う (language側は混ぜない)
      - 1 chunk 生成中に gemma forward が複数回走ることがあるので, それらを1つの“chunk更新”としてまとめる
      - π0 は SmolVLA と同じ目的に寄せて, q(=アクション列) を exec_horizon で平均した “集約ヒートマップ” を出す
        (per-step でフラフラする見え方と, 1フレームだけ違う attention を引く現象を避ける)
      - π0.5 は従来どおり per-step 表示を維持
    """

    def __init__(self, policy: PreTrainedPolicy):
        self.policy = policy
        self._enabled = False

        self._chunk_id = 0
        self._step = 0
        self._last_seen_version = -1

        self._last_good_attn_suffix: torch.Tensor | None = None
        self._suppressed = 0

        # chunk生成の forward 群を “1つの更新” にまとめるためのタイミング
        self._last_forward_ts = -1e9
        self._chunk_gap_s = 0.35  # chunk生成間は >1s 程度になる想定, 生成中は <0.1s 程度
        self._pending_new_version = True
        self._best_score_in_group = -1e18

    @staticmethod
    def _entropy_score(img_slice: torch.Tensor, eps: float = 1e-9) -> tuple[float, float, float]:
        """
        img_slice: (n_img_tokens,) の attention (平均済み). softmax後なので非負.
        戻り値:
          - img_mass: 画像トークンへ割り当てられた総質量
          - entropy_norm: 0(尖り) .. 1(一様)
          - score: img_mass * (1 - entropy_norm)
        """
        if img_slice.numel() <= 1:
            return 0.0, 1.0, 0.0

        img_mass_t = img_slice.sum()
        img_mass = float(img_mass_t.item())
        if not math.isfinite(img_mass) or img_mass <= eps:
            return 0.0, 1.0, 0.0

        p = img_slice / (img_mass_t + eps)
        p = torch.clamp(p, min=eps)
        ent = float((-(p * torch.log(p))).sum().item())
        ent_norm = ent / float(max(math.log(float(p.numel())), eps))
        ent_norm = float(max(0.0, min(1.0, ent_norm)))
        score = float(img_mass * (1.0 - ent_norm))
        return img_mass, ent_norm, score

    def enable(self) -> None:
        if self._enabled:
            return

        model = getattr(self.policy, "model", None)
        paligemma = getattr(model, "paligemma_with_expert", None)
        if model is None or paligemma is None:
            return

        original_preprocess_images = self.policy._preprocess_images

        def preprocess_images_with_names(self, batch: dict[str, Any]):
            images, masks = original_preprocess_images(batch)
            present_img_keys = [key for key in self.config.image_features if key in batch]
            model._attn_present_img_keys = present_img_keys
            return images, masks

        self.policy._preprocess_images = MethodType(preprocess_images_with_names, self.policy)

        original_embed_prefix = model.embed_prefix

        def embed_prefix_with_ranges(
            self_model, images: Iterable[torch.Tensor], img_masks: Iterable[torch.Tensor], tokens, masks
        ):
            embs: list[torch.Tensor] = []
            pad_masks: list[torch.Tensor] = []
            att_masks: list[int] = []
            patch_ranges: list[tuple[int, int]] = []
            patch_names: list[str] = []
            cursor = 0

            present_img_keys = getattr(self_model, "_attn_present_img_keys", [])

            for idx, (img, img_mask) in enumerate(zip(images, img_masks, strict=True)):
                img_emb = self_model._apply_checkpoint(self_model.paligemma_with_expert.embed_image, img)
                bsize, num_img_embs = img_emb.shape[:2]
                embs.append(img_emb)
                pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
                att_masks += [0] * num_img_embs

                if bool(torch.any(img_mask)):
                    patch_ranges.append((cursor, cursor + num_img_embs))
                    if idx < len(present_img_keys):
                        patch_names.append(present_img_keys[idx])

                cursor += num_img_embs

            def lang_embed_func(tok):
                lang_emb = self_model.paligemma_with_expert.embed_language_tokens(tok)
                lang_emb_dim = lang_emb.shape[-1]
                return lang_emb * math.sqrt(lang_emb_dim)

            lang_emb = self_model._apply_checkpoint(lang_embed_func, tokens)
            embs.append(lang_emb)
            pad_masks.append(masks)
            att_masks += [0] * lang_emb.shape[1]

            embs_cat = torch.cat(embs, dim=1)
            pad_masks_cat = torch.cat(pad_masks, dim=1)
            att_masks_tensor = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks_cat.device)
            bsize = pad_masks_cat.shape[0]
            att_masks_tensor = att_masks_tensor[None, :].expand(bsize, len(att_masks))

            self_model.last_image_patch_ranges = patch_ranges
            self_model.last_image_patch_names = patch_names

            if patch_ranges:
                self_model._attn_total_img_range = (patch_ranges[0][0], patch_ranges[-1][1])
            else:
                self_model._attn_total_img_range = None

            return embs_cat, pad_masks_cat, att_masks_tensor

        model.embed_prefix = MethodType(embed_prefix_with_ranges, model)

        # language_model側のattentionは可視化に混ぜない
        lm = paligemma.paligemma.language_model
        try:
            lm.config.attn_implementation = "eager"
            lm.config._attn_implementation = "eager"
        except Exception:
            pass

        gemma_expert = getattr(paligemma, "gemma_expert", None)
        gemma_model = getattr(gemma_expert, "model", None) if gemma_expert is not None else None
        if gemma_model is None:
            return

        try:
            gemma_model.config.attn_implementation = "eager"
            gemma_model.config._attn_implementation = "eager"
        except Exception:
            pass
        gemma_model.config.output_attentions = True

        # 状態保存領域
        if not hasattr(model, "_attn_suffix_version"):
            model._attn_suffix_version = 0
        if not hasattr(model, "_attn_expected_q_len"):
            model._attn_expected_q_len = None
        if not hasattr(model, "_attn_expected_k_len"):
            model._attn_expected_k_len = None
        if not hasattr(model, "_attn_suffix_best_layer"):
            model._attn_suffix_best_layer = None
        if not hasattr(model, "_attn_suffix_best_mass"):
            model._attn_suffix_best_mass = None
        if not hasattr(model, "_attn_suffix_best_score"):
            model._attn_suffix_best_score = None
        if not hasattr(model, "_attn_suffix_q1"):
            model._attn_suffix_q1 = None

        original_gemma_forward = gemma_model.forward

        def gemma_forward_with_attn(self_gemma, *args, **kwargs):
            kwargs["output_attentions"] = True
            out = original_gemma_forward(*args, **kwargs)

            attn_layers = getattr(out, "attentions", None)
            if not attn_layers:
                return out

            # ---- chunk生成の forward 群をまとめる (時間ギャップで切る) ----
            now = time.monotonic()
            if (now - self._last_forward_ts) > self._chunk_gap_s:
                # 新しい chunk 生成が始まったとみなす
                self._pending_new_version = True
                self._best_score_in_group = -1e18
                # shape ロックは chunk 毎にやり直す (π0 で shape が変わるケースの対策)
                model._attn_expected_q_len = None
                model._attn_expected_k_len = None
            self._last_forward_ts = now

            a_last = attn_layers[-1]
            if not isinstance(a_last, torch.Tensor) or a_last.ndim != 4 or a_last.shape[0] < 1:
                self._suppressed += 1
                return out
            if not torch.isfinite(a_last).all():
                self._suppressed += 1
                return out

            q_len = int(a_last.shape[2])
            k_len = int(a_last.shape[3])

            total_img_range = getattr(model, "_attn_total_img_range", None)
            if total_img_range is None:
                self._suppressed += 1
                return out
            img0, img1 = int(total_img_range[0]), int(total_img_range[1])
            if not (0 <= img0 < img1 <= k_len):
                self._suppressed += 1
                return out

            rtc = getattr(self.policy.config, "rtc_config", None)
            exec_h = int(getattr(rtc, "execution_horizon", 0) or 0) if rtc is not None else 0
            if exec_h <= 0:
                exec_h = 10
            q1 = min(q_len, max(1, exec_h))

            expected_q = getattr(model, "_attn_expected_q_len", None)
            expected_k = getattr(model, "_attn_expected_k_len", None)

            # chunk内での shape 外れ値だけ捨てる
            if expected_q is None or expected_k is None:
                if q_len < q1 or img1 > k_len:
                    self._suppressed += 1
                    return out
                model._attn_expected_q_len = int(q_len)
                model._attn_expected_k_len = int(k_len)
                expected_q = int(q_len)
                expected_k = int(k_len)

            if int(q_len) != int(expected_q) or int(k_len) != int(expected_k):
                self._suppressed += 1
                if _attn_debug_enabled():
                    logging.info(
                        "[attn][%s] suppress(shape_outlier) q_len=%d k_len=%d expected_q=%d expected_k=%d",
                        getattr(self.policy, "name", "pi"),
                        q_len,
                        k_len,
                        int(expected_q),
                        int(expected_k),
                    )
                return out

            policy_name = getattr(self.policy, "name", "pi")
            use_entropy_score = policy_name == "pi0"  # π0 は “尖り” を重視して選ぶ

            best_layer = None
            best_mass = -1.0
            best_score = -1e18

            # 各レイヤーで “画像への注目度” を評価して最良を選ぶ
            for li, a in enumerate(attn_layers):
                if not isinstance(a, torch.Tensor) or a.ndim != 4 or a.shape[0] < 1:
                    continue
                if not torch.isfinite(a).all():
                    continue

                m = a[0].to(dtype=torch.float32).mean(0)  # (q, k)
                if m.shape[1] < img1:
                    continue

                q_avg = m[:q1].mean(0)  # (k,)
                img_slice = q_avg[img0:img1]

                img_mass, ent_norm, score = self._entropy_score(img_slice)
                layer_mass = float(img_mass)
                layer_score = float(score if use_entropy_score else img_mass)

                if layer_score > best_score:
                    best_score = layer_score
                    best_mass = layer_mass
                    best_layer = int(li)

            if best_layer is None:
                self._suppressed += 1
                return out

            chosen = attn_layers[best_layer]
            if not isinstance(chosen, torch.Tensor) or not torch.isfinite(chosen).all():
                self._suppressed += 1
                return out

            # chosen の “代表スコア” を計算して, chunk生成中のベストだけ残す
            m_chosen = chosen[0].to(dtype=torch.float32).mean(0)  # (q, k)
            q_avg_chosen = m_chosen[:q1].mean(0)
            img_slice_chosen = q_avg_chosen[img0:img1]
            img_mass_c, ent_norm_c, score_c = self._entropy_score(img_slice_chosen)
            rep_mass = float(img_mass_c)
            rep_score = float(score_c if use_entropy_score else img_mass_c)

            # “より良い” と判断したときだけ更新する (chunk中の中途半端な snapshot で上書きしない)
            if rep_score >= (self._best_score_in_group + 1e-12):
                self._best_score_in_group = rep_score

                model.last_attn_suffix = chosen.detach()
                model._attn_suffix_best_layer = int(best_layer)
                model._attn_suffix_best_mass = float(rep_mass)
                model._attn_suffix_best_score = float(rep_score)
                model._attn_suffix_q1 = int(q1)

                if self._pending_new_version:
                    model._attn_suffix_version = int(getattr(model, "_attn_suffix_version", 0)) + 1
                    self._pending_new_version = False

                if _attn_debug_enabled():
                    logging.info(
                        "[attn][%s] chunk_update version=%d best_layer=%d mass=%.6f score=%.6f entropy_norm=%.3f q_len=%d k_len=%d q1=%d",
                        policy_name,
                        int(model._attn_suffix_version),
                        int(best_layer),
                        float(rep_mass),
                        float(rep_score),
                        float(ent_norm_c),
                        q_len,
                        k_len,
                        int(q1),
                    )

            return out

        gemma_model.forward = MethodType(gemma_forward_with_attn, gemma_model)

        self._enabled = True

    @staticmethod
    def _render_overlay(img_bgr: np.ndarray, attn_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = img_bgr.shape[:2]
        attn_resized = cv2.resize(attn_map, (w, h), interpolation=cv2.INTER_LINEAR)
        attn_uint8 = (attn_resized * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(attn_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_bgr, 0.5, heatmap, 0.5, 0.0)
        return overlay, attn_resized

    def collect_attentions(self, policy: PreTrainedPolicy, images_bgr: dict[str, np.ndarray]) -> list[AttentionSample]:
        model = getattr(policy, "model", None)
        if model is None:
            return []

        ranges = getattr(model, "last_image_patch_ranges", None)
        names = getattr(model, "last_image_patch_names", None)
        if ranges is None or names is None:
            return []

        # suffixのみ使う. prefix fallbackはしない
        attn = getattr(model, "last_attn_suffix", None)
        if attn is None:
            attn = self._last_good_attn_suffix
        if attn is None:
            return []

        if not isinstance(attn, torch.Tensor) or attn.ndim != 4 or attn.shape[0] < 1:
            return []
        if not torch.isfinite(attn).all():
            return []

        # chunk update検出
        version = int(getattr(model, "_attn_suffix_version", 0))
        if version != self._last_seen_version:
            self._last_seen_version = version
            self._chunk_id += 1
            self._step = 0

        self._last_good_attn_suffix = attn

        attn_b = attn[0].to(dtype=torch.float32)  # (heads, q, k)
        attn_mean = attn_b.mean(0)  # (q, k)
        q_len = int(attn_mean.shape[0])
        k_len = int(attn_mean.shape[1])

        policy_name = getattr(policy, "name", "pi")
        is_pi0 = policy_name == "pi0"

        # π0: アクション列(先頭q1)で平均した “集約ヒートマップ”
        # π0.5: 従来どおりステップごとの query 行
        if is_pi0:
            q1 = int(getattr(model, "_attn_suffix_q1", 0) or 0)
            if q1 <= 0:
                rtc = getattr(policy.config, "rtc_config", None)
                exec_h = int(getattr(rtc, "execution_horizon", 0) or 0) if rtc is not None else 0
                if exec_h <= 0:
                    exec_h = 10
                q1 = min(q_len, max(1, exec_h))
            q1 = max(1, min(q_len, q1))
            q_row = attn_mean[:q1].mean(0)  # (k,)
            q_dbg = (0, q1)
        else:
            q_idx = int(max(0, min(self._step, q_len - 1)))
            q_row = attn_mean[q_idx]  # (k,)
            q_dbg = (q_idx, q_idx + 1)

        samples: list[AttentionSample] = []
        for img_range, cam_key in zip(ranges, names, strict=False):
            img_bgr = images_bgr.get(cam_key)
            if img_bgr is None:
                continue

            img_start, img_end = img_range
            if img_end > k_len:
                continue

            row_raw = q_row[img_start:img_end]
            n_patches = int(row_raw.shape[0])
            if n_patches <= 0:
                continue

            grid_size = int(round(float(n_patches) ** 0.5))
            if grid_size * grid_size != n_patches:
                continue

            raw_sum = float(row_raw.sum().item())
            raw_max = float(row_raw.max().item())
            raw_mean = float(row_raw.mean().item())

            if raw_sum <= 1e-9:
                continue

            row_norm = (row_raw / raw_sum).detach().cpu().numpy()
            row_raw_np = row_raw.detach().cpu().numpy()

            attn_map = row_norm.reshape(grid_size, grid_size)
            attn_map_raw = row_raw_np.reshape(grid_size, grid_size)

            attn_vis = attn_map - attn_map.min()
            maxv = float(attn_vis.max())
            if maxv > 0:
                attn_vis = attn_vis / maxv

            overlay, attn_resized = self._render_overlay(img_bgr, attn_vis)

            if _attn_debug_enabled():
                argmax = np.unravel_index(int(attn_map.argmax()), attn_map.shape)
                logging.info(
                    "[attn][%s] cam=%s chunk_id=%d step=%d q=(%d..%d)/%d k=%d img_range=%s img_mass=%.6f suppressed=%d argmax(rc)=(%d,%d) mode=%s",
                    policy_name,
                    cam_key,
                    int(self._chunk_id),
                    int(self._step),
                    int(q_dbg[0]),
                    int(q_dbg[1]),
                    int(q_len),
                    int(k_len),
                    str(img_range),
                    float(raw_sum),
                    int(self._suppressed),
                    int(argmax[0]),
                    int(argmax[1]),
                    "mean_q" if is_pi0 else "per_step",
                )

            samples.append(
                AttentionSample(
                    camera_key=cam_key,
                    overlay_bgr=overlay,
                    attention_resized=attn_resized,
                    attention_patches=attn_vis,
                    attention_raw_patches=attn_map_raw,
                    original_bgr=img_bgr,
                    raw_max=raw_max,
                    raw_mean=raw_mean,
                    raw_sum=raw_sum,
                )
            )

        # π0.5 のみ step を進める (π0 は mean_q なので固定)
        if not is_pi0:
            n_action_steps = int(getattr(policy.config, "n_action_steps", 1) or 1)
            if n_action_steps > 1:
                self._step = min(self._step + 1, n_action_steps - 1)

        return samples
