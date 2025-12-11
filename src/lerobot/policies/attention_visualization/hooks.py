from __future__ import annotations

import math
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


def resolve_attention_context(policy: PreTrainedPolicy) -> "SmolVlaAttentionContext | None":
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

    既存のクラスをパッチするだけで、元ファイルを変更しない。
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

        # --- attention を保存するために attention forward をラップ ---
        vlm.save_attn = True
        vlm.last_attn = None

        original_eager_forward = vlm.eager_attention_forward

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

        # --- 画像キーを把握するため prepare_images をラップ ---
        original_prepare_images = self.policy.prepare_images

        def prepare_images_with_names(
            self, batch: dict[str, Any]
        ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
            images, masks = original_prepare_images(batch)
            present_img_keys = [key for key in self.config.image_features if key in batch]
            setattr(self.model, "_attn_present_img_keys", present_img_keys[: len(images)])
            return images, masks

        self.policy.prepare_images = MethodType(prepare_images_with_names, self.policy)

        # --- embed_prefix を差し替えてパッチ範囲を記録 ---
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
        # 最後の query トークンの分布を使用（各画像パッチに対応するスライス）
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
        # bfloat16 を numpy で扱えるように float32 に揃える
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
            # 個別オーバーレイ用にカメラ内で正規化
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

    PaliGemma の language_model から attentions を抜き出し、
    embed_prefix の画像パッチ範囲を記録して可視化に利用する。
    """

    def __init__(self, policy: PreTrainedPolicy):
        self.policy = policy
        self._enabled = False

    def enable(self) -> None:
        if self._enabled:
            return

        model = getattr(self.policy, "model", None)
        paligemma = getattr(model, "paligemma_with_expert", None)
        if model is None or paligemma is None:
            return

        # --- 画像キーを覚えておくために _preprocess_images をラップ ---
        original_preprocess_images = self.policy._preprocess_images

        def preprocess_images_with_names(self, batch: dict[str, Any]):
            images, masks = original_preprocess_images(batch)
            present_img_keys = [key for key in self.config.image_features if key in batch]
            missing_img_keys = [key for key in self.config.image_features if key not in batch]
            # 期待順でキーを保持（欠損も埋めておく）
            model._attn_present_img_keys = present_img_keys + missing_img_keys
            return images, masks

        self.policy._preprocess_images = MethodType(preprocess_images_with_names, self.policy)

        # --- embed_prefix をラップして画像パッチ範囲を記録 ---
        original_embed_prefix = model.embed_prefix

        def embed_prefix_with_ranges(
            self_model, images: Iterable[torch.Tensor], img_masks: Iterable[torch.Tensor], tokens, masks
        ):
            embs = []
            pad_masks = []
            att_masks: list[int] = []
            patch_ranges: list[tuple[int, int]] = []
            cursor = 0

            for img, img_mask in zip(images, img_masks, strict=True):
                img_emb = self_model._apply_checkpoint(self_model.paligemma_with_expert.embed_image, img)
                bsize, num_img_embs = img_emb.shape[:2]
                embs.append(img_emb)
                pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
                att_masks += [0] * num_img_embs
                patch_ranges.append((cursor, cursor + num_img_embs))
                cursor += num_img_embs

            def lang_embed_func(tok):
                lang_emb = self_model.paligemma_with_expert.embed_language_tokens(tok)
                lang_emb_dim = lang_emb.shape[-1]
                return lang_emb * math.sqrt(lang_emb_dim)

            lang_emb = self_model._apply_checkpoint(lang_embed_func, tokens)
            embs.append(lang_emb)
            pad_masks.append(masks)
            att_masks += [0] * lang_emb.shape[1]

            embs = torch.cat(embs, dim=1)
            pad_masks = torch.cat(pad_masks, dim=1)
            att_masks_tensor = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
            bsize = pad_masks.shape[0]
            att_masks_tensor = att_masks_tensor[None, :].expand(bsize, len(att_masks))

            # 画像パッチの範囲と名前を記録（可視化用）
            self_model.last_image_patch_ranges = patch_ranges
            self_model.last_image_patch_names = getattr(self_model, "_attn_present_img_keys", [])

            return embs, pad_masks, att_masks_tensor

        model.embed_prefix = MethodType(embed_prefix_with_ranges, model)

        # --- language_model の forward をラップして attentions を保存 ---
        lm = paligemma.paligemma.language_model
        # sdpa のまま output_attentions を有効化すると transformers 側で拒否されるため eager に切り替える
        try:
            lm.config.attn_implementation = "eager"
            lm.config._attn_implementation = "eager"  # some versions check the private field
        except Exception:
            pass
        lm.config.output_attentions = True
        original_lm_forward = lm.forward

        def lm_forward_with_attn(self_lm, *args, **kwargs):
            kwargs["output_attentions"] = True
            out = original_lm_forward(*args, **kwargs)
            attn = getattr(out, "attentions", None)
            if attn is not None:
                setattr(model, "last_attn", attn[-1])
            else:
                setattr(model, "last_attn", None)
            return out

        lm.forward = MethodType(lm_forward_with_attn, lm)

        self._enabled = True

    @staticmethod
    def _compute_attention_map(attn: torch.Tensor, img_range: tuple[int, int]) -> tuple[np.ndarray, float, float, float] | tuple[None, None, None, None]:
        if not isinstance(attn, torch.Tensor) or attn.ndim != 4 or attn.shape[0] < 1:
            return None, None, None, None
        attn_b = attn[0]
        attn_mean_heads = attn_b.mean(0)  # (query_len, key_len)
        img_start, img_end = img_range
        if img_end > attn_mean_heads.shape[-1]:
            return None, None, None, None

        # PI0.5 では「最後のクエリ」だけでは弱くなるので、全クエリ平均で画像パッチ方向を集計
        img_attn = attn_mean_heads[:, img_start:img_end].mean(0)
        n_patches = img_attn.shape[0]
        if n_patches <= 0:
            return None, None, None, None
        grid_size = int(round(float(n_patches) ** 0.5))
        if grid_size * grid_size != n_patches:
            return None, None, None, None
        # bfloat16 を numpy に渡せない環境があるので float32 に揃える
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
        attn = getattr(model, "last_attn", None)
        ranges = getattr(model, "last_image_patch_ranges", None)
        names = getattr(model, "last_image_patch_names", None)

        if model is None or attn is None or ranges is None or names is None:
            return []

        samples: list[AttentionSample] = []
        for (img_range, cam_key) in zip(ranges, names, strict=False):
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
