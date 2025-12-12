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
# SmolVLA: last(以前の動き) / best_mass / best_peaky / best_mix
_SMOLVLA_MODE = os.getenv("LEROBOT_SMOLVLA_ATTN_MODE", "last").strip().lower()
# best_* モードのときだけ使う. 最後のN回の attention 呼び出しだけ候補にする(大きいほど後段寄り)
_SMOLVLA_TAIL = int(os.getenv("LEROBOT_SMOLVLA_ATTN_TAIL", "24"))

# PI: suffix attention を平均する denoise step 数(1で無効)
_PI_SUFFIX_AVG_STEPS = int(os.getenv("LEROBOT_ATTN_SUFFIX_AVG_STEPS", "4"))
# PI: query の window (1ならピッタリ1行)
_PI_Q_WINDOW = max(1, int(os.getenv("LEROBOT_ATTN_Q_WINDOW", "1")))

# PI: pi0 は step 表示を切って平均表示にする(デフォルト1)
_PI0_DISABLE_STEP = os.getenv("LEROBOT_PI0_DISABLE_STEP", "1").lower() not in ("0", "false", "no", "off", "")


def resolve_attention_context(
    policy: PreTrainedPolicy,
) -> "SmolVlaAttentionContext | Pi0xAttentionContext | None":
    name = getattr(policy, "name", None)
    name_s = str(name).lower() if isinstance(name, str) else ""

    if name_s == "smolvla":
        ctx = SmolVlaAttentionContext(policy)
        ctx.enable()
        return ctx

    # pi0 / pi0.5 / pi05 を同じ実装で扱う
    if name_s in ("pi0", "pi05") or name_s.startswith("pi0"):
        ctx = Pi0xAttentionContext(policy)
        ctx.enable()
        return ctx

    # 名前が違っても構造で拾う
    model = getattr(policy, "model", None)
    if model is not None and getattr(model, "paligemma_with_expert", None) is not None:
        if "pi" in name_s or name_s == "":
            ctx = Pi0xAttentionContext(policy)
            ctx.enable()
            return ctx

    return None


# -----------------------------
# SmolVLA
# -----------------------------
class SmolVlaAttentionContext:
    """
    SmolVLA 用のアテンション保存＋可視化。

    重要:
    - SmolVLA の画像トークンは 64 (=8x8) になりがちで, pi05(16x16)より粗いのは仕様。
    - 今回の不一致は「best_mass で早い段の attention を拾う」ことで起きやすい。
      デフォルトは "last" で以前の挙動に戻す。
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

        # attention を保存するために attention forward をラップ
        vlm.save_attn = True
        vlm.last_attn = None

        mode = _SMOLVLA_MODE

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

            query_states_f = query_states.to(dtype=torch.float32)
            key_states_f = key_states_expanded.to(dtype=torch.float32)

            query_states_f = query_states_f.transpose(1, 2)
            key_states_f = key_states_f.transpose(1, 2)

            att_weights = torch.matmul(query_states_f, key_states_f.transpose(2, 3))
            att_weights *= head_dim**-0.5

            big_neg = torch.finfo(att_weights.dtype).min
            masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
            probs = torch.nn.functional.softmax(masked_att_weights, dim=-1)
            probs = probs.to(dtype=value_states_expanded.dtype)

            if getattr(self, "save_attn", False):
                # 以前と同じ: 「最後の attention」を保存
                self.last_attn = probs.detach()

                # best_* モードだけ候補収集
                if mode != "last":
                    ranges = getattr(self, "_attn_vis_img_patch_ranges", None)
                    names = getattr(self, "_attn_vis_img_patch_names", None)
                    q_idx = getattr(self, "_attn_vis_query_indices", None)

                    if isinstance(ranges, list) and isinstance(names, list) and ranges and names:
                        if not isinstance(q_idx, list) or len(q_idx) == 0:
                            q_idx = [int(probs.shape[2] - 1)]
                        q_idx = [int(i) for i in q_idx if 0 <= int(i) < int(probs.shape[2])]
                        if len(q_idx) == 0:
                            q_idx = [int(probs.shape[2] - 1)]

                        # (q,k) を CPU に落として軽量に評価
                        p = probs[0].to(dtype=torch.float32).mean(dim=0).detach().cpu()  # (q,k)

                        total_mass = 0.0
                        total_peaky = 0.0
                        slices: dict[str, torch.Tensor] = {}

                        for cam_key, (s, e) in zip(names, ranges, strict=False):
                            s_i = int(s)
                            e_i = int(e)
                            if not (0 <= s_i < e_i <= int(p.shape[1])):
                                continue

                            q_to_img = p[q_idx, s_i:e_i]  # (nq, n_patches)
                            slices[cam_key] = q_to_img

                            mass_q = q_to_img.sum(dim=-1)  # (nq,)
                            mass = float(mass_q.mean().item())
                            total_mass += mass

                            denom = mass_q.clamp_min(1e-9)[:, None]
                            q_norm = q_to_img / denom
                            w = mass_q.clamp_min(0)
                            if float(w.sum().item()) > 0:
                                w = w / w.sum()
                                cond_vec = (q_norm * w[:, None]).sum(dim=0)
                            else:
                                cond_vec = q_norm.mean(dim=0)

                            peaky = float(cond_vec.max().item())
                            total_peaky += peaky

                        if len(slices) > 0:
                            cand = {
                                "mass": float(total_mass),
                                "peaky": float(total_peaky),
                                "mix": float(total_mass * total_peaky),
                                "slices": slices,
                            }
                            cand_list = getattr(self, "_attn_vis_candidates", None)
                            if not isinstance(cand_list, list):
                                cand_list = []
                            cand_list.append(cand)
                            # 多少多くても問題ないが, 念のため制限
                            if len(cand_list) > 256:
                                cand_list = cand_list[-256:]
                            setattr(self, "_attn_vis_candidates", cand_list)

            att_output = torch.matmul(probs, value_states_expanded.permute(0, 2, 1, 3))
            att_output = att_output.permute(0, 2, 1, 3)
            att_output = att_output.reshape(
                batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim
            )
            return att_output

        vlm.eager_attention_forward = MethodType(eager_attention_forward_with_save, vlm)

        # 画像キーを把握するため prepare_images をラップ
        original_prepare_images = self.policy.prepare_images

        def prepare_images_with_names(
            self, batch: dict[str, Any]
        ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
            images, masks = original_prepare_images(batch)
            present_img_keys = [key for key in self.config.image_features if key in batch]
            setattr(self.model, "_attn_present_img_keys", present_img_keys[: len(images)])

            # processed HW を覚えておく(可視化の参考)
            hw_by_cam: dict[str, tuple[int, int]] = {}
            for cam_key, img in zip(getattr(self.model, "_attn_present_img_keys", []), images, strict=False):
                if isinstance(img, torch.Tensor):
                    if img.ndim == 4:
                        hw_by_cam[str(cam_key)] = (int(img.shape[-2]), int(img.shape[-1]))
                    elif img.ndim == 3:
                        hw_by_cam[str(cam_key)] = (int(img.shape[-2]), int(img.shape[-1]))
            setattr(self.model, "_attn_vis_processed_hw_by_cam", hw_by_cam)

            return images, masks

        self.policy.prepare_images = MethodType(prepare_images_with_names, self.policy)

        # embed_prefix を差し替えてパッチ範囲を記録 + query indices を vlm に渡す
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

            # forward 1回ごとに候補とログ状態をリセット
            vlm_local = getattr(self, "vlm_with_expert", None)
            if vlm_local is not None:
                call_id = int(getattr(vlm_local, "_attn_vis_call_id", 0) or 0) + 1
                setattr(vlm_local, "_attn_vis_call_id", call_id)
                setattr(vlm_local, "_attn_vis_candidates", [])
                setattr(vlm_local, "_attn_vis_logged_call_id", -1)

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
            self.last_image_patch_names = present_img_keys if img_patch_ranges else None

            # vlm に可視化情報を渡す
            if vlm_local is not None:
                q_idx = torch.nonzero(att_masks_tensor[0], as_tuple=False).squeeze(-1).tolist()
                if not isinstance(q_idx, list) or len(q_idx) == 0:
                    q_idx = [int(att_masks_tensor.shape[1] - 1)]
                setattr(vlm_local, "_attn_vis_query_indices", [int(x) for x in q_idx])

                names_for_vlm = present_img_keys[: len(img_patch_ranges)] if img_patch_ranges else []
                setattr(vlm_local, "_attn_vis_img_patch_ranges", list(img_patch_ranges))
                setattr(vlm_local, "_attn_vis_img_patch_names", list(names_for_vlm))

                if _ATTN_DEBUG:
                    call_id = int(getattr(vlm_local, "_attn_vis_call_id", 0) or 0)
                    logging.info(
                        "[attn][smolvla] call_id=%d q_idx_len=%d n_imgs=%d mode=%s",
                        call_id,
                        int(len(q_idx)),
                        int(len(img_patch_ranges)),
                        _SMOLVLA_MODE,
                    )

            return embs, pad_masks, att_masks_tensor

        model.embed_prefix = MethodType(embed_prefix_with_ranges, model)

        self._enabled = True

    @staticmethod
    def _compute_maps_from_q_to_img(
        q_to_img: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float, float] | tuple[None, None, None, None, None, None]:
        """
        q_to_img: (nq, n_patches)  (CPU float32 推奨)
        return:
          vis_grid_01, raw_grid, raw_max, raw_mean, raw_sum, img_mass
        """
        if not isinstance(q_to_img, torch.Tensor) or q_to_img.ndim != 2:
            return None, None, None, None, None, None

        q_to_img = q_to_img.to(dtype=torch.float32)

        raw_vec = q_to_img.mean(dim=0)  # (n_patches,)
        img_mass = float(raw_vec.sum().item())

        n_patches = int(raw_vec.shape[0])
        if n_patches <= 0:
            return None, None, None, None, None, None

        grid_size = int(round(float(n_patches) ** 0.5))
        if grid_size * grid_size != n_patches:
            return None, None, None, None, None, None

        raw_max = float(raw_vec.max().item())
        raw_mean = float(raw_vec.mean().item())
        raw_sum = float(raw_vec.sum().item())
        raw_grid = raw_vec.detach().cpu().numpy().reshape(grid_size, grid_size)

        # 画像パッチ内で再正規化して query を集約
        denom = q_to_img.sum(dim=-1, keepdim=True).clamp_min(1e-9)  # (nq,1)
        q_norm = q_to_img / denom
        w = denom.squeeze(-1).clamp_min(0)  # (nq,)
        if float(w.sum().item()) > 0:
            w = w / w.sum()
            cond_vec = (q_norm * w[:, None]).sum(dim=0)
        else:
            cond_vec = q_norm.mean(dim=0)

        cond_grid = cond_vec.detach().cpu().numpy().reshape(grid_size, grid_size).astype(np.float32)
        cond_grid = cond_grid - float(cond_grid.min())
        vmax = float(cond_grid.max())
        if vmax > 0:
            cond_grid = cond_grid / vmax

        return cond_grid, raw_grid, raw_max, raw_mean, raw_sum, img_mass

    @staticmethod
    def _render_overlay(img_bgr: np.ndarray, attn_map_01_fullres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = img_bgr.shape[:2]
        attn_resized = cv2.resize(attn_map_01_fullres, (w, h), interpolation=cv2.INTER_LINEAR)
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

        ranges = getattr(model, "last_image_patch_ranges", None)
        names = getattr(model, "last_image_patch_names", None)
        if ranges is None or names is None:
            return []

        call_id = int(getattr(vlm, "_attn_vis_call_id", 0) or 0)

        # 1) best_* モードなら候補から選ぶ
        q_to_img_by_cam: dict[str, torch.Tensor] | None = None
        if _SMOLVLA_MODE != "last":
            cand_list = getattr(vlm, "_attn_vis_candidates", None)
            if isinstance(cand_list, list) and len(cand_list) > 0:
                tail = max(1, int(_SMOLVLA_TAIL))
                view = cand_list[-tail:] if tail > 0 else cand_list
                key = "mass"
                if _SMOLVLA_MODE == "best_peaky":
                    key = "peaky"
                elif _SMOLVLA_MODE == "best_mix":
                    key = "mix"
                best = max(view, key=lambda c: float(c.get(key, -1.0)))
                q_to_img_by_cam = best.get("slices", None)

        # 2) last モード(デフォルト)は last_attn から計算
        attn = getattr(vlm, "last_attn", None)
        q_idx = getattr(vlm, "_attn_vis_query_indices", None)
        if not isinstance(q_idx, list) or len(q_idx) == 0:
            q_idx = None

        samples: list[AttentionSample] = []

        for cam_key, img_range in zip(names, ranges, strict=False):
            img_bgr = images_bgr.get(cam_key)
            if img_bgr is None:
                continue

            # q_to_img を作る
            q_to_img = None
            if isinstance(q_to_img_by_cam, dict):
                q_to_img = q_to_img_by_cam.get(cam_key)
            if q_to_img is None:
                if not isinstance(attn, torch.Tensor) or attn.ndim != 4 or attn.shape[0] < 1:
                    continue
                a = attn[0].to(dtype=torch.float32).mean(dim=0)  # (q,k)
                q_len, k_len = a.shape
                s, e = int(img_range[0]), int(img_range[1])
                if not (0 <= s < e <= k_len):
                    continue
                if q_idx is None:
                    q_use = [q_len - 1]
                else:
                    q_use = [int(i) for i in q_idx if 0 <= int(i) < q_len]
                    if len(q_use) == 0:
                        q_use = [q_len - 1]
                q_to_img = a[q_use, s:e].detach().cpu()

            vis_grid, raw_grid, raw_max, raw_mean, raw_sum, img_mass = self._compute_maps_from_q_to_img(q_to_img)
            if vis_grid is None or raw_grid is None:
                continue

            # overlay 用に fullres map を作る(単純拡大)
            h, w = img_bgr.shape[:2]
            fullres = cv2.resize(vis_grid, (w, h), interpolation=cv2.INTER_LINEAR)
            overlay, attn_resized = self._render_overlay(img_bgr, fullres)

            samples.append(
                AttentionSample(
                    camera_key=cam_key,
                    overlay_bgr=overlay,
                    attention_resized=attn_resized,
                    attention_patches=vis_grid,          # 低解像度(8x8など)を保存
                    attention_raw_patches=raw_grid,      # 低解像度の生値
                    original_bgr=img_bgr,
                    raw_max=float(raw_max or 0.0),
                    raw_mean=float(raw_mean or 0.0),
                    raw_sum=float(raw_sum or 0.0),
                )
            )

        # デバッグログは call_id ごとに1回だけ
        if _ATTN_DEBUG and call_id > 0 and len(samples) > 0:
            last_logged = int(getattr(vlm, "_attn_vis_logged_call_id", -1))
            if call_id != last_logged:
                setattr(vlm, "_attn_vis_logged_call_id", call_id)
                for s in samples:
                    arg = np.unravel_index(int(np.argmax(s.attention_patches)), s.attention_patches.shape)
                    logging.info(
                        "[attn][smolvla] call_id=%d cam=%s img_range=%s raw_sum=%.6f argmax(rc)=%s",
                        call_id,
                        s.camera_key,
                        str(dict(zip(names, ranges)).get(s.camera_key, None)),
                        float(s.raw_sum),
                        str(arg),
                    )

        return samples


# -----------------------------
# PI0 / PI0.5
# -----------------------------
class Pi0xAttentionContext:
    """
    PI0 / PI0.5 用のアテンション保存＋可視化。

    pi05 は step=t の query を可視化(今の良い挙動を維持)。
    pi0 はデフォルトで step を追わず, exec_horizon 分の query を平均して固定表示(フラッシュ回避)。
    """

    def __init__(self, policy: PreTrainedPolicy):
        self.policy = policy
        self._enabled = False

        name = str(getattr(policy, "name", "pi0x")).lower()
        self._tag = name

        # pi05 は step 表示, pi0 は平均表示
        self._step_mode = True
        if name == "pi0" and _PI0_DISABLE_STEP:
            self._step_mode = False

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
            model._attn_present_img_keys = present_img_keys
            return images, masks

        self.policy._preprocess_images = MethodType(preprocess_images_with_names, self.policy)

        # embed_prefix をラップして画像パッチ範囲を記録(あなたの pi05 実装と同じ方針)
        original_embed_prefix = model.embed_prefix

        def embed_prefix_with_ranges(
            self_model, images: Iterable[torch.Tensor], img_masks: Iterable[torch.Tensor], tokens, masks
        ):
            embs = []
            pad_masks = []
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

            embs = torch.cat(embs, dim=1)
            pad_masks = torch.cat(pad_masks, dim=1)
            att_masks_tensor = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
            bsize = pad_masks.shape[0]
            att_masks_tensor = att_masks_tensor[None, :].expand(bsize, len(att_masks))

            self_model.last_image_patch_ranges = patch_ranges
            self_model.last_image_patch_names = patch_names
            self_model._attn_vis_chunk_id = int(getattr(self_model, "_attn_vis_chunk_id", 0) or 0) + 1

            return embs, pad_masks, att_masks_tensor

        model.embed_prefix = MethodType(embed_prefix_with_ranges, model)

        # language_model forward をラップして prefix attentions を保存
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

        # gemma_expert 側 forward をラップして suffix attentions を保存 + best layer 選択
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

            def _exec_horizon() -> int:
                rtc = getattr(self.policy.config, "rtc_config", None)
                if rtc is not None and getattr(rtc, "execution_horizon", None) is not None:
                    return int(getattr(rtc, "execution_horizon"))
                return int(getattr(self.policy.config, "n_action_steps", 1) or 1)

            def _score_layer(att: torch.Tensor) -> float:
                ranges = getattr(model, "last_image_patch_ranges", None)
                if not ranges:
                    return -1.0
                img0 = min(int(s) for s, _ in ranges)
                img1 = max(int(e) for _, e in ranges)

                if att.ndim != 4 or att.shape[0] < 1:
                    return -1.0
                m = att[0].to(dtype=torch.float32).mean(0)  # (q,k)
                q_len, k_len = m.shape
                if img1 > k_len:
                    return -1.0

                q1 = min(q_len, _exec_horizon())
                return float(m[:q1, img0:img1].sum(dim=-1).mean().item())

            def gemma_forward_with_attn(self_gemma, *args, **kwargs):
                kwargs["output_attentions"] = True
                out = original_gemma_forward(*args, **kwargs)
                attn_list = getattr(out, "attentions", None)

                if attn_list is None:
                    setattr(model, "last_attn_suffix", None)
                    return out

                best_i = int(len(attn_list) - 1)
                best_s = -1.0
                for i, a in enumerate(attn_list):
                    s = _score_layer(a)
                    if s > best_s:
                        best_s = s
                        best_i = int(i)

                picked = attn_list[best_i].detach()
                setattr(model, "last_attn_suffix", picked)

                # denoise step 平均用にキューへ
                steps = getattr(model, "_attn_suffix_steps", None)
                if not isinstance(steps, list):
                    steps = []
                steps.append(picked)
                if len(steps) > 64:
                    steps = steps[-64:]
                setattr(model, "_attn_suffix_steps", steps)

                if _ATTN_DEBUG:
                    chunk_id = int(getattr(model, "_attn_vis_chunk_id", 0) or 0)
                    last_logged = int(getattr(model, "_attn_best_layer_logged_chunk_id", -1))
                    if chunk_id != last_logged:
                        setattr(model, "_attn_best_layer_logged_chunk_id", chunk_id)
                        logging.info(
                            "[attn][%s] chunk_id=%d best_layer=%d best_layer_mass=%.6f step_mode=%s",
                            self._tag,
                            chunk_id,
                            best_i,
                            float(best_s),
                            "yes" if self._step_mode else "no",
                        )

                return out

            gemma_model.forward = MethodType(gemma_forward_with_attn, gemma_model)

        self._enabled = True

    @staticmethod
    def _compute_attention_grid(
        attn: torch.Tensor,
        img_range: tuple[int, int],
        *,
        q_index: int | None,
        q_limit: int | None,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float, dict[str, float]] | tuple[None, None, None, None, None, None]:
        if not isinstance(attn, torch.Tensor) or attn.ndim != 4 or attn.shape[0] < 1:
            return None, None, None, None, None, None

        m = attn[0].to(dtype=torch.float32).mean(0)  # (q,k)
        q_len, k_len = m.shape
        s, e = int(img_range[0]), int(img_range[1])
        if not (0 <= s < e <= k_len):
            return None, None, None, None, None, None

        if q_index is None:
            q0 = 0
            q1 = min(q_len, int(q_limit)) if q_limit is not None else q_len
        else:
            qi = int(max(0, min(q_len - 1, int(q_index))))
            win = int(_PI_Q_WINDOW)
            half = win // 2
            q0 = max(0, qi - half)
            q1 = min(q_len, qi + half + 1)

        q_to_img = m[q0:q1, s:e]  # (nq, n_patches)
        if q_to_img.numel() == 0:
            return None, None, None, None, None, None

        # raw(= key 全体 softmax のまま) の統計
        raw_vec = q_to_img.mean(0)
        raw_max = float(raw_vec.max().item())
        raw_mean = float(raw_vec.mean().item())
        raw_sum = float(raw_vec.sum().item())

        n_patches = int(raw_vec.shape[0])
        grid_size = int(round(float(n_patches) ** 0.5))
        if grid_size * grid_size != n_patches:
            return None, None, None, None, None, None

        raw_grid = raw_vec.detach().cpu().numpy().reshape(grid_size, grid_size)

        # conditioned(= 画像パッチ内で再正規化)
        denom = q_to_img.sum(dim=-1).clamp_min(1e-9)  # (nq,)
        q_norm = q_to_img / denom[:, None]
        w = denom.clamp_min(0)
        if float(w.sum().item()) > 0:
            w = w / w.sum()
            cond_vec = (q_norm * w[:, None]).sum(0)
        else:
            cond_vec = q_norm.mean(0)

        cond_grid = cond_vec.detach().cpu().numpy().reshape(grid_size, grid_size).astype(np.float32)
        cond_grid = cond_grid - float(cond_grid.min())
        vmax = float(cond_grid.max())
        if vmax > 0:
            cond_grid = cond_grid / vmax

        arg = int(np.argmax(cond_grid))
        diag = {
            "q_len": float(q_len),
            "k_len": float(k_len),
            "q0": float(q0),
            "q1": float(q1),
            "argmax_r": float(arg // grid_size),
            "argmax_c": float(arg % grid_size),
        }
        return cond_grid, raw_grid, raw_max, raw_mean, raw_sum, diag

    @staticmethod
    def _render_overlay(img_bgr: np.ndarray, attn_grid_01: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = img_bgr.shape[:2]
        attn_resized = cv2.resize(attn_grid_01, (w, h), interpolation=cv2.INTER_LINEAR)
        attn_uint8 = (attn_resized * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(attn_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_bgr, 0.5, heatmap, 0.5, 0.0)
        return overlay, attn_resized

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

        # suffix があれば優先, なければ prefix の最後
        steps = getattr(model, "_attn_suffix_steps", None)
        attn = None
        if isinstance(steps, list) and len(steps) > 0 and _PI_SUFFIX_AVG_STEPS > 1:
            tail = steps[-max(1, _PI_SUFFIX_AVG_STEPS) :]
            try:
                attn = torch.stack(tail, dim=0).mean(0)
            except Exception:
                attn = tail[-1]
        if attn is None:
            attn = getattr(model, "last_attn_suffix", None)
        if attn is None:
            attn_list = getattr(model, "last_attn_prefix", None)
            if isinstance(attn_list, (list, tuple)) and len(attn_list) > 0:
                attn = attn_list[-1]
        if attn is None:
            return []

        chunk_id = int(getattr(model, "_attn_vis_chunk_id", 0) or 0)
        if self._last_seen_chunk_id is None or chunk_id != self._last_seen_chunk_id:
            self._last_seen_chunk_id = chunk_id
            self._vis_step_in_chunk = 0

        chunk_size = int(getattr(policy.config, "n_action_steps", 1) or 1)
        rtc = getattr(policy.config, "rtc_config", None)
        exec_h = int(getattr(rtc, "execution_horizon", 0) or 0) if rtc is not None else 0
        if exec_h <= 0:
            exec_h = min(chunk_size, 10)

        if self._step_mode:
            q_index = int(self._vis_step_in_chunk)
            q_limit = None
            self._vis_step_in_chunk = (self._vis_step_in_chunk + 1) % max(1, chunk_size)
        else:
            # pi0: step を追わずに平均表示で固定(フラッシュ回避)
            q_index = None
            q_limit = exec_h

        samples: list[AttentionSample] = []
        for img_range, cam_key in zip(ranges, names, strict=False):
            img_bgr = images_bgr.get(cam_key)
            if img_bgr is None:
                continue

            vis_grid, raw_grid, raw_max, raw_mean, raw_sum, diag = self._compute_attention_grid(
                attn,
                img_range,
                q_index=q_index,
                q_limit=q_limit,
            )
            if vis_grid is None or raw_grid is None:
                continue

            overlay, attn_resized = self._render_overlay(img_bgr, vis_grid)

            if _ATTN_DEBUG and diag is not None:
                logging.info(
                    "[attn][%s] cam=%s chunk_id=%d step=%s q=(%d..%d)/%d k=%d img_range=%s argmax(rc)=(%d,%d)",
                    self._tag,
                    cam_key,
                    chunk_id,
                    str(q_index) if q_index is not None else f"avg0..{q_limit}",
                    int(diag["q0"]),
                    int(diag["q1"]),
                    int(diag["q_len"]),
                    int(diag["k_len"]),
                    str(img_range),
                    int(diag["argmax_r"]),
                    int(diag["argmax_c"]),
                )

            samples.append(
                AttentionSample(
                    camera_key=cam_key,
                    overlay_bgr=overlay,
                    attention_resized=attn_resized,
                    attention_patches=vis_grid,      # 16x16 などの低解像度を保存
                    attention_raw_patches=raw_grid,  # 生の低解像度
                    original_bgr=img_bgr,
                    raw_max=float(raw_max or 0.0),
                    raw_mean=float(raw_mean or 0.0),
                    raw_sum=float(raw_sum or 0.0),
                )
            )

        return samples
