# Plan: Implement XAI Methods for X-VLA (feat/x-ai)

## Context

`docs/xai.md` đặc tả 7 phương pháp XAI (Explainable AI) cho hệ thống robot X-VLA/VLAR với SO-ARM-101. Mục tiêu: hiện thực hóa toàn bộ 7 phương pháp + XAI buffer/pipeline vào branch `feat/x-ai`, không phá vỡ code hiện tại, overhead = 0 khi không bật.

---

## Architecture Clarifications

### DaViT ≠ SigLIP

`docs/xai.md` giả định SigLIP 14×14 = 196 patch tokens. Thực tế:
- **Vision tower:** `DaViT` (Dense Vision Transformer) 4-stage hierarchical — **không phải SigLIP**
- `tokens_per_view` phụ thuộc vào resolution, không hardcode là 196
- **Florence2 encoder:** T5-style (`Florence2Encoder`) với `Florence2Attention` — hook qua `output_attentions=True`

### Confirmed Hook Paths

```
model.model.vlm.vision_tower                          # DaViT
model.model.vlm.language_model.model.encoder.layers  # Florence2EncoderLayer list
  [i].self_attn  →  forward(..., output_attentions=True)
                    returns (attn_out, attn_weights[B,H,seq,seq], past_kv)
model.model.transformer                               # SoftPromptedTransformer
model.model.action_space
model.model.rtc_processor                             # RTCProcessor | None
```

⚠️ `Florence2SdpaAttention` trả về `None` cho `attn_weights`. Cần context manager `_force_eager_attention(model)` khi chạy XAI hoặc log warning + fallback.

---

## Module Structure (tất cả là file mới)

```
src/lerobot/xai/
├── __init__.py                        # Public API
├── config.py                          # XAIConfig dataclass (enable flags, thresholds)
├── buffer.py                          # StepRecord, EpisodeXAIBuffer
├── pipeline.py                        # XAIPipeline (orchestrator)
│
├── vision/
│   ├── attention_hook.py              # P0-V: AttentionHook, attention_entropy()
│   └── gmar.py                        # P1-V: GMARAnalyzer.compute()
│
├── action/
│   ├── denoising_tracker.py           # P1-A: DenoisingTracker
│   ├── sample_bundle.py               # P2-A: ActionSampleBundle
│   └── correlation.py                 # P3-A: ActionCorrelationAnalyzer
│
├── cross_modal/
│   └── integrated_gradients.py        # P2-X: IntegratedGradientsXVLA
│
├── rtc/
│   └── boundary_smoothness.py         # P3-RTC: BoundarySmoothnessMonitor
│
├── visualization/
│   ├── attention_viz.py               # overlay heatmap lên camera frame
│   ├── action_viz.py                  # plot denoising trajectory, action bundle
│   └── rerun_logger.py                # XAI namespace cho Rerun SDK
│
└── scripts/
    ├── run_gmar.py                    # CLI: P1-V offline
    ├── run_sample_bundle.py           # CLI: P2-A offline
    ├── run_ig.py                      # CLI: P2-X offline
    ├── run_correlation.py             # CLI: P3-A offline
    └── probe_architecture.py          # Dev utility: verify hook paths

tests/xai/
├── conftest.py                        # tiny_xvla_config, tiny_batch fixtures
├── test_attention_hook.py
├── test_gmar.py
├── test_denoising_tracker.py
├── test_sample_bundle.py
├── test_integrated_gradients.py
├── test_correlation.py
├── test_boundary_smoothness.py
├── test_buffer.py
└── test_pipeline.py
```

---

## Changes to Existing Files (chỉ 2 file)

### 1. `src/lerobot/policies/xvla/modeling_xvla.py`

**Change A — cache `tokens_per_view` trong `forward_vlm()`:**
```python
# Sau dòng: tokens_per_view, hidden_dim = valid_feats.shape[1:]
self._last_tokens_per_view = tokens_per_view  # XAI hook
```

**Change B — optional tracker trong `generate_actions()`:**
```python
def generate_actions(self, ..., xai_denoising_tracker=None):
    ...
    # Trong denoising loop, sau mỗi update x_t:
    if xai_denoising_tracker is not None:
        xai_denoising_tracker.track(x_t)
```

### 2. `src/lerobot/scripts/lerobot_record.py`

Thêm optional `xai_config: XAIConfig | None = None` vào record config, wrap episode loop:
```python
if xai_pipeline: xai_pipeline.start_episode(episode_id)
# ... trong step loop ...
if xai_pipeline: xai_pipeline.step(batch, action_chunk, step_idx)
# ... cuối episode ...
if xai_pipeline:
    ep_buf = xai_pipeline.end_episode()
    ep_buf.save(output_dir / f"episode_{ep_idx}.json")
```

---

## Phase Breakdown

### Phase 0 — Foundation (1 ngày)
- Tạo skeleton module + subdirs
- `XAIConfig` dataclass (tất cả flags mặc định `False`)
- `tests/xai/conftest.py` với `tiny_xvla_config` và `tiny_batch` fixtures

### Phase 1 — Real-time Methods (2–3 ngày)

**P3-RTC `BoundarySmoothnessMonitor`** (đơn giản nhất, làm trước):
- `update(chunk) → float | None` — cosine_similarity giữa `prev_tail` và `new_head`
- `get_status(sim) → "ok" | "warning_jerk" | "critical_jerk"`
- `episode_quality() → float` — tỷ lệ transitions smooth
- Không cần model access

**P0-V `AttentionHook`**:
- `register(model, layer_indices=[-1,-3,-6])` — hook vào `encoder.layers[i].self_attn`
- Capture `output[1]` (attn_weights) — phát hiện và warn khi SDPA trả về None
- `get_heatmap(B_idx, num_img_tokens)` — `num_img_tokens` từ `model.model._last_tokens_per_view`
- `attention_entropy(attn_1d) → float`

**P1-A `DenoisingTracker`**:
- `track(x_t)`, `get_convergence_speed()`, `get_final_std()`, `clear()`
- Tích hợp vào `generate_actions()` qua optional kwarg
- Với RTC path: đọc từ `rtc_processor.get_all_debug_steps()` nếu RTC debug bật
- **Không** extend `rtc.debug_tracker.Tracker` — giữ tách biệt

### Phase 2 — XAI Buffer & Pipeline (1 ngày)

**`StepRecord`** dataclass: `step_idx, timestamp, attn_entropy, attn_compressed[7,7], convergence_speed, boundary_sim, flagged, flag_reason`

**`EpisodeXAIBuffer`**: `add_step()`, `should_run_offline_xai()` (>10% flagged), `summary()`, `save()/load()`

**`XAIPipeline`**:
```python
class XAIPipeline:
    def __init__(self, model: XVLAPolicy, config: XAIConfig)
    def start_episode(self, episode_id: str)
    def step(self, batch, action_chunk, step_idx) -> StepRecord
    def end_episode(self) -> EpisodeXAIBuffer
    def should_include_in_training(self, ep: EpisodeXAIBuffer) -> bool
```
Zero-overhead guarantee: sub-components là `None` khi flag = `False`.

### Phase 3 — Offline Methods (5–7 ngày)

**P1-V `GMARAnalyzer`**:
- Dùng `encoder.forward(..., output_attentions=True)` trực tiếp (không hook) để gradient flow qua
- Context manager `_force_eager_attention(model)` khi cần
- `compute(model, batch, target_action_dim, layer_range, n_img_tokens) → Tensor[B,H,W]`
- `_rollout(attention_maps, attention_grads, seq_len, B)`

**P2-A `ActionSampleBundle`**:
- Cache VLM encoding một lần, chạy `transformer(...)` N lần với noise khác nhau
- `compute(model, batch, n_samples=50) → {samples, mean, std, cv, is_multimodal, uncertainty_score}`
- `detect_multimodal(samples, n_clusters=2)` — sklearn GMM với import guard

**P2-X `IntegratedGradientsXVLA`**:
- Baseline: black image + zero proprio
- Interpolate `vlm_features` và `proprio` trong loop n_steps lần
- `compute(model, batch, n_steps=50) → {vision_pct, language_pct, proprio_pct, ig_vlm, ig_proprio}`
- `check_attribution_health(result) → list[str]`
- `num_img_tokens` từ `model.model._last_tokens_per_view` (không hardcode)

**P3-A `ActionCorrelationAnalyzer`**:
- Xây dựng trên `ActionSampleBundle`
- `compute(model, observations, n_samples=100) → {correlation, covariance, std_per_dim}`
- `detect_spurious_correlations(corr_matrix, kinematic_pairs, threshold=0.7)`

### Phase 4 — Visualization & Integration (2 ngày)
- `overlay_heatmap_on_image()`, `plot_denoising_trajectory()`, `plot_action_bundle()`
- `XAIRerunLogger` extend `log_rerun_data()` pattern
- Tích hợp `XAIPipeline` vào `lerobot_record.py` và `lerobot_eval.py`

---

## Key Design Decisions

| Quyết định | Lựa chọn | Lý do |
|------------|----------|-------|
| Vị trí module | `src/lerobot/xai/` standalone | Cross-cutting, testable độc lập |
| P0-V capture | `register_forward_hook` | Lightweight real-time, no API change |
| P1-V capture | Direct `output_attentions=True` | Gradient flow qua hook không đáng tin |
| DenoisingTracker | Class mới, không extend RTC Tracker | RTC Tracker chỉ active khi RTC debug=True |
| `num_img_tokens` | Dynamic từ `_last_tokens_per_view` | DaViT không guarantee 196 tokens |
| Optional deps | Import guard cho matplotlib/seaborn/sklearn | Không thêm required deps |

---

## Optional Dependencies

Thêm vào `pyproject.toml` extras:
```toml
[project.optional-dependencies]
xai = ["matplotlib", "seaborn", "scikit-learn"]
```

---

## Test Strategy

- `conftest.py`: `tiny_xvla_config` (2 encoder layers, d_model=64), `tiny_batch` — không cần weights thật
- Mỗi component: unit tests với mock tensors (không cần forward pass thật)
- Integration tests: `test_pipeline.py` dùng tiny model để verify lifecycle end-to-end
- Zero-overhead test: pipeline với tất cả flags=False, 100 step calls < 1ms/call

---

## Timeline

| Tuần | Nội dung | Effort |
|------|----------|--------|
| Tuần 1 | Phase 0-2: skeleton + real-time methods + buffer/pipeline | 5–6 ngày |
| Tuần 2 | Phase 3: GMAR + Sample Bundle + Integrated Gradients | 4–5 ngày |
| Tuần 3 | Phase 3d + Phase 4: Correlation + Viz + Integration | 3–4 ngày |

**Tổng: ~10–12 developer-days**

---

## Verification

1. `pytest tests/xai/ -v` — tất cả unit tests pass
2. `python src/lerobot/xai/scripts/probe_architecture.py --config <config_path>` — verify hook paths với model thật
3. Chạy `lerobot_record.py` với `xai_config` bật P3-RTC + P0-V, verify `EpisodeXAIBuffer` được lưu sau mỗi episode
4. Chạy `run_gmar.py` trên episode buffer có flagged steps, verify heatmap output
5. So sánh `episode_quality()` score với quan sát thực tế từ teleoperation

---

## Critical Files

| File | Thay đổi |
|------|---------|
| `src/lerobot/policies/xvla/modeling_xvla.py` | +cache `_last_tokens_per_view`, +`xai_denoising_tracker` kwarg |
| `src/lerobot/policies/xvla/modeling_florence2.py` | **Chỉ đọc** — phải hiểu `Florence2Attention.forward()` return signature |
| `src/lerobot/policies/rtc/debug_tracker.py` | **Chỉ đọc** — tham khảo pattern |
| `src/lerobot/scripts/lerobot_record.py` | +optional XAIPipeline lifecycle |
| `src/lerobot/scripts/lerobot_eval.py` | +optional XAIPipeline lifecycle (priority thấp hơn) |
