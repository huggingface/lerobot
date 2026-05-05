**X-VLA / VLAR Platform**

**Explainable AI (XAI) Methods**

Research Report & Implementation Guide

_SO-ARM-101 | Wrist Cam + Global Cam | Florence-2 + Flow Matching_

April 2026

Table of Contents

[0\. Tổng quan & Phạm vi Báo cáo 2](#_Toc351185920)

[Bảng tổng hợp 7 phương pháp theo Priority 2](#_Toc642937891)

[1\. Bối cảnh Kiến trúc X-VLA 3](#_Toc114391383)

[1.1 Luồng dữ liệu Forward Pass 4](#_Toc2006459529)

[2\. P0-V: Raw Attention Map 4](#_Toc1564020429)

[2.1 Cơ sở lý thuyết 5](#_Toc1910576357)

[2.2 Cách triển khai 5](#_Toc1725150422)

[2.3 Thời điểm thực hiện 6](#_Toc1551431055)

[2.4 Ứng dụng ngược lại vào Data Collection 6](#_Toc904708576)

[2.4.1 Điều chỉnh vị trí Global Cam 6](#_Toc690189986)

[2.4.2 Phát hiện mâu thuẫn giữa Wrist Cam và Global Cam 6](#_Toc1310214673)

[2.4.3 Phát hiện Attention Collapse 6](#_Toc81144494)

[3\. P1-V: GMAR - Gradient-weighted Multi-head Attention Rollout 7](#_Toc1914958117)

[3.1 Cơ sở lý thuyết 8](#_Toc1641402776)

[3.2 Cách triển khai 8](#_Toc135100909)

[3.3 Thời điểm thực hiện 9](#_Toc129522819)

[3.4 Ứng dụng ngược lại vào Data Collection 9](#_Toc346799282)

[3.4.1 Language Instruction Sensitivity Test 9](#_Toc1562926133)

[3.4.2 Contrastive Demo Strategy 10](#_Toc826662205)

[3.4.3 Task-phase Attention Audit 10](#_Toc870236858)

[4\. P1-A: Denoising Trajectory Visualization 10](#_Toc1204463755)

[4.1 Cơ sở lý thuyết 11](#_Toc490197754)

[4.2 Cách triển khai 11](#_Toc1573967803)

[4.3 Thời điểm thực hiện 12](#_Toc1553119081)

[4.4 Ứng dụng ngược lại vào Data Collection 12](#_Toc1487994185)

[4.4.1 Phát hiện Ambiguous Observations 12](#_Toc37690955)

[4.4.2 Curriculum Learning Signal 12](#_Toc359509)

[5\. P2-A: Action Sample Bundle 12](#_Toc918030698)

[5.1 Cơ sở lý thuyết 13](#_Toc1100607877)

[5.2 Cách triển khai 13](#_Toc904280029)

[5.3 Thời điểm thực hiện 14](#_Toc799762604)

[5.4 Ứng dụng ngược lại vào Data Collection 14](#_Toc1672781655)

[5.4.1 Uncertainty-based Active Data Collection 14](#_Toc2015955122)

[5.4.2 Multimodal Detection → Demo Stratification 14](#_Toc319044635)

[5.4.3 Task Difficulty Mapping 14](#_Toc196127621)

[6\. P2-X: Integrated Gradients - Cross-modal Attribution 14](#_Toc128872274)

[6.1 Cơ sở lý thuyết 15](#_Toc668152810)

[6.2 Cách triển khai 15](#_Toc1950552160)

[6.3 Thời điểm thực hiện 17](#_Toc1185763469)

[6.4 Ứng dụng ngược lại vào Data Collection 17](#_Toc514095605)

[6.4.1 Modality Imbalance Detection 17](#_Toc684701117)

[6.4.2 Wrist Cam vs Global Cam Attribution 17](#_Toc1105845386)

[6.4.3 Proprio Feature Importance 17](#_Toc1620824512)

[7\. P3-A: Action Dimension Correlation Heatmap 17](#_Toc1571261937)

[7.1 Cơ sở lý thuyết 18](#_Toc1739722376)

[7.2 Cách triển khai 18](#_Toc2044846771)

[7.3 Thời điểm thực hiện 19](#_Toc10561378)

[7.4 Ứng dụng ngược lại vào Data Collection 19](#_Toc936195821)

[7.4.1 Phát hiện Spurious Correlation từ Demo 19](#_Toc2028676145)

[7.4.2 Demo Diversity Metric 19](#_Toc850354556)

[8\. P3-RTC: Chunk Boundary Smoothness 19](#_Toc257763849)

[8.1 Cơ sở lý thuyết 20](#_Toc1760739337)

[8.2 Cách triển khai 20](#_Toc2118947623)

[8.3 Thời điểm thực hiện 21](#_Toc1438877159)

[8.4 Ứng dụng ngược lại vào Data Collection 21](#_Toc1555453364)

[8.4.1 Automatic Demo Quality Filter 21](#_Toc2001883346)

[8.4.2 Jerk Source Analysis 21](#_Toc62642933)

[8.4.3 Global Cam Angle Optimization 21](#_Toc1445266679)

[9\. Tích hợp Hệ thống: XAI Buffer & Pipeline 22](#_Toc538630338)

[9.1 XAI Buffer - Cấu trúc dữ liệu 23](#_Toc2030134543)

[9.2 Decision Pipeline: Khi nào chạy gì 23](#_Toc1426972434)

[10\. Tổng kết: Ma trận Quyết định Data Collection 25](#_Toc1809428021)

[11\. Quy trình Làm việc Khuyến nghị 26](#_Toc1442205735)

[11.1 Trước khi thu thập dữ liệu (Setup) 26](#_Toc1733063413)

[11.2 Trong quá trình thu thập dữ liệu (Real-time) 27](#_Toc2063208389)

[11.3 Sau mỗi collection session (Offline) 27](#_Toc749858494)

[11.4 Chu kỳ cải tiến (Iteration) 27](#_Toc503459414)

[12\. Kết luận 27](#_Toc143391576)

[References - Phương pháp XAI cho VLA 28](#_Toc1663391487)

[Vision XAI (Florence-2 / Transformer Encoder) 28](#_Toc821995108)

[Action XAI (Flow Matching) 29](#_Toc429391553)

[Cross-modal Attribution 30](#_Toc802221588)

[Survey / Tổng quan XAI 30](#_Toc1249721329)

[Về X-VLA cụ thể 30](#_Toc1625141884)

# 0\. Tổng quan & Phạm vi Báo cáo

Báo cáo này trình bày đầy đủ **7 phương pháp XAI (Explainable AI)** được thiết kế đặc thù cho kiến trúc X-VLA sử dụng trong nền tảng robot VLAR với SO-ARM-101. Mỗi phương pháp được phân tích từ 4 góc độ: **(1) cơ sở lý thuyết**, **(2) cách triển khai cụ thể** vào code X-VLA hiện có, **(3) thời điểm thực hiện** (on-the-fly / offline / triggered), và **(4) ứng dụng ngược lại vào quá trình thu thập dữ liệu** để cải thiện hiệu suất model.

Hệ thống VLAR sử dụng **Florence-2** làm vision-language encoder (encoder-only, decoder và lm_head bị xóa), backbone **SoftPromptedTransformer** với domain-specific soft prompts, và **Flow Matching** để generate continuous action chunks. Camera setup gồm **wrist cam** (gắn cố định với cổ tay SO-ARM-101) và **global cam** (toàn cảnh, có thể thay đổi góc nhìn).

## Bảng tổng hợp 7 phương pháp theo Priority

| **ID** | **Tên phương pháp**        | **Target**           | **Mục đích chính**                    | **Chi phí compute** | **Thời điểm**       |
| ------ | -------------------------- | -------------------- | ------------------------------------- | ------------------- | ------------------- |
| P0-V   | Raw Attention Map          | Florence-2 Encoder   | Xem nhanh model đang nhìn vào đâu     | Gần bằng 0          | Real-time           |
| P1-V   | GMAR (Grad x Attn Rollout) | Florence-2 Encoder   | Action-conditioned heatmap faithful   | 1.5x forward        | Offline / Triggered |
| P1-A   | Denoising Trajectory Plot  | Flow Matching ODE    | Visualize quá trình commit action     | Gần bằng 0          | Real-time log       |
| P2-A   | Action Sample Bundle       | Flow Matching Output | Uncertainty / multimodality analysis  | Nx forward          | Offline             |
| P2-X   | Integrated Gradients       | Cross-modal          | Vision vs Language attribution        | 50x forward         | Offline             |
| P3-A   | Action Dim Correlation     | Action Space         | Joint coupling / covariance structure | Thấp                | Offline             |
| P3-RTC | Chunk Boundary Smoothness  | RTC Pipeline         | Jerk detection, demo quality filter   | Gần bằng 0          | Real-time           |

# 1\. Bối cảnh Kiến trúc X-VLA

Trước khi đi vào từng phương pháp, cần hiểu rõ luồng dữ liệu trong XVLAModel để biết chính xác hook point nào cần can thiệp.

## 1.1 Luồng dữ liệu Forward Pass

Input: input_ids \[B, seq_len\] + image_input \[B, num_views, C, H, W\]

domain_id \[B\] + proprio \[B, dim_proprio\]

action \[B, chunk_size, dim_action\] (training only)

Step 1: forward_vlm()

SigLIP vision_tower --> \_encode_image() --> image_features \[B, V, P, D\]

get_input_embeddings()(input_ids) --> lang_embeds \[B, L, D\]

\_merge_input_ids_with_image_features() --> merged_embeds \[B, L+P, D\]

language_model.model.encoder() --> vlm_features \[B, L+P, D\]

\+ aux_visual_inputs \[B, (V-1)\*P, D\] (extra camera views)

Step 2: Flow Matching (training)

t ~ Uniform(0,1) with stratified sampling

action_noisy = randn \* t + action \* (1-t) (linear interpolation)

action_space.preprocess(proprio, action_noisy)

transformer(domain_id, action_noisy, t, proprio, vlm_features, aux)

\--> pred_action --> action_space.compute_loss()

Step 3: Inference (generate_actions)

x1 = randn \[B, chunk_size, dim_action\] (pure noise)

for i in \[steps, steps-1, ..., 1\]:

t = i/steps

x_t = x1\*t + action\*(1-t) (denoising interpolation)

pred_action = transformer(x_t, t, enc, ...)

\--> postprocess() --> final actions

**Key insight:** Có 3 hook points quan trọng nhất để XAI: **(A)** attention weights bên trong encoder layers (vision XAI), **(B)** gradient w.r.t. vlm_features (cross-modal attribution), và **(C)** x_t tại mỗi denoising step (action XAI). Cả 3 đều accessible mà không cần thay đổi architecture.

# 2\. P0-V: Raw Attention Map

**Priority: P0 | Target: Florence-2 Encoder | Cost: ~0 | Timing: Real-time**

## 2.1 Cơ sở lý thuyết

Florence-2 dùng kiến trúc encoder transformer. Tại mỗi self-attention layer, model tính ma trận attention A ∈ R^{heads x seq_len x seq_len}. Sequence gồm image patch tokens (từ SigLIP) và language tokens được merge lại. Attention weight A\[h, i, j\] thể hiện mức độ token i tập trung vào token j. Visualize phần attention từ language tokens sang image patch tokens cho thấy model đang "nhìn" vào vùng nào của ảnh khi xử lý instruction.

Nhược điểm đã biết: raw attention không nhất thiết phản ánh importance thực sự về mặt causal, nhưng đủ nhanh để dùng làm **first-pass sanity check** trong quá trình debug và data collection monitoring.

## 2.2 Cách triển khai

Đăng ký forward hook vào encoder layer cuối (hoặc nhiều layers). Hook capture attention weights mà không cần thêm compute:

class AttentionHook:

def \__init_\_(self):

self.attention_maps = \[\]

self.hooks = \[\]

def register(self, model, layer_indices=\[-1, -3, -6\]):

encoder_layers = model.vlm.language_model.model.encoder.layers

for idx in layer_indices:

layer = encoder_layers\[idx\]

hook = layer.self_attn.register_forward_hook(self.\_hook_fn)

self.hooks.append(hook)

def \_hook_fn(self, module, input, output):

\# output: attn_output, attn_weights

\# attn_weights shape: \[B, num_heads, seq_len, seq_len\]

if isinstance(output, tuple) and len(output) > 1:

self.attention_maps.append(output\[1\].detach().cpu())

def get_image_attention(self, num_img_tokens, num_lang_tokens):

\# Lấy attention từ language tokens sang image patch tokens

\# attention_maps\[-1\]: \[B, H, seq, seq\]

attn = self.attention_maps\[-1\] # last layer

\# lang_to_img: language tokens attend vào image patches

lang_to_img = attn\[:, :, num_img_tokens:, :num_img_tokens\]

return lang_to_img.mean(dim=(1, 2)) # avg heads + lang_tokens -> \[B, P\]

def get_heatmap(self, B_idx=0, patch_grid=(14, 14)):

img_attn = self.get_image_attention(...)\[B_idx\] # \[num_patches\]

heatmap = img_attn.reshape(patch_grid) # \[14, 14\]

\# Normalize

heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

return heatmap # upsample x16 để overlay lên ảnh gốc 224px

def clear(self):

self.attention_maps.clear()

def remove(self):

for h in self.hooks: h.remove()

## 2.3 Thời điểm thực hiện

- **Trong data collection (real-time):** bật hook, lưu compressed heatmap (7x7 sau avg_pool2d) + entropy scalar vào XAI buffer mỗi step. Chi phí thêm chỉ là memory bandwidth.
- **Khi debug nhanh:** visualize overlay lên frame camera để xem model nhìn đúng chỗ không, không cần restart.
- **Khi validation sau training:** so sánh heatmap của model mới vs model cũ trên cùng test episodes.

\# Tính attention entropy (real-time monitoring)

def attention_entropy(attn_map_1d):

p = F.softmax(attn_map_1d, dim=-1)

entropy = -(p \* (p + 1e-8).log()).sum().item()

return entropy # cao = model confused, thấp = focused

\# Flag step nếu entropy quá cao

ENTROPY_THRESHOLD = 3.5 # tune theo task

if attention_entropy(img_attn) > ENTROPY_THRESHOLD:

xai_buffer.flag_step(step_idx, reason='high_entropy')

## 2.4 Ứng dụng ngược lại vào Data Collection

### 2.4.1 Điều chỉnh vị trí Global Cam

Nếu attention centroid thường xuyên nằm ở edge of frame hoặc attention mass trên object region < 30%: global cam đang ở góc không tốt. Cần pan/tilt để object chiếm phần trung tâm frame.

def attention_quality_score(heatmap_14x14, object_bbox_norm):

'''

object_bbox_norm: (x1, y1, x2, y2) trong \[0,1\]

Trả về fraction attention mass nằm trong object bbox

'''

H, W = heatmap_14x14.shape

x1 = int(object_bbox_norm\[0\] \* W)

y1 = int(object_bbox_norm\[1\] \* H)

x2 = int(object_bbox_norm\[2\] \* W)

y2 = int(object_bbox_norm\[3\] \* H)

mask = torch.zeros(H, W)

mask\[y1:y2, x1:x2\] = 1.0

return (heatmap_14x14 \* mask).sum() / (heatmap_14x14.sum() + 1e-8)

### 2.4.2 Phát hiện mâu thuẫn giữa Wrist Cam và Global Cam

Florence-2 nhận aux_visual_inputs cho các camera views phụ. So sánh attention pattern từ mỗi camera: nếu wrist cam và global cam cho attention centroid khác xa nhau trên cùng task phase → hai cameras đang cung cấp thông tin conflicting → cần điều chỉnh góc global cam hoặc kiểm tra lại image preprocessing pipeline.

### 2.4.3 Phát hiện Attention Collapse

Nếu model liên tục attend vào token đầu tiên hoặc CLS token thay vì image patches → đây là dấu hiệu training collapse hoặc dataset imbalance. Cần kiểm tra lại language instruction diversity trong training data.

# 3\. P1-V: GMAR - Gradient-weighted Multi-head Attention Rollout

**Priority: P1 | Target: Florence-2 Encoder | Cost: 1.5x forward | Timing: Offline / Triggered**

## 3.1 Cơ sở lý thuyết

GMAR kết hợp hai kỹ thuật: **Attention Rollout** (tích lũy attention qua tất cả layers bằng phép nhân ma trận, bao gồm residual connection) và **Gradient-CAM** (tính gradient của output target w.r.t. intermediate activations để weight từng attention head theo contribution thực sự của nó).

Điểm mấu chốt với X-VLA: thay vì dùng class score như trong image classification, ta dùng **action prediction score** làm gradient target. Cụ thể: predicted action vector của joint quan trọng nhất (ví dụ end-effector) làm scalar để backprop. Điều này tạo ra heatmap **action-conditioned** - tức "vùng ảnh nào ảnh hưởng nhiều nhất đến action này" thay vì chỉ "vùng ảnh nào model nhìn vào".

## 3.2 Cách triển khai

def compute_gmar(model, batch, target_action_dim=None, target_layer_range=None):

'''

Tính GMAR heatmap cho Florence-2 encoder trong X-VLA.

target_action_dim: joint index quan trọng (None = avg all joints)

'''

model.eval()

encoder_layers = model.vlm.language_model.model.encoder.layers

n_layers = len(encoder_layers)

layer_range = target_layer_range or list(range(n_layers))

\# Storage cho attention maps và gradients

attention_maps = \[\] # \[layers, B, H, seq, seq\]

attention_grads = \[\]

hooks = \[\]

for i in layer_range:

def make_hook(idx):

def fwd_hook(m, inp, out):

if isinstance(out, tuple):

attn = out\[1\].requires*grad*(True)

attention_maps.append(attn)

return fwd_hook

h = encoder_layers\[i\].self_attn.register_forward_hook(make_hook(i))

hooks.append(h)

\# Forward pass - giữ graph để backward

with torch.enable_grad():

inputs = model.\_build_model_inputs(batch)

enc = model.model.forward_vlm(\*\*inputs)

\# Tạo noisy action tại t=0.5 (midpoint của flow)

t_mid = torch.full((batch\['input_ids'\].shape\[0\],), 0.5,

device=enc\['vlm_features'\].device)

x_noisy = torch.randn_like(model.model.action_space.zero_action(...))

pred = model.model.transformer(

domain_id=inputs\['domain_id'\],

action_with_noise=x_noisy,

proprio=inputs\['proprio'\],

t=t_mid, \*\*enc

) # \[B, chunk, dim_action\]

\# Target scalar: mean của end-effector joints

if target_action_dim is not None:

target = pred\[:, :, target_action_dim\].mean()

else:

target = pred.mean()

\# Backward để lấy gradient w.r.t. attention weights

target.backward()

\# Thu thập gradients

for attn in attention_maps:

if attn.grad is not None:

attention_grads.append(attn.grad.detach())

\# GMAR computation

B = batch\['input_ids'\].shape\[0\]

seq_len = attention_maps\[0\].shape\[-1\]

rollout = torch.eye(seq_len).unsqueeze(0).expand(B, -1, -1)

for attn, grad in zip(attention_maps, attention_grads):

\# Weight attention heads by gradient magnitude

head_weights = grad.abs().mean(dim=(-2, -1), keepdim=True) # \[B,H,1,1\]

weighted_attn = (attn \* head_weights).sum(dim=1) # \[B, seq, seq\]

weighted_attn = F.relu(weighted_attn) # only positive contributions

\# Add residual connection

weighted_attn = weighted_attn + torch.eye(seq_len, device=weighted_attn.device)

\# Normalize rows

weighted_attn = weighted_attn / (weighted_attn.sum(dim=-1, keepdim=True) + 1e-8)

\# Rollout: matrix multiplication across layers

rollout = weighted_attn @ rollout

\# Extract image patch attention (first P tokens = image)

\# SigLIP default: 14x14 = 196 patches for 224px input

num_img_tokens = 196

img_rollout = rollout\[:, num_img_tokens:, :num_img_tokens\] # lang->img

heatmap = img_rollout.mean(dim=1) # avg over lang tokens -> \[B, 196\]

heatmap = heatmap.reshape(B, 14, 14)

heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

\# Cleanup

for h in hooks: h.remove()

return heatmap # \[B, 14, 14\] -> upsample to 224x224 để overlay

## 3.3 Thời điểm thực hiện

- **Triggered mode (khuyến nghị):** chỉ chạy GMAR khi XAI buffer phát hiện flagged step (high attention entropy từ P0-V). Tránh overhead trên mọi step.
- **Offline batch:** sau mỗi batch data collection, chạy GMAR trên 10-20% steps được sample ngẫu nhiên + toàn bộ flagged steps.
- **Pre-deployment validation:** chạy trên test suite tasks để verify model attend đúng object trong mỗi task phase.

## 3.4 Ứng dụng ngược lại vào Data Collection

### 3.4.1 Language Instruction Sensitivity Test

Chạy GMAR với các instruction variants khác nhau trên cùng một observation frame. Nếu heatmap không thay đổi dù instruction thay đổi rõ ràng (ví dụ: "pick up RED cup" vs "pick up BLUE bottle") → model đang ignore language input.

instructions = \[

'Pick up the red cup',

'Pick up the blue bottle',

'Grasp the object on the left',

'Pour water from the container',

\]

heatmaps = {}

for inst in instructions:

batch\['language'\] = tokenizer(inst)

heatmaps\[inst\] = compute_gmar(model, batch) # \[B, 14, 14\]

\# Pairwise L2 distance giữa heatmaps

from itertools import combinations

diversity_scores = \[\]

for i1, i2 in combinations(instructions, 2):

diff = (heatmaps\[i1\] - heatmaps\[i2\]).pow(2).mean().sqrt().item()

diversity_scores.append(diff)

mean_diversity = sum(diversity_scores) / len(diversity_scores)

if mean_diversity < 0.05: # threshold

print('WARNING: Model ignoring language instructions!')

print('Action: Add contrastive demos to dataset')

### 3.4.2 Contrastive Demo Strategy

Khi phát hiện language insensitivity: cần collect thêm **contrastive episodes** - cùng một scene visual nhưng khác object được chỉ định trong instruction và khác action ground truth. Ví dụ: scene có cả red cup và blue bottle, collect demo "pick red" và "pick blue" trong cùng scene setup.

### 3.4.3 Task-phase Attention Audit

Chia episode thành phases (approach, grasp, manipulate, release). Chạy GMAR per-phase và kiểm tra: trong phase grasp, model có attend vào end-effector region của wrist cam không? Trong phase approach, model có attend vào global object location từ global cam không?

# 4\. P1-A: Denoising Trajectory Visualization

**Priority: P1 | Target: Flow Matching ODE | Cost: ~0 (log only) | Timing: Real-time log**

## 4.1 Cơ sở lý thuyết

Trong Flow Matching, action được generate bằng cách giải ODE: bắt đầu từ pure Gaussian noise x1 ~ N(0,I) và iteratively denoising về action x0. Tại mỗi step i, model dự đoán pred_action (điểm đích), và trajectory di chuyển theo hướng x1\*t + pred\*( 1-t).

Visualizing sequence \[x1, x\_{steps-1}, ..., x_0\] cho thấy quá trình "commit" của model: trajectory mượt và hội tụ nhanh = model confident. Trajectory zigzag hoặc hội tụ chậm = model uncertain, có thể do observation ambiguous hoặc thiếu training data cho tình huống này.

## 4.2 Cách triển khai

Patch vào generate_actions() trong XVLAModel để log x_t tại mỗi denoising step. Không cần thay đổi logic, chỉ thêm một list để collect:

class DenoisingTracker:

def \__init_\_(self):

self.trajectory = \[\] # list of \[B, chunk, dim_action\] tensors

self.enabled = False

def track(self, x_t):

if self.enabled:

self.trajectory.append(x_t.detach().cpu().clone())

def get_convergence_speed(self):

'''

Tính tốc độ hội tụ: delta giữa consecutive x_t

Nhanh = model confident sớm

'''

if len(self.trajectory) < 2:

return \[\]

deltas = \[\]

for i in range(1, len(self.trajectory)):

delta = (self.trajectory\[i\] - self.trajectory\[i-1\]).pow(2).mean().sqrt().item()

deltas.append(delta)

return deltas

def get_final_std(self):

'''Std của last x_t qua chunk dimension = action consistency'''

if not self.trajectory:

return None

final = self.trajectory\[-1\] # \[B, chunk, dim\]

return final.std(dim=1).mean().item() # smooth actions = low std

def clear(self):

self.trajectory.clear()

\# Tích hợp vào generate_actions() - thêm 1 dòng sau mỗi update:

\# denoising_tracker.track(action) # hoặc x_t tùy nhánh

\# Visualize

import matplotlib.pyplot as plt

def plot_denoising_trajectory(tracker, joint_idx=0, batch_idx=0):

'''Plot trajectory của 1 joint qua denoising steps'''

traj = torch.stack(tracker.trajectory) # \[steps, B, chunk, dim\]

joint_traj = traj\[:, batch_idx, :, joint_idx\].numpy() # \[steps, chunk\]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

\# Left: trajectory as heatmap \[steps x chunk\]

axes\[0\].imshow(joint_traj, aspect='auto', cmap='RdBu_r')

axes\[0\].set_xlabel('Action chunk timestep')

axes\[0\].set_ylabel('Denoising step')

axes\[0\].set_title(f'Denoising trajectory - Joint {joint_idx}')

\# Right: convergence speed

deltas = tracker.get_convergence_speed()

axes\[1\].plot(deltas)

axes\[1\].set_xlabel('Denoising step')

axes\[1\].set_ylabel('Delta magnitude')

axes\[1\].set_title('Convergence speed')

plt.tight_layout()

return fig

## 4.3 Thời điểm thực hiện

- **Real-time logging (production):** chỉ log convergence speed scalar (1 float per step, tổng ~10-20 floats per inference). Không lưu full x_t tensor.
- **Triggered full logging:** khi boundary smoothness (P3-RTC) thấp hoặc action entropy cao, enable full x_t logging cho episode đó.
- **Debug mode:** enable toàn bộ tracker, visualize denoising trajectory để hiểu model behavior trên specific scenarios.

## 4.4 Ứng dụng ngược lại vào Data Collection

### 4.4.1 Phát hiện Ambiguous Observations

Khi convergence speed chậm (delta vẫn lớn ở step cuối) → observation đang ambiguous với model. Đây là signal để: **(a)** thu thập thêm demo cho situation này, hoặc **(b)** cải thiện camera angle để giảm ambiguity (ví dụ: thêm lighting, thay đổi global cam góc nhìn).

### 4.4.2 Curriculum Learning Signal

Convergence speed nhanh + std thấp = model đã master situation này → không cần thêm demo. Convergence chậm + high variance = model yếu ở situation này → ưu tiên collect thêm. Đây cho phép xây dựng **active learning strategy**: chỉ collect thêm demo ở regions model uncertain, thay vì collect uniformly.

# 5\. P2-A: Action Sample Bundle

**Priority: P2 | Target: Flow Matching Output | Cost: N x forward | Timing: Offline**

## 5.1 Cơ sở lý thuyết

Flow Matching là generative model - nó học **distribution** của actions, không phải single action. Bằng cách sample nhiều lần với cùng một observation (khởi tạo noise khác nhau mỗi lần), ta có thể ước lượng phân phối action posterior p(a | obs, instruction).

Mean và variance của bundle cho biết: **(1)** action trung bình mà model cho là đúng, **(2)** model uncertain đến mức nào (variance cao), **(3)** action space có multimodal không (bimodal distribution gợi ý có 2 cách thực hiện task hợp lệ).

## 5.2 Cách triển khai

def action_sample_bundle(model, batch, n_samples=50, steps=None):

'''

Sample n_samples action trajectories từ cùng observation.

Returns: dict với mean, std, và full samples

'''

model.eval()

steps = steps or model.config.num_denoising_steps

samples = \[\]

with torch.no_grad():

inputs = model.\_build_model_inputs(batch)

\# Cache VLM encoding - không cần recompute mỗi lần sample

enc = model.model.forward_vlm(

inputs\['input_ids'\],

inputs\['image_input'\],

inputs\['image_mask'\]

)

for \_in range(n_samples):

\# Mỗi sample có noise khác nhau

action = model.model.generate_actions(

\*\*inputs, steps=steps

\# Note: forward_vlm đã cached, nhưng cần truyền lại

\# Có thể patch để reuse enc

) # \[B, chunk, dim_action\]

samples.append(action)

samples = torch.stack(samples) # \[N, B, chunk, dim\]

return {

'samples': samples,

'mean': samples.mean(dim=0), # \[B, chunk, dim\]

'std': samples.std(dim=0), # \[B, chunk, dim\]

'cv': (samples.std(dim=0) / (samples.mean(dim=0).abs() + 1e-6)),

'is_multimodal': detect_multimodal(samples),

}

def detect_multimodal(samples, n_clusters=2):

'''

Dùng simple 1D GMM trên action magnitude để detect bimodal distribution

samples: \[N, B, chunk, dim\]

'''

from sklearn.mixture import GaussianMixture

\# Flatten sang \[N, features\]

N = samples.shape\[0\]

feats = samples\[:, 0, 0, :\].numpy() # first batch, first timestep

gm = GaussianMixture(n_components=n_clusters).fit(feats)

\# Bimodal nếu 2 components đều có weight > 0.2

return all(w > 0.2 for w in gm.weights\_)

def plot_action_bundle(bundle, joint_names=None):

'''Visualize mean +/- std cho từng joint qua chunk timesteps'''

mean = bundle\['mean'\]\[0\] # \[chunk, dim\] - batch 0

std = bundle\['std'\]\[0\]

chunk_len, dim = mean.shape

t_axis = range(chunk_len)

fig, axes = plt.subplots(dim // 3 + 1, 3, figsize=(15, 3\*(dim//3+1)))

for j in range(dim):

ax = axes\[j // 3, j % 3\]

ax.plot(t_axis, mean\[:, j\].numpy(), 'b-', linewidth=2)

ax.fill_between(t_axis,

(mean\[:, j\] - 2\*std\[:, j\]).numpy(),

(mean\[:, j\] + 2\*std\[:, j\]).numpy(),

alpha=0.3, color='blue')

ax.set_title(joint_names\[j\] if joint_names else f'Joint {j}')

return fig

## 5.3 Thời điểm thực hiện

- **Không dùng real-time:** N=50 samples × forward pass chi phí quá cao cho robot control loop.
- **Offline validation post-collection:** sau mỗi session thu thập dữ liệu, chạy bundle analysis trên 20-30 representative observations để đánh giá model uncertainty landscape.
- **Trước khi deploy:** chạy trên toàn bộ task test suite để map uncertainty distribution theo task phases và object types.

## 5.4 Ứng dụng ngược lại vào Data Collection

### 5.4.1 Uncertainty-based Active Data Collection

Tính uncertainty score = mean(std) trên toàn chunk và dim. Rank tất cả observations theo uncertainty. Collect thêm demo tại **top-K uncertain situations** thay vì collect uniformly theo script. Điều này tối ưu hóa data collection effort - 30 targeted demos đôi khi hiệu quả hơn 200 demos trải đều.

### 5.4.2 Multimodal Detection → Demo Stratification

Khi detect bimodal distribution tại một task phase: model nhận ra có 2 cách thực hiện hợp lệ (ví dụ: grasp từ trên hay từ bên). Trong dataset có thể thiếu balance giữa 2 modes này. Cần collect thêm demo của mode ít gặp hơn để force model học cả hai cách.

### 5.4.3 Task Difficulty Mapping

Plot uncertainty heatmap theo object position trong workspace: vùng nào của workspace cho uncertainty cao = robot cần thêm training data ở góc đó. Với global cam có thể thay đổi góc nhìn: căn chỉnh global cam để high-uncertainty workspace regions được nhìn thấy rõ hơn.

# 6\. P2-X: Integrated Gradients - Cross-modal Attribution

**Priority: P2 | Target: Cross-modal (Vision + Language + Proprio) | Cost: 50x forward | Timing: Offline**

## 6.1 Cơ sở lý thuyết

Integrated Gradients (IG) là phương pháp attribution có tính completeness guarantee: tổng attribution của tất cả input features bằng đúng với sự khác biệt output giữa actual input và baseline. Công thức:

IG(x) = (x - x_baseline) × ∫₀¹ ∇_x f(x_baseline + α(x - x_baseline)) dα

Trong đó:

x = actual input (vlm_features, proprio, ...)

x_baseline = zero tensor (neutral input)

f(·) = action prediction function

∇_x f = gradient of action w.r.t. input

Với X-VLA, IG cho phép trả lời câu hỏi quan trọng nhất: **trong observation này, bao nhiêu % của action quyết định đến từ vision (Florence-2), bao nhiêu từ language instruction, và bao nhiêu từ proprioception?** Đây là thông tin không thể có được từ các phương pháp attention-based.

## 6.2 Cách triển khai

def integrated_gradients_xvla(

model, batch, n_steps=50, target_joints=None

):

'''

Tính IG attribution cho vlm_features (vision+language combined).

Để tách vision vs language: cần hook vào image_features và lang_embeds riêng.

'''

model.eval()

inputs = model.\_build_model_inputs(batch)

B = inputs\['input_ids'\].shape\[0\]

\# Baseline: zero image (black frame) + empty language

baseline_inputs = {

'input_ids': inputs\['input_ids'\], # giữ nguyên token IDs

'image_input': torch.zeros_like(inputs\['image_input'\]), # black frames

'image_mask': inputs\['image_mask'\],

'domain_id': inputs\['domain_id'\],

'proprio': torch.zeros_like(inputs\['proprio'\]), # zero proprio

}

\# Compute baseline encoding

baseline_enc = model.model.forward_vlm(

baseline_inputs\['input_ids'\],

baseline_inputs\['image_input'\],

baseline_inputs\['image_mask'\]

)

\# Compute actual encoding

actual_enc = model.model.forward_vlm(

inputs\['input_ids'\],

inputs\['image_input'\],

inputs\['image_mask'\]

)

\# Interpolate và tích phân

grads_vlm = torch.zeros_like(actual_enc\['vlm_features'\])

grads_proprio = torch.zeros_like(inputs\['proprio'\])

for alpha in torch.linspace(0, 1, n_steps):

alpha = alpha.item()

\# Interpolated inputs

interp_vlm = (baseline_enc\['vlm_features'\] +

alpha \* (actual_enc\['vlm_features'\] - baseline_enc\['vlm_features'\])

).requires*grad*(True)

interp_proprio = (baseline_inputs\['proprio'\] +

alpha \* (inputs\['proprio'\] - baseline_inputs\['proprio'\])

).requires*grad*(True)

\# Forward với interpolated features

t_mid = torch.full((B,), 0.5, device=interp_vlm.device)

x_noisy = torch.randn(B, model.model.chunk_size, model.model.dim_action,

device=interp_vlm.device)

proprio_m, x_noisy_m = model.model.action_space.preprocess(

interp_proprio, x_noisy

)

pred = model.model.transformer(

domain_id=inputs\['domain_id'\],

action_with_noise=x_noisy_m,

proprio=proprio_m,

t=t_mid,

vlm_features=interp_vlm,

aux_visual_inputs=actual_enc\['aux_visual_inputs'\],

)

if target_joints:

target = pred\[:, :, target_joints\].mean()

else:

target = pred.mean()

target.backward()

grads_vlm += interp_vlm.grad.detach()

grads_proprio += interp_proprio.grad.detach()

\# Trung bình gradient (trapezoid rule)

grads_vlm /= n_steps

grads_proprio /= n_steps

\# IG = (actual - baseline) \* avg_grad

ig_vlm = (actual_enc\['vlm_features'\] - baseline_enc\['vlm_features'\]) \* grads_vlm

ig_proprio = (inputs\['proprio'\] - baseline_inputs\['proprio'\]) \* grads_proprio

\# Attribution scores

vision_score = ig_vlm\[:, :196, :\].abs().sum().item() # image patch tokens

language_score = ig_vlm\[:, 196:, :\].abs().sum().item() # lang tokens

proprio_score = ig_proprio.abs().sum().item()

total = vision_score + language_score + proprio_score + 1e-8

return {

'vision_pct': 100 \* vision_score / total,

'language_pct': 100 \* language_score / total,

'proprio_pct': 100 \* proprio_score / total,

'ig_vlm': ig_vlm, # \[B, seq, hidden\] - map về ảnh

'ig_proprio': ig_proprio, # \[B, dim_proprio\]

}

## 6.3 Thời điểm thực hiện

- **Không dùng real-time:** 50 forward passes là quá đắt cho robot control loop (có thể mất vài giây/inference trên GPU nhỏ).
- **Monthly model audit:** sau mỗi training cycle, chạy IG analysis trên test set để track xem vision/language/proprio contribution có thay đổi không.
- **Task-specific analysis:** trước khi deploy model cho task mới, chạy IG để verify attribution pattern hợp lý (grasp task nên dominated bởi vision, không phải proprio).

## 6.4 Ứng dụng ngược lại vào Data Collection

### 6.4.1 Modality Imbalance Detection

\# Expected attribution ranges (heuristic cho manipulation tasks):

EXPECTED_VISION_PCT = (40, 70) # vision dominant nhưng không quá overwhelming

EXPECTED_LANGUAGE_PCT = (15, 40) # language có influence rõ rệt

EXPECTED_PROPRIO_PCT = (10, 30) # proprio cần thiết cho fine control

def check_attribution_health(ig_result):

v = ig_result\['vision_pct'\]

l = ig_result\['language_pct'\]

p = ig_result\['proprio_pct'\]

issues = \[\]

if v > 75:

issues.append('Model over-relying on vision - add more language-diverse demos')

if l < 10:

issues.append('Model ignoring language - add contrastive language demos')

if p < 5:

issues.append('Model ignoring proprio - check proprio normalization')

if p > 45:

issues.append('Model over-relying on proprio - check visual diversity')

return issues

### 6.4.2 Wrist Cam vs Global Cam Attribution

Tách IG attribution theo image tokens của wrist cam vs global cam (dựa trên vị trí trong token sequence sau merge). Nếu một camera chiếm attribution > 85%: model đang ignore camera còn lại. Điều này thường xảy ra khi một camera có góc nhìn quá tệ hoặc bị occlusion thường xuyên trong training data.

### 6.4.3 Proprio Feature Importance

IG cho proprio vector cho biết feature nào trong proprio (joint positions, velocities, gripper state) đóng góp nhiều nhất. Nếu gripper state không được attend = model không biết gripper đang mở hay đóng → cần thêm demo với diverse gripper states rõ ràng trong training data.

# 7\. P3-A: Action Dimension Correlation Heatmap

**Priority: P3 | Target: Action Space | Cost: Thấp (post-sampling) | Timing: Offline**

## 7.1 Cơ sở lý thuyết

Flow Matching có thể học **correlated action distributions** - tức các joints không independent mà có coupling structure (ví dụ: shoulder và elbow thường di chuyển phối hợp trong reaching tasks). Visualize correlation matrix giữa action dimensions qua nhiều samples cho thấy coupling structure mà model đã học.

Nếu coupling structure của model khớp với vật lý robot (elbow-shoulder correlation trong reach) → model đang học đúng. Nếu coupling structure kỳ lạ (gripper correlated với base rotation) → model có thể đang học spurious correlation từ training data.

## 7.2 Cách triển khai

def action_correlation_analysis(model, observations, n_samples=100):

'''

Tính correlation matrix giữa action dimensions.

Sử dụng Action Sample Bundle (P2-A) làm input.

'''

all_samples = \[\]

for obs in observations:

bundle = action_sample_bundle(model, obs, n_samples=n_samples)

\# Lấy samples tại timestep đầu của chunk \[N, dim_action\]

first_step = bundle\['samples'\]\[:, 0, 0, :\] # \[N, dim\]

all_samples.append(first_step)

all_samples = torch.cat(all_samples, dim=0) # \[N_total, dim\]

\# Covariance matrix

cov = torch.cov(all_samples.T) # \[dim, dim\]

\# Normalize to correlation

std = all_samples.std(dim=0)

corr = cov / (std.outer(std) + 1e-8)

corr = corr.clamp(-1, 1)

return {'correlation': corr, 'covariance': cov, 'std_per_dim': std}

def plot_correlation_heatmap(corr_matrix, joint_names=None):

import seaborn as sns

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(

corr_matrix.numpy(),

annot=True, fmt='.2f', cmap='RdBu_r',

center=0, vmin=-1, vmax=1,

xticklabels=joint_names or True,

yticklabels=joint_names or True,

ax=ax

)

ax.set_title('Action Dimension Correlation Matrix')

return fig

def detect_spurious_correlations(corr_matrix, robot_kinematic_pairs):

'''

robot_kinematic_pairs: list of (i, j) tuples - cặp joints

nên có correlation cao theo kinematics

Trả về: unexpected_high_corr (không phải kinematic nhưng |corr| > 0.7)

'''

dim = corr_matrix.shape\[0\]

expected = set(map(frozenset, robot_kinematic_pairs))

unexpected = \[\]

for i in range(dim):

for j in range(i+1, dim):

if abs(corr_matrix\[i, j\].item()) > 0.7:

if frozenset({i, j}) not in expected:

unexpected.append((i, j, corr_matrix\[i, j\].item()))

return unexpected

## 7.3 Thời điểm thực hiện

- **Post-training analysis:** sau mỗi training run, chạy correlation analysis để verify coupling structure. So sánh với kinematic constraints của SO-ARM-101.
- **Dataset quality check:** chạy correlation analysis trên ground truth demo actions để có reference matrix, sau đó compare với model-predicted correlation.

## 7.4 Ứng dụng ngược lại vào Data Collection

### 7.4.1 Phát hiện Spurious Correlation từ Demo

Nếu model học được correlation giữa 2 joints không có quan hệ kinematic → trong demo data, hai joints này luôn di chuyển cùng nhau (do người demo có habit nhất định). Cần collect thêm demo với **decoupled motion** của hai joints đó - ví dụ chủ động di chuyển joint A mà không di chuyển joint B.

### 7.4.2 Demo Diversity Metric

So sánh correlation matrix của dataset (ground truth) với correlation matrix của model predictions. Nếu model correlation matrix có off-diagonal entries lớn hơn dataset matrix → model overfitting vào specific patterns trong training data. Giải pháp: tăng diversity của demo styles trong data collection.

# 8\. P3-RTC: Chunk Boundary Smoothness

**Priority: P3 | Target: RTC Pipeline | Cost: ~0 | Timing: Real-time**

## 8.1 Cơ sở lý thuyết

Trong X-VLA với Real-Time Chunking (RTC), action được generate theo chunks. Tại boundary giữa chunk n và chunk n+1, nếu model không nhất quán, robot sẽ có **jerk** - tức thay đổi velocity đột ngột. Jerk trong robot manipulation là nguy hiểm (stress cơ khí) và thường dẫn đến task failure.

Cosine similarity giữa tail của chunk n và head của chunk n+1 đo mức độ smooth của transition. Score gần 1.0 = transition mượt. Score thấp hoặc âm = jerk mạnh.

Trong X-VLA code: rtc_processor.denoise_step() đã nhận prev_chunk_left_over để cải thiện consistency. Boundary smoothness metric đo lường hiệu quả của mechanism này.

## 8.2 Cách triển khai

class BoundarySmoothnessMonitor:

def \__init_\_(self, overlap_steps=3, low_threshold=0.75, critical_threshold=0.5):

self.overlap_steps = overlap_steps # số bước check ở boundary

self.low_threshold = low_threshold

self.critical_threshold = critical_threshold

self.history = \[\]

self.prev_chunk = None

def update(self, new_chunk):

'''

new_chunk: \[B, chunk_size, dim_action\]

'''

if self.prev_chunk is None:

self.prev_chunk = new_chunk

return None

\# So sánh tail của chunk cũ với head của chunk mới

prev_tail = self.prev_chunk\[:, -self.overlap_steps:, :\] # \[B, k, dim\]

new_head = new_chunk\[:, :self.overlap_steps, :\] # \[B, k, dim\]

\# Cosine similarity trung bình qua overlap window và batch

sim = F.cosine_similarity(

prev_tail.flatten(1),

new_head.flatten(1),

dim=1

).mean().item() # scalar

self.history.append(sim)

self.prev_chunk = new_chunk

return sim

def get_status(self, sim):

if sim is None:

return 'ok'

if sim < self.critical_threshold:

return 'critical_jerk' # dừng lại xem xét

if sim < self.low_threshold:

return 'warning_jerk' # flag episode

return 'ok'

def episode_quality(self):

'''

Trả về tỷ lệ boundary transitions smooth trong episode

'''

if not self.history:

return 1.0

good = sum(s >= self.low_threshold for s in self.history)

return good / len(self.history)

def reset(self):

self.history.clear()

self.prev_chunk = None

\# Real-time integration trong robot control loop:

monitor = BoundarySmoothnessMonitor()

\# Sau mỗi generate_actions():

sim = monitor.update(new_action_chunk)

status = monitor.get_status(sim)

if status == 'critical_jerk':

\# Tùy chọn: safe stop hoặc rerun với different seed

logger.warning(f'Critical jerk detected at step {step}: sim={sim:.3f}')

\# Flag episode cho offline review

episode_flags.append(('jerk', step, sim))

## 8.3 Thời điểm thực hiện

- **Real-time trong data collection:** luôn bật, chi phí chỉ là 1 cosine_similarity call. Log scalar mỗi chunk boundary.
- **Post-episode filter:** dùng episode_quality() score để tự động lọc demo chất lượng thấp.
- **Safety gate trong deployment:** nếu sim < critical_threshold, có thể trigger safe stop và request model rerun.

## 8.4 Ứng dụng ngược lại vào Data Collection

### 8.4.1 Automatic Demo Quality Filter

Sau mỗi teleoperation session, lọc tất cả episodes có episode_quality() < 0.80 khỏi training set. Hoặc áp dụng per-step loss weight dựa trên smoothness score (smooth transitions có weight cao hơn trong training loss).

def weighted_imitation_loss(pred_actions, target_actions, smoothness_weights):

'''

smoothness_weights: \[B, chunk\] - cao hơn ở những step smooth trong demo

'''

per_step_loss = F.mse_loss(pred_actions, target_actions, reduction='none')

per_step_loss = per_step_loss.mean(dim=-1) # \[B, chunk\]

weighted_loss = (per_step_loss \* smoothness_weights).mean()

return weighted_loss

### 8.4.2 Jerk Source Analysis

Tương quan timestep của jerk detection với: **(a)** observation entropy từ P0-V (jerk xảy ra khi observation ambiguous?), **(b)** task phase (jerk xảy ra nhiều nhất ở phase nào?), **(c)** wrist cam occlusion events. Điều này giúp xác định root cause: nếu jerk luôn xảy ra khi wrist cam bị object che khuất → cần collect thêm demo trong điều kiện occlusion đó.

### 8.4.3 Global Cam Angle Optimization

Track xem jerk rate thay đổi như thế nào khi thay đổi góc global cam. Nếu một góc cam nhất định cho jerk rate thấp hơn trên cùng task → đó là góc cam tốt hơn cho data collection và deployment.

# 9\. Tích hợp Hệ thống: XAI Buffer & Pipeline

Bảy phương pháp trên được tổ chức thành một pipeline nhất quán: real-time monitoring layer (P0, P1-A, P3) + offline analysis layer (P1-V, P2, P3-A, full IG) + data collection feedback loop.

## 9.1 XAI Buffer - Cấu trúc dữ liệu

from dataclasses import dataclass, field

from typing import List, Optional

import torch

@dataclass

class StepRecord:

step_idx: int

timestamp: float

attn_entropy: float # P0-V

attn_compressed: torch.Tensor # P0-V: \[7, 7\]

convergence_speed: List\[float\] # P1-A: \[n_denoising_steps\]

boundary_sim: Optional\[float\] # P3-RTC

flagged: bool

flag_reason: Optional\[str\]

@dataclass

class EpisodeXAIBuffer:

episode_id: str

step_records: List\[StepRecord\] = field(default_factory=list)

flagged_steps: List\[int\] = field(default_factory=list)

\# Computed post-episode

episode_quality: Optional\[float\] = None # từ P3-RTC

mean_entropy: Optional\[float\] = None # từ P0-V

attention_stability: Optional\[float\] = None # std of attn centroid

def add_step(self, record: StepRecord):

self.step_records.append(record)

if record.flagged:

self.flagged_steps.append(record.step_idx)

def should_run_offline_xai(self) -> bool:

'''Offline XAI nếu > 10% steps bị flag'''

if not self.step_records: return False

return len(self.flagged_steps) / len(self.step_records) > 0.1

def summary(self) -> dict:

entropies = \[r.attn_entropy for r in self.step_records\]

sims = \[r.boundary_sim for r in self.step_records if r.boundary_sim\]

return {

'n_steps': len(self.step_records),

'n_flagged': len(self.flagged_steps),

'mean_entropy': sum(entropies) / len(entropies),

'mean_boundary_sim': sum(sims)/len(sims) if sims else None,

'episode_quality': self.episode_quality,

}

## 9.2 Decision Pipeline: Khi nào chạy gì

class XAIPipeline:

ENTROPY_FLAG = 3.5 # P0-V flag threshold

BOUNDARY_LOW = 0.75 # P3-RTC warning

BOUNDARY_CRITICAL = 0.50 # P3-RTC critical

QUALITY_THRESHOLD = 0.80 # Episode quality để include trong training

def \__init_\_(self, model, offline_runner=None):

self.model = model

self.attn_hook = AttentionHook()

self.deno_tracker = DenoisingTracker()

self.boundary_mon = BoundarySmoothnessMonitor()

self.offline = offline_runner # runs GMAR, IG, bundle offline

self.current_ep = None

def start_episode(self, episode_id):

self.current_ep = EpisodeXAIBuffer(episode_id=episode_id)

self.attn_hook.register(self.model)

self.deno_tracker.enabled = True

self.boundary_mon.reset()

def step(self, batch, action_chunk, step_idx):

\# P0-V: attention entropy

img_attn = self.attn_hook.get_image_attention(...)

entropy = attention_entropy(img_attn\[0\])

compressed = img_attn\[0\].reshape(14,14)

compressed = F.avg_pool2d(compressed\[None,None\], 2)\[0,0\]

\# P3-RTC: boundary smoothness

sim = self.boundary_mon.update(action_chunk)

status = self.boundary_mon.get_status(sim)

\# P1-A: convergence (từ denoising_tracker.get_convergence_speed())

conv = self.deno_tracker.get_convergence_speed()

self.deno_tracker.clear()

\# Flagging logic

flagged = False

reason = None

if entropy > self.ENTROPY_FLAG:

flagged, reason = True, 'high_entropy'

elif status == 'critical_jerk':

flagged, reason = True, 'critical_jerk'

elif status == 'warning_jerk':

flagged, reason = True, 'warning_jerk'

record = StepRecord(

step_idx=step_idx, timestamp=time.time(),

attn_entropy=entropy, attn_compressed=compressed,

convergence_speed=conv, boundary_sim=sim,

flagged=flagged, flag_reason=reason

)

self.current_ep.add_step(record)

self.attn_hook.clear()

def end_episode(self) -> EpisodeXAIBuffer:

self.attn_hook.remove()

self.deno_tracker.enabled = False

self.current_ep.episode_quality = self.boundary_mon.episode_quality()

\# Schedule offline XAI nếu cần

if self.current_ep.should_run_offline_xai() and self.offline:

self.offline.schedule(self.current_ep.episode_id)

ep = self.current_ep

self.current_ep = None

return ep

def should_include_in_training(self, episode: EpisodeXAIBuffer) -> bool:

return episode.episode_quality >= self.QUALITY_THRESHOLD

# 10\. Tổng kết: Ma trận Quyết định Data Collection

Mỗi XAI signal dẫn đến một hoặc nhiều action cụ thể trong quá trình thu thập dữ liệu:

| **XAI Signal**                             | **Triệu chứng**                        | **Root Cause (Khả năng)**                         | **Action Data Collection**                                                      |
| ------------------------------------------ | -------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------- |
| P0-V: Attention entropy cao                | Model attend dàn trải, không focus     | Cam angle tệ, object quá nhỏ trong frame          | Điều chỉnh global cam gần/zoom hơn. Kiểm tra lighting.                          |
| P0-V: Attention mass thấp trên object      | < 30% attention trong object region    | Object bị edge of frame hoặc occlude              | Pan/tilt global cam để object ở center. Tăng object size trong frame.           |
| P1-V: GMAR không thay đổi theo instruction | Language insensitive                   | Thiếu contrastive language demos                  | Collect demos: cùng scene, khác object được chỉ định, khác action.              |
| P1-A: Convergence chậm                     | Delta lớn ở late denoising steps       | Observation ambiguous với model                   | Collect thêm demo cho situation này. Cải thiện lighting/angle.                  |
| P2-A: Std cao (uncertainty)                | Action bundle có spread lớn            | Model chưa master situation này                   | Active collection: tập trung collect ở uncertain situations.                    |
| P2-A: Bimodal distribution                 | Detect 2 clusters trong samples        | Imbalance giữa 2 execution modes                  | Collect thêm demo cho mode ít gặp hơn trong dataset.                            |
| P2-X: Vision % > 75%                       | Model ignore language                  | Thiếu language diversity trong data               | Collect contrastive language demos. Augment instruction phrasings.              |
| P2-X: Language % < 10%                     | Model ignore instructions hoàn toàn    | Possible training bug hoặc data issue             | Kiểm tra tokenizer. Verify instruction diversity trong dataset.                 |
| P2-X: Proprio % > 45%                      | Model overfit vào proprio patterns     | Visual diversity thấp trong dataset               | Collect demos với varied object positions, varied scene layouts.                |
| P2-X: Wrist cam attribution > 85%          | Model ignore global cam                | Global cam angle không tốt hoặc redundant         | Thay đổi global cam angle. Collect demos với varied global cam positions.       |
| P3-A: Spurious joint correlation           | Unexpected coupling trong action space | Demo bias: người demo luôn move 2 joints together | Collect demos với decoupled joint motion.                                       |
| P3-RTC: Boundary sim thấp                  | Jerk tại chunk transitions             | Model uncertain tại specific observations         | Thu thập thêm demo tại task phases có jerk. Có thể thêm overlap trong chunking. |
| P3-RTC: Episode quality < 0.80             | Demo bị jerky                          | Teleoperation quality thấp                        | Lọc episode này khỏi training set. Hướng dẫn lại người thực hiện demo.          |

# 11\. Quy trình Làm việc Khuyến nghị

## 11.1 Trước khi thu thập dữ liệu (Setup)

- Calibrate global cam: chạy P0-V với dummy observations, kiểm tra attention centroid có nằm trong workspace center không.
- Set thresholds cho XAI buffer dựa trên pilot episodes (10-20 episodes). Tune ENTROPY_FLAG và BOUNDARY thresholds theo task cụ thể.
- Verify attribution health bằng P2-X trên pilot dataset. Nếu có modality imbalance từ đầu, điều chỉnh collection protocol trước khi thu thập lớn.

## 11.2 Trong quá trình thu thập dữ liệu (Real-time)

- Bật P0-V (attention entropy) và P3-RTC (boundary smoothness) cho mọi episodes.
- Enable P1-A (denoising tracker) với lightweight logging (scalar only).
- Sau mỗi episode: kiểm tra episode_quality score. Nếu < 0.80, discard hoặc repeat.
- Nếu episode bị flag nhiều (> 10% steps): đưa vào offline XAI queue.

## 11.3 Sau mỗi collection session (Offline)

- Chạy P1-V (GMAR) trên tất cả flagged episodes + 10% random sample.
- Chạy P2-A (action sample bundle) trên representative observations theo task phases.
- Update uncertainty map: vùng nào của workspace/task space cần thêm demos.
- Nếu có model version mới sau re-training: chạy P2-X (Integrated Gradients) để verify attribution health.

## 11.4 Chu kỳ cải tiến (Iteration)

- Training: dùng episode quality score làm sample weight trong imitation learning loss.
- Evaluation: compare P0-V attention quality scores trước và sau training. Metric: attention mass trên object region.
- Camera adjustment: dùng P1-V attention centroid statistics để quyết định có cần điều chỉnh global cam vị trí không.
- Language audit: dùng P1-V instruction sensitivity test để verify model đang học language grounding đúng cách.

# 12\. Kết luận

Bảy phương pháp XAI trong báo cáo này được thiết kế để bổ sung cho nhau: **P0-V và P3-RTC** làm real-time sentinel, **P1-V và P1-A** đi sâu vào cơ chế quyết định của model, **P2-A và P2-X** cung cấp statistical understanding về uncertainty và attribution, và **P3-A** audit coupling structure của action space.

Điểm quan trọng nhất: XAI không chỉ là công cụ debug sau khi model fail. Khi tích hợp vào data collection pipeline qua XAI buffer, chúng trở thành **vòng phản hồi liên tục** giúp tối ưu hóa: (1) vị trí camera, (2) ngôn ngữ instruction, (3) phân phối demo theo task phases, và (4) chất lượng demo teleoperation. Điều này đặc biệt quan trọng với setup 2-camera wrist + global của SO-ARM-101, nơi global cam có thể điều chỉnh để maximize information value dựa trên XAI feedback.

**Khuyến nghị implement theo thứ tự:** P3-RTC (1 ngày) → P0-V với XAI buffer (2-3 ngày) → P1-A denoising tracker (1 ngày) → P1-V GMAR offline (3-5 ngày) → P2-A bundle + P2-X IG (1-2 tuần). P3-A là optional, chỉ cần thiết khi suspect demo bias.

## **References - Phương pháp XAI cho VLA**

### **Vision XAI (Florence-2 / Transformer Encoder)**

**P0-V: Raw Attention Map**

- Dosovitskiy et al. (2020). _An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale._ arXiv:2010.11929 - baseline ViT attention visualization.
- Caron et al. (2021). _Emerging Properties in Self-Supervised Vision Transformers (DINO)._ ICCV 2021 - chứng minh raw attention head của ViT có thể segment object mà không cần label, motivation cho việc visualize attention trong robotics.

**P1-V: Attention Rollout**

- Abnar & Zuidema (2020). _Quantifying Attention Flow in Transformers._ ACL 2020. arXiv:2005.00928 - paper gốc đề xuất Attention Rollout với residual connection trick.

**P1-V: GMAR (Gradient × Attention Rollout)**

- Selvaraju et al. (2017). _Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization._ ICCV 2017 - paper gốc Grad-CAM, nền tảng của gradient-weighted attention.
- Chefer et al. (2021). _Transformer Interpretability Beyond Attention Visualization._ CVPR 2021. arXiv:2012.09838 - LRP-based relevancy propagation cho transformer, class-specific và xử lý được negative contributions mà Grad-CAM bỏ qua.
- Bahrami et al. (2025). _GMAR: Gradient-Driven Multi-Head Attention Rollout for Vision Transformer Interpretability._ arXiv:2504.19414 - paper mới nhất kết hợp gradient với multi-head attention rollout, outperform cả Attention Rollout và Grad-CAM trên faithfulness metrics.

**Áp dụng XAI vào VLA / Robot Vision cụ thể**

- Pani & Yang (2025). _Gaze-Regularized Vision-Language-Action Models for Robotic Manipulation._ arXiv:2603.23202 - dùng human gaze heatmap để regularize VLA attention bằng KL divergence, 4-12% improvement trên manipulation benchmarks. Trực tiếp validate rằng attention pattern của VLA có thể được đo và cải thiện.
- Bousselham et al. (2025). _DEX-AR: A Dynamic Explainability Method for Autoregressive Vision-Language Models._ arXiv:2603.06302 - per-token heatmap cho autoregressive VLM, dynamic head filtering để phân biệt visual attention head vs linguistic attention head - áp dụng được cho Florence-2 encoder của X-VLA.

### **Action XAI (Flow Matching)**

**P1-A: Denoising Trajectory & P2-A: Action Sample Bundle**

- Lipman et al. (2022). _Flow Matching for Generative Modeling._ arXiv:2210.02747 - paper gốc Flow Matching, định nghĩa ODE trajectory từ noise đến data.
- Black et al. (2024). _π₀: A Vision-Language-Action Flow Model for General Robot Control._ Physical Intelligence. arXiv:2410.24164 - paper đầu tiên áp dụng flow matching vào VLA với action chunking 50Hz; action sample bundle concept xuất phát từ đây.
- Park et al. (2025). _ACG: Action Coherence Guidance for Flow-based VLA Models._ arXiv:2510.22201 - phân tích incoherence trong flow matching policy do demo noise (jerk, pause, jitter), đề xuất training-free guidance. Trực tiếp motivate Denoising Trajectory visualization và Chunk Boundary metric.

**P3-RTC: Chunk Boundary Smoothness**

- Black et al. (2024, π₀) - action chunking và inference delay analysis (synchronous vs asynchronous chunk execution).
- Zhao et al. (2023). _Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)._ RSS 2023 - paper gốc action chunking trong robot learning, giới thiệu temporal ensemble để smooth chunk boundaries.

**P3-A: Action Dimension Correlation**

- Domingo-Enrich et al. (2025). _Flow Policy Gradients for Robot Control (FPO++)._ arXiv:2602.02481 - visualize flow field và cross-correlation heatmap giữa action dimensions trong locomotion policy, chứng minh coupling structure học được consistent với kinematic constraints.

### **Cross-modal Attribution**

**P2-X: Integrated Gradients**

- Sundararajan et al. (2017). _Axiomatic Attribution for Deep Networks._ ICML 2017 - paper gốc Integrated Gradients với completeness axiom, đây là phương pháp duy nhất trong nhóm gradient-based có lý thuyết attribution guarantee.
- Lundberg & Lee (2017). _A Unified Approach to Interpreting Model Predictions (SHAP)._ NeurIPS 2017 - alternative attribution method dùng game theory; SHAP values thường được so sánh với IG trong ablation studies.

### **Survey / Tổng quan XAI**

- Kawaharazuka et al. (2025). _Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications._ IEEE Access, vol. 13. - survey đầy đủ nhất về VLA hiện tại, có section về interpretability gaps.
- Sapkota et al. (2025/2026). _Vision-Language-Action (VLA) Models: Concepts, Progress, Applications and Challenges._ arXiv:2505.04769 (v2 Jan 2026) - 80+ VLA models, section về evaluation và XAI needs.
- Zablocki et al. (2022). _Explainability of Vision-based Autonomous Driving Systems._ arXiv:2101.05307 - mặc dù về autonomous driving, đây là reference quan trọng nhất về XAI cho end-to-end visuomotor policies với cả attention maps, saliency, và causal intervention methods.

### **Về X-VLA cụ thể**

- Zheng et al. (2025). _X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model._ arXiv:2510.10274. ICLR 2026. - paper gốc của model bạn đang dùng, đặc biệt phần soft prompt mechanism và cross-embodiment generalization.
- Xiao et al. (2023). _Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks._ arXiv:2311.06242 - Florence-2 backbone được dùng trong X-VLA.