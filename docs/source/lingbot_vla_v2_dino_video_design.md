# DINO-video 第一方 Runtime 设计

## 决策

DINO-video 作为 native-depth 蒸馏 PR 的一个独立教师实现，不作为上游 checkout
provider，不 vendor Lumos / Meta DINOv3 源码，也不把视频模型逻辑塞入
`depth_teachers.py`。

稳定边界如下：

```text
LingbotVLAV2Policy
  └─ DepthTeacherBundle                         # 生命周期、no-grad、目标抽取
       └─ DinoVideoTeacher                      # 训练侧公共教师 API
            └─ FirstPartyDinoVideoBackbone      # 视频 ViT / RoPE / causal attention
                 └─ SDPA backend                # 默认纯 torch；可选优化后端
```

开发者只提供：

```text
dino_video/teacher_step_10000.pth
dino_video/config.yaml
```

不需要 clone、环境变量、`sys.path` 注入或第三方代码仓。

## 许可证门槛

上游 Lumos DINOv3 runtime 使用 Meta DINOv3 License（非 OSI）。因此本仓不得复制、
裁剪、改名或直接迁入其源码。权重兼容实现必须先得到项目/法务对以下边界的确认：

1. 允许以权重 layout、公开模型配置和公开论文行为为输入编写独立实现；
2. 开发者自行下载的 DINO 权重的许可接受与分发责任归属明确；
3. 实现过程不能以复制 Lumos 源码为来源。

在这一确认之前，PR 应保持 DINO 明确禁用；不要以外部 checkout 作为临时 fallback。

## 推荐目录

DINO-video 走独立子包(二级目录),与 depth 教师平级隔离;骨架已落盘:

```text
src/lerobot/policies/lingbot_vla_v2/teachers/
├── depth_teachers.py                 # 既有: bundle、图像预处理、target 解包
├── native_depth_models.py             # 已有: MoGe v2
├── morgbd_teacher.py                  # 已有: MoRGBD
└── dino_video/                        # 新子包: DINO-video 第一方 runtime
    ├── __init__.py                    # 唯一公开出口: DinoVideoTeacher, build_dino_video_teacher
    ├── teacher.py                     # public facade: from_pretrained + get_future_feature
    ├── checkpoint.py                  # weights_only loader、严格 coverage validator
    ├── backbone.py                    # patch embed、token layout(PackedVideoTokens)、ViT blocks
    ├── rope.py                        # 3D RoPE、fps / temporal coordinate
    └── attention.py                   # block-causal mask、SDPA/flex dispatch
```

包内互相引用用相对 import(``from .backbone import PackedVideoTokens``);
``depth_teachers.py`` 在 P4 接线时才懒加载 ``from .dino_video import
build_dino_video_teacher``,保证未启用 DINO 时零导入开销。子包不得再嵌套
深层目录,不得出现 ``vendor/`` / ``third_party/`` / ``lumos_dinov3/`` 等上游
命名。

不要新建 `vendor/`、`third_party/`、`lumos_dinov3/` 或用上游包名伪装第一方代码。

## 必须提供的接口

### 1. 公开 builder

`dino_video_teacher.py`：

```python
def build_dino_video_teacher(config: dict) -> DinoVideoTeacher: ...

class DinoVideoTeacher(nn.Module):
    @classmethod
    def from_pretrained(
        cls,
        ckpt_path: str | Path,
        config_path: str | Path,
        *,
        device: torch.device,
    ) -> "DinoVideoTeacher": ...

    def get_future_feature(
        self,
        video: torch.Tensor,             # [B, C, T, H, W]，已 ImageNet normalize
        *,
        return_cls: bool = False,
        return_current: bool = False,
        current_index: int = 0,
        fps: float | torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]: ...
```

返回契约必须保持现有 `DepthTeacherBundle.video_targets()` 的解包语义：

| 开关 | 返回 |
|---|---|
| patch only | `future_patch: [B, 256, 1024]` |
| `return_cls=True` | `(future_patch, future_cls[B,1024])` |
| `return_current=True` | `(future_patch, current_patch)` |
| 两者为真 | `(future_patch, future_cls, current_patch, current_cls)` |

这样策略、processor、forward kwargs、loss heads 和 converter 均不需要为 DINO
重新改接口。

### 2. 配置 schema

`align_params.video` 已有的字段应继续是唯一用户面：

```json
{
  "ckpt_path": "/weights/teacher_step_10000.pth",
  "config_path": "/weights/config.yaml",
  "attention_mode": "flex_block_causal",
  "input_size": 256,
  "n_blocks": 1,
  "effective_fps": 1.0,
  "cls_pool": "last",
  "num_future_frames": 1,
  "use_warmup_frame": true
}
```

新增的 runtime-only 字段应明确、可选且有安全默认值：

```json
{
  "runtime": "first_party",
  "attention_backend": "sdpa",
  "strict_checkpoint": true,
  "compile": false
}
```

不应再接受 `upstream_root`、`provider_path`、`LINGBOT_VLA_V2_UPSTREAM`。

### 3. Checkpoint loader

`dino_video_checkpoint.py`：

```python
def load_dino_video_checkpoint(path: Path) -> dict[str, torch.Tensor]: ...
def load_backbone_strict(model: nn.Module, state: dict[str, torch.Tensor]) -> LoadReport: ...
```

要求：

- 使用 `torch.load(..., map_location="cpu", weights_only=True)`；
- 只从顶层 `teacher` 读取 state dict；
- backbone 张量必须严格覆盖：`cls_token`、`mask_token`、`storage_tokens`、
  `patch_embed`、`rope_embed`、24 个 block、norm；
- `dino_head.*` / `ibot_head.*` 不参与 `get_future_feature`，可被显式列为
  `allowed_unused_prefixes`，但不得无声跳过其他 tensor；
- 出现未知 missing/unexpected/shape mismatch 时给出完整、可行动的错误；
- 记录 checkpoint tensor 数、模型 tensor 数、unused head tensor 数，便于测试固定。

已知发布 checkpoint 的 backbone 是 ViT-L/16：hidden=1024、24 层、16 heads、
MLP=4096、4 个 storage token、3D RoPE、fp32 权重。

## 第一方内部实现

### Token layout

以一个显式函数固定 token 顺序，避免未来改动造成 silent numerical drift：

```python
def pack_video_tokens(
    patch_tokens: Tensor,      # [B, T, H*W, D]
    cls_token: Tensor,         # [1, 1, D]
    storage_tokens: Tensor,    # [1, S, D]
) -> PackedVideoTokens: ...
```

`PackedVideoTokens` 应携带：

- 扁平 tokens；
- 每个 token 的 `(frame, row, col, kind)` 坐标；
- patch token 的 frame offsets；
- 后续 causal mask 需要的 block id；
- 当前/未来帧切片索引。

不要在 attention layer 里用魔法 offset 推算 frame 索引。

### 3D RoPE

`dino_video_rope.py` 的接口：

```python
class VideoRoPE(nn.Module):
    def forward(
        self,
        q: Tensor, k: Tensor,              # [B, heads, tokens, head_dim]
        token_coordinates: TokenCoordinates,
        *,
        fps: float | Tensor | None,
    ) -> tuple[Tensor, Tensor]: ...
```

需要覆盖：空间 `periods`、时间 `periods_t`、`base_fps`、prefix temporal 规则、
fp32 position buffer / bf16 activation 的精度分离。RoPE buffer 不能随 `.to(bfloat16)`
损失为 bf16；这是模型数值稳定性的硬约束。

### Block-causal attention

`dino_video_attention.py` 应只暴露一个 attention API：

```python
def block_causal_attention(
    q: Tensor, k: Tensor, v: Tensor,
    layout: PackedVideoTokens,
    *, backend: Literal["sdpa", "flex"] = "sdpa",
) -> Tensor: ...
```

实现策略：

1. **默认后端：PyTorch SDPA。** 生成布尔/加性 attention mask，确保某一帧的
   patch token 只能访问规定的前缀帧 block；这是零额外依赖、CI 可执行的真值实现。
2. **可选优化：torch flex attention。** 仅在 torch 版本、GPU 能力和 runtime
   检查通过时启用；必须与 SDPA reference 逐位/容差对照，不能成为安装必需项。
3. 不引入 xformers、flash-attn 或上游 attention wrapper 作为硬依赖。

mask 的构造应单测覆盖：`T=2/3`、warmup frame、不同 storage-token 数、current
index、batch > 1、非方形 patch grid。

### Transformer block

`dino_video_backbone.py` 内部仅用 stock torch：

- `Conv2d` patch embedding；
- LayerNormBF16 等价层（归一化计算 fp32、输出按 activation dtype）；
- fused QKV `Linear` + SDPA；
- LayerScale；
- MLP `Linear → GELU → Linear`；
- 24 层 ViT-L block；
- 最终 norm 与 intermediate layer 输出。

先实现 inference-only runtime：`requires_grad_(False)`，不引入训练、DINO loss、
iBOT loss、EMA、masking augmentation、数据加载或分布式代码。发布权重里的
`dino_head` / `ibot_head` 不会被 native-depth 蒸馏 target 使用，不应进入首版。

## 接回现有教师 bundle

`depth_teachers.py::_load_video_teacher()` 只需要替换为：

```python
from .dino_video_teacher import build_dino_video_teacher

video_cfg = params["video"]
_require_file(video_cfg.get("ckpt_path"), "align_params.video.ckpt_path")
_require_file(video_cfg.get("config_path"), "align_params.video.config_path")
config = dict(video_cfg)
config["device"] = str(device)
return _freeze(build_dino_video_teacher(config), device)
```

`video_targets()` 的图像 `/255`、resize、ImageNet normalize、warmup frame、fps
转发、tuple/dict 解包逻辑已在现有 bundle 中；不要在 DINO runtime 再做一次归一化
或 resize。

## 分阶段实施和验收

### P0：许可与权重契约（先于编码）

- 法务/项目确认独立权重兼容实现及用户自下载权重的边界；
- 固定 checkpoint SHA256、config 内容、359 tensor manifest；
- 记录 backbone 的 345 ���必须 tensor，14 个允许未使用 head tensor。

### P1：加载与 2D 骨架

- checkpoint strict coverage；
- patch embed、storage token、ViT block、final norm；
- 单帧/无 causal mask 下与可用受控 reference 对照；
- CPU 测试只检 state coverage、shape、输入验证。

### P2：视频 token 和 3D RoPE

- `T=2`，256×256，输出 patch `[B, T*256, 1024]`；
- 验证 fps 变化只影响 temporal RoPE；
- fp32 RoPE buffer 不被 dtype cast。

### P3：block-causal SDPA

- SDPA mask reference；
- `get_future_feature()` 四种返回组合；
- `use_warmup_frame=true`、`current_index=1`、有效 fps 回归测试。

### P4：接入策略与性能

- `_load_video_teacher()` 改成第一方 builder；
- 解除 DINO runtime 缺失错误；
- 用现有 `test_align_teachers.py` 的 stub 契约替换为真权重 gated integration；
- 先测单卡 CUDA，再在需要时加 flex backend；
- 全 PR 需跑 CPU 单测、真实权重 shape/dtype、训练 forward smoke、DDP rank-local
  teacher smoke。

## 明确不引入的代码或依赖

- 不引入上游 LingBot checkout；
- 不引入 `lumos_dinov3`、Meta DINOv3 runtime、其 import path 或源码复制；
- 不引入 xformers / flash-attn 作为必需依赖；
- 不引入 DINO/iBOT training head、trainer、EMA、数据集或分布式训练代码；
- 不让教师成为 policy 的 `nn.Module` 子模块，不进 optimizer/DDP/state dict；
- 不更改 processor / policy forward 的已存在 target API。
