# 实验计划:动作 token 预训练 VLM → 冻结 + flow expert(SmolVLA / LIBERO)

> 起草 2026-06-13。目标:验证"先用 FAST 动作 token 自回归预训练 VLM(注入空间/动作理解),再冻结 VLM、拼接随机初始化 flow-matching action expert(AE)训练"是否优于直接冻结 stock VLM 训 AE。在 LIBERO 上对齐论文级评测。

---

## 0. TL;DR / 核心结论(先看这里)

1. **你的设想 = π0.5 / Knowledge Insulation(KI)的路线**。这条路是对的,但有一个关键偏差:**PI 自己做过"冻结 VLM + 训全新 expert"的消融,结果接近 0%**;他们因此放弃硬冻结,改用 **stop-gradient 共训**(KI)。你的 Stage-2(冻结预训练后的 VLM + 训全新 expert)正好是被放弃的那个配置 → **建议加第 3 个 KI 共训臂**。
2. **SmolVLA 本身没有 FAST/文本 token 头**,只有 flow expert。**在 smolvla 里无法直接做 Stage-1 的 FAST 自回归预训练**。
3. **lerobot 的 FAST 机制(`pi0_fast`、`pi052`)全建在 PaliGemma-2B 上,不是 SmolVLM2-500M**。其中 **`pi052` 已经把 π0.5/KI 完整实现了**(FAST-AR + flow + 文本 + stop-gradient + 自动拟合 FAST tokenizer)。
4. 因此有两条路线(见 §4):
   - **路线 A(省力、有据)**:直接用 `pi052`(PaliGemma 家族),通过开关 flag 构造对比臂。几乎零新代码。
   - **路线 B(贴合你的原意、需自研)**:坚持 SmolVLM2-500M,自己给它加 FAST-AR 头做 Stage-1,再喂给 smolvla 冻结训 AE。工程量大。
5. **校正一个理解**:`smolvla_base` 出厂配方就是"冻结 VLM(`freeze_vision_encoder=True`)+ 只训 expert(`train_expert_only=True`)",论文 87.3% 就是这么来的 —— 所以对 SmolVLA 而言"冻结"并非 0%,与 KI 的 0% 消融是不同上下文(KI 是 PaliGemma-π0 设定)。证据是混合的,这正是值得实验的地方。

---

## 1. 背景与已核实事实(带 file:line)

### 1.1 SmolVLA 架构 / 训练(`src/lerobot/policies/smolvla/`)
- VLM = `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`(SigLIP 视觉 + SmolLM2 文本,文本 **32 层**),smolvla 只取**前 16 层**(`num_vlm_layers=16`,半数)。`configuration_smolvla.py:96`
- AE = 并行 Gemma transformer,宽度 `0.75×`(`expert_width_multiplier`),层数同 VLM,每 2 层插一个 self-attn(`self_attn_every_n_layers=2`),通过 **cross-attention**(`attention_mode="cross_attn"`)读 VLM 特征,**随机初始化**。`smolvlm_with_expert.py:73-145`
- 动作头 = **flow matching**(`num_steps=10` 去噪步),输出 `chunk_size=50`。**没有任何 FAST/文本 token 损失**(grep 确认)。
- 冻结逻辑 `set_requires_grad()` `smolvlm_with_expert.py:150-181`:
  - `freeze_vision_encoder=True` → 视觉编码器全冻
  - `train_expert_only=True` → 整个 VLM 全冻,只训 AE(+`state_proj`)
  - `train_expert_only=False` → 训 VLM,但冻结最后 1-2 层 + lm_head + norm(DDP 未用参数规避)
- **出厂默认 vs `smolvla_base` 实配**:
  - 代码默认:`load_vlm_weights=False, freeze_vision_encoder=True, train_expert_only=True`(`configuration_smolvla.py:69-85`)
  - `smolvla_base` config.json 实配:`load_vlm_weights=True, freeze_vision_encoder=True, train_expert_only=True` → **加载 stock SmolVLM2 权重 + 冻结 VLM + 只训 AE**

### 1.2 `lerobot/smolvla_libero` 不是论文级模型(已实测)
- `train_config.json`:只在 `libero_spatial` 单 suite 训 **25k 步**,batch 32,数据集 `lerobot/libero`,`control_mode=relative`,`n_action_steps=50`。
- 本机实测(per-task 协议,400 episodes):spatial 55 / object 63 / goal 80 / long 49 / **avg ~62%**;object/goal/long 是分布外。论文 0.45B = 90/96/92/71 / **87.3%**。→ 公开 checkpoint 不可比。

### 1.3 FAST 基础设施(都在 PaliGemma 家族,不是 SmolVLM2)
- `pi0_fast`:PaliGemma-2B + FAST 自回归 token。backbone `paligemma_variant="gemma_2b"`,`action_tokenizer_name="lerobot/fast-action-tokenizer"`。`configuration_pi0_fast.py:31,62-63`
- `pi052`(继承 `pi05` = PaliGemma-2B + Gemma-300M flow expert),**已实现 KI**:
  - `enable_fast_action_loss=True`(`:112`)、`flow_loss_weight=5.0`、`text_loss_weight=1.0`、`fast_action_loss_weight=1.0`
  - `knowledge_insulation`(`:163`,§III.B,monkey-patch `forward` 做 stop-gradient)
  - `auto_fit_fast_tokenizer`:用 `fit_fast_tokenizer.py` 在数据集动作分布上拟合专用 FAST tokenizer(DCT+BPE,quantile 归一)。`pi052/fit_fast_tokenizer.py:81-304`
- FAST tokenizer 拟合工具 `lerobot-train-tokenizer`(`--vocab_size 1024 --scale 10.0 --normalization_mode QUANTILES` 等);通用 tokenizer `physical-intelligence/fast`(100 万真实轨迹拟合,接近专用 tokenizer 性能)。

### 1.4 训练入口
- `lerobot-train`,`--policy.type=<...>` 从零、`--policy.path=<ckpt>` 从 checkpoint(默认 `strict=True` 全量加载,**不支持部分权重加载**)。`lerobot_train.py`;`--resume` 续训会恢复 optimizer/scheduler/RNG。
- 评测复用本仓库 `scripts/run_libero_persuite_eval.sh`(per-task 隔离,规避 MuJoCo EGL 多上下文段错误)。

---

## 2. 文献证据(决定设计的几条)

| 主题 | 结论 | 来源 |
|---|---|---|
| **FAST 原理** | DCT(逐维)+ 量化 + BPE,约 10× 压缩;解决高频控制下 AR token 边际信息趋零、模型只学"复制上一动作"的问题 | [arXiv 2501.09747](https://arxiv.org/html/2501.09747v1) |
| **FAST 速度权衡** | AR 解码慢(~750ms/chunk vs flow ~100ms@4090),但训练快 ~5×、收敛步数少 ~3× | 同上 / [LeRobot pi0fast docs](https://huggingface.co/docs/lerobot/pi0fast) |
| **π0.5 两阶段** | Stage1:全离散 token(含动作 FAST)在异构混合数据上预训练;Stage2:换成 flow-matching 动作头 post-train | π0.5 review |
| **KI(关键)** | 单阶段共训:backbone 上做 FAST 离散 token NTP(+VLM 数据),同时训 flow expert,**stop-gradient 阻断 expert→backbone 梯度**。原因:新随机初始化 expert 的梯度会破坏 backbone 表征 | [arXiv 2505.23705](https://arxiv.org/html/2505.23705v1) |
| **KI 结果** | LIBERO-90 **96.0%** vs 普通 π0 85.2%;训练快如 π0-FAST(普通 π0 需 ~7.5× 步数) | 同上 |
| **冻结消融(对你的 baseline/Stage2 的警告)** | "VLM 预训练表征不足以支撑机器人 —— 冻结不工作" → **0%** | KI Fig.4a/8 |
| **遗忘** | VL backbone 是遗忘主因,动作头稳定;预训练 VLA 用 <10% 步数即可恢复 → 动作预训练后的 backbone 表征"黏性"强,冻结它比冻结纯网页 VLM 可信得多 | [Forget-Me-Not](https://continual-vlas.github.io/forget-me-not/) |
| **SmolVLA 预算** | 预训练 200k 步 / global batch 256 / 4 GPU;LIBERO 微调 **100k 步 / batch 64** → 87.3% | [arXiv 2506.01844](https://arxiv.org/html/2506.01844v1) |

**对假设的净判断**:Stage-1(动作 token 预训练 VLM)**有力支持**;Stage-2(硬冻结 + 训全新 expert)**有风险**,是 PI 明确放弃的配置。最强的去风险动作 = **加 KI 共训臂**。

---

## 3. 实验设计(对比臂)

固定:同一数据集(`HuggingFaceVLA/libero`,4 suite 混合)、同一评测协议(per-task,4×10×10=400 episodes)、同一 VLM backbone、尽量同等训练预算(见 §5)。

| 臂 | Stage-1(VLM 预训练) | Stage-2(动作训练) | 对应文献 | 预期 |
|---|---|---|---|---|
| **A0 baseline** | 无(stock 网页预训练 VLM) | 冻结 VLM,训 flow AE | SmolVLA 出厂配方 | SmolVLA 基线(参照 87%-class,取决于预算) |
| **A1(你的主张)** | FAST-AR 全量微调 VLM | **冻结**预训练 VLM,训全新 flow AE | π0.5 Stage 化 + 硬冻结 | **待测**;KI 预测可能偏弱 |
| **A2(强烈建议加)** | — | **KI 共训**:FAST-AR NTP + flow,同时训,stop-gradient 阻断 expert→backbone | KI / π0.5 | KI 预测最优 |
| **(可选)A3** | FAST-AR 全量微调 VLM | **不冻结**,全量微调 VLM+AE | 全量 FT 上界(有遗忘风险) | 上界/对照 |

> A0 vs A1 vs A2 三方对比能干净地回答:**到底是"动作预训练"有用,还是"共训/stop-gradient"才是关键**。这正是 KI 论文的核心论点,你能在 SmolVLA/LIBERO 上独立复现验证。

---

## 4. 两条实现路线(需要你拍板)

### 路线 A —— 用 `pi052`(PaliGemma-2B),省力、有现成代码
`pi052` 已实现 A2(KI 共训)和 A3 的全部机制,通过 flag 即可构造对比臂:
- **A2(KI)**:`--policy.type=pi052 --policy.enable_fast_action_loss=true --policy.knowledge_insulation=true --policy.auto_fit_fast_tokenizer=true`
- **A0 类比**:`pi05`(纯 flow)`--policy.type=pi05 --policy.train_expert_only=true`
- **A1 类比**:`pi052` 先只开 FAST/文本损失预训练 → 存 ckpt → 改 `train_expert_only=true` + 关 FAST 损失续训(注意 `strict=True` 全量加载的限制,见 §6 风险)
- 代价:backbone 是 2B PaliGemma(比 SmolVLM2-500M 大 ~4×),显存/速度更高;但**这是 π0.5/KI 的原生设定,可比性最强**。

### 路线 B —— 坚持 SmolVLM2-500M / smolvla(贴合你原话),需自研
你的原话是"按 smolvla 方式拼接随机 AE",即想保留 SmolVLM2-500M。但:
- **Stage-1 需要给 SmolVLM2 加 FAST-AR 头**(扩词表 + FAST token offset,仿 `pi0_fast` 对 PaliGemma 的做法)——**lerobot 无现成代码**,需新增一个 "smolvlm + fast head" 训练路径(中等工程量)。
- Stage-2:把 Stage-1 微调后的 SmolVLM2 通过 `--policy.vlm_model_name=<本地ckpt> --policy.load_vlm_weights=true` 加载进 smolvla,设 `train_expert_only=true` 冻结、训 AE。**这一步 lerobot 现成支持**(`smolvlm_with_expert.py:88-94` 用 `AutoModelForImageTextToText.from_pretrained(model_id)` 加载任意 HF id/本地路径)。
- 难点:① 自研 FAST-AR 训练 SmolVLM2;② Stage-1 的 SmolVLM2 需用**完整 32 层**预训练,而 smolvla Stage-2 只取前 16 层 —— 预训练目标层与下游取用层不一致需斟酌(建议 Stage-1 也只在前 16 层上接 FAST 头,保持表征一致)。

**建议**:先用**路线 A(pi052)**跑通三臂对比拿到科学结论(动作预训练 / KI 是否有用),若确实有用且你必须要 SmolVLM2-500M 的小体积优势,再投入路线 B 把结论迁移到 smolvla。

---

## 5. 训练预算与评测

- **预算(单卡 4090,48GB)**:参照 SmolVLA LIBERO 微调 100k 步 / batch 64。4090 单卡 batch 64 对 0.45B 可行;pi052(2B)需降 batch(16-32)+ 梯度累积,步数相应增加。先各臂跑 **短 pilot(~20k 步)**比趋势,再对最优臂跑满。
- **数据**:`HuggingFaceVLA/libero`(35GB,下载中 → `~/.cache/huggingface/lerobot/hub/`)。
- **评测**:`scripts/run_libero_persuite_eval.sh`(改 `--policy.path`),per-task 隔离,400 episodes,产出 4 suite + avg,与论文表对照。
- **指标**:per-suite pc_success、avg;另记训练 wall-clock / GPU-h(FAST 训快是其卖点之一,值得量化)。

---

## 6. 风险与决策点

1. **`strict=True` 全量加载**:`lerobot-train --policy.path` 不支持部分权重加载。A1 的"Stage1 ckpt → 换头续训"可能需要小补丁(参考 `pi052` 的 `_restore_pi052_pretrained_state` 部分加载模式),否则换 expert 维度/结构会 load 失败。**需先验证**。
2. **A1 可能偏弱**(KI 0% 消融的警示)—— 这本身是有价值的负结果,但要保证 baseline A0 公平(同预算)。
3. **路线 B 的 Stage-1 自研**是主要工程不确定性;先不要投入,等路线 A 出结论。
4. **显存**:pi052(2B)+ flow + FAST + 文本三损失,4090 单卡需谨慎调 batch / 序列长度。
5. **公平性**:三臂的总训练步数 / 见到的样本数尽量对齐,否则无法归因。

---

## 6.5 已定决策(2026-06-13)
- **路线 = B,但在 smolvla(0.45B)上自研适配 KI,参考 pi052**(不用 pi052 的 2B PaliGemma —— 4090 跑全量实验时间来不及)。
- **加 A2(KI stop-gradient 共训)臂**。
- 即:目标是把 pi052 的 KI 机制(FAST-AR + 文本 token 损失 + stop-gradient + 自动拟合 FAST tokenizer)**移植到 smolvla 的 SmolVLM2-500M backbone 上**。

### 路线B-KI 的核心实现难点(需先解决)
smolvla 截断 VLM 到前 16 层,expert cross-attend 第 16 层特征。要加 FAST-AR token 损失需要一个能在 16 层特征上输出 token 分布的 lm_head,但 SmolVLM2 原 lm_head 配 32 层 → 直接用会 off-distribution。两种解法:
1. 给截断的 16 层接**新的/部分 lm_head**(扩 FAST token 词表,仿 pi0_fast 的 `fast_skip_tokens`),只训该头;
2. token 损失路径**多保留几层 VLM**,与 expert 取特征层解耦。

### 实现状态(分支 `feat/smolvla-ki`)
**阶段1 已完成并验证**(2026-06-13):新建 policy `src/lerobot/policies/smolvla_ki/`(`configuration_smolvla_ki.py` / `modeling_smolvla_ki.py` / `__init__.py`),factory 已注册 `smolvla_ki`。
- 架构 Option A:`SmolVLMWithExpertModelKI` 保留完整 32 层 VLM(`num_vlm_layers=-1`)、expert = `expert_attend_layers` 层、重写 `get_model_layers` 实现"前 N 层连续耦合、其余 None(VLM 独立前向)"。
- KI:`_DetachKV` 包住 expert cross-attn 的 k_proj/v_proj 输入(`.detach()`),含 `.weight/.bias` 代理(父类 forward 要读 `.weight.dtype`)。
- 冒烟验证:全 cross 模式下只对动作 suffix 求导 → `knowledge_insulation=True` 时 VLM 梯度 = **0.0**(隔离),`False` 时 = 2.9e6(耦合)。forward/shape 均通过(expert hidden 720=960×0.75)。
- 已知近似:`self_attn_every_n_layers` 的自注意力层仍混合 VLM/expert KV(部分耦合);主导的 cross-attn 路径已完全隔离。如需严格隔离,阶段2 再把自注意力层的 VLM KV 也 detach。
- 旁注:smolvla 父类 `SmolVLMWithExpertModel.train()` 未 `return self`(上游小瑕疵),勿链式 `.eval()`。

**阶段2 已完成并单元验证**(2026-06-13):
- FAST-AR 损失走 **HF 原生 causal-LM forward**(`self.vlm(inputs_embeds, attention_mask, labels)`,prefix 用 -100 屏蔽),`SmolVLAKIPolicy.forward` 做 `flow_w·flow + fast_w·fast` 融合;`set_requires_grad` 在 KI 下保持 VLM+lm_head 可训。
- FAST tokenizer 用 `lerobot/fast-action-tokenizer`(`physical-intelligence/fast` 在当前 transformers 下 slow→fast 转换失败);`auto_fit_fast_tokenizer=False`。依赖补:`sentencepiece`、`protobuf`(未进 lockfile)。
- **真实权重验证(A2)**:607M params(32 VLM + 16 expert),flow loss 1.37 / fast loss 14.5,反向 → expert grad 3.8e4 + lm_head grad 1.2e3。flow 训 expert、fast 训 lm_head,KI detach 隔离确认。
- 归一化 caveat:喂 FAST 的是 MEAN_STD 归一化动作(非 quantile),后续可重拟合修正。

### 准备好的训练命令(数据下完即用)
**A0 baseline(flow-only,最快验证 pipeline 通)**
```bash
uv run lerobot-train --policy.type=smolvla_ki \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --policy.load_vlm_weights=true --policy.keep_full_vlm=false \
  --policy.knowledge_insulation=false --policy.enable_fast_action_loss=false \
  --policy.train_expert_only=true \
  --dataset.repo_id=HuggingFaceVLA/libero --env.type=libero --env.task=libero_spatial \
  --batch_size=4 --steps=500 --log_freq=10 --eval_freq=100000 --save_freq=100000 \
  --policy.device=cuda --output_dir=./outputs/train/smolvla_ki_A0_smoke
```
**A2 KI(双损失)** — 同上但:`--policy.keep_full_vlm=true --policy.knowledge_insulation=true --policy.enable_fast_action_loss=true --policy.train_expert_only=false`,`--output_dir=...A2_smoke`。
> 注意:A0 用 keep_full_vlm=false(走截断 16 层、近 stock smolvla);A2 用 keep_full_vlm=true(完整 32 层 + FAST 头)。环境变量需带 hp 代理 + `HF_HUB_DOWNLOAD_TIMEOUT=300`。smoke 阶段 `eval_freq` 设很大跳过 env eval,只看 train loss 下降。

### 待落地的代码改动(参考 pi052)
- `forward` 里把 VLM→expert 的特征 `.detach()`(stop-gradient,= pi052 `knowledge_insulation` 的 monkey-patch)。
- 给 SmolVLM2 加 FAST token CE 损失头 + 损失项(参考 `pi052/configuration_pi052.py` 的 `enable_fast_action_loss/flow_loss_weight/text_loss_weight`)。
- 复用 `pi052/fit_fast_tokenizer.py` 在 `HuggingFaceVLA/libero` 动作分布上拟合 FAST tokenizer。
- 三臂总训练步数/样本数对齐以保证可归因。

## 7. 待你拍板的问题
1. **路线 A(pi052/PaliGemma,省力有据)还是路线 B(SmolVLM2 自研,贴合原意)?** 建议先 A。
2. 是否接受**加 A2(KI 共训)臂**?(强烈建议,否则实验缺最关键对照)
3. Pilot 预算:各臂先 20k 步比趋势 OK 吗?
4. 评测用全 4 suite 400 episodes,还是 pilot 阶段先 spatial+long 两 suite 省时间?

---

## 附:已核实的关键命令片段(待路线确定后细化)
```bash
# 评测(任意 checkpoint):改 --policy.path 重跑
bash scripts/run_libero_persuite_eval.sh   # 内部 per-task 隔离

# 路线A - A2(KI 共训)pilot 示例(flag 名已核实,值待调)
lerobot-train --policy.type=pi052 \
  --policy.enable_fast_action_loss=true \
  --policy.knowledge_insulation=true \
  --policy.auto_fit_fast_tokenizer=true \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --batch_size=16 --steps=20000 --output_dir=./outputs/train/A2_ki_pilot

# 路线B - Stage2:把自研预训练的 SmolVLM2 加载进 smolvla 并冻结训 AE
lerobot-train --policy.type=smolvla \
  --policy.vlm_model_name=<本地预训练SmolVLM2路径> \
  --policy.load_vlm_weights=true \
  --policy.freeze_vision_encoder=true --policy.train_expert_only=true \
  --dataset.repo_id=HuggingFaceVLA/libero ...
```
