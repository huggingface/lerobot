# Norm-stats CICD 一致性验证

端到端验证 `lingbot_vla_v2` 的归一化统计计算与上游 LingBot-VLA 2.0 的实现一致。
**当前状态:13/13 通过**(2026-09-01)。详细报告见 `platform/20-training/docs/lingbot-norm-stats-parity-report.md`。

两个测试文件,均纯 CPU、秒级:

| 文件 | 覆盖 | 说明 |
|---|---|---|
| `tests/policies/lingbot_vla_v2/test_norm_stats_parity.py` | **golden-vector 对照** | 同种子同批数据,移植版 `RunningStats` vs 上游 `lingbotvla.utils.normalize.RunningStats` 逐位比对:mean/std/min/max 要求 diff<1e-10,分位数(5000-bin 直方图)要求一致到 1e-12;另测 merge / state roundtrip / chunk reshape |
| `tests/policies/lingbot_vla_v2/test_norm_stats_pipeline.py` | **端到端管线** | synthetic 数据集走 `FeatureTransform(do_normalize=False, return_item_before_padding=True, disabled_image_features=True, processor=None)` → `RunningStats` → `save`;验证 ① 图像无关路径 ② 统计与 numpy 真值一致 ③ subtract_state 相对动作语义 ④ 产出被 `Normalizer` 正确消费并往返无损 |

上游对照测试需要本机有上游仓 checkout(`/home/nvidia/platform/20-training/upstream/lingbot-vla-v2-upstream` 或 `~/lingbot-vla-v2-upstream`);缺了自动 skip,其余测试照常跑。

```bash
# 快速(只跑这两个文件)
pytest tests/policies/lingbot_vla_v2/test_norm_stats_parity.py \
       tests/policies/lingbot_vla_v2/test_norm_stats_pipeline.py -q

# 只要端到端管线(不依赖上游 checkout)
pytest tests/policies/lingbot_vla_v2/test_norm_stats_pipeline.py -q
```
