# LingBot-VLA 2.0

LingBot-VLA 2.0 is Robbyant's vision-language-action model, the successor to LingBot-VLA 1.0. It pairs a
Qwen3-VL-4B backbone with a sparse Mixture-of-Experts action expert and generates action chunks with flow
matching over a 55-D canonical state/action space shared across embodiments.

```bash
pip install -e ".[training,lingbot_vla_v2]"
```

Fine-tune the base checkpoint by mapping your robot's state and action dims onto the canonical slots:

```bash
lerobot-train \
  --dataset.repo_id=your_dataset \
  --policy.path=lerobot/lingbot_vla_v2_base \
  --policy.state_slots='{"observation.state.arm.position": {"origin_keys": [{"observation.state": {"start": 0, "end": 6}}]}}' \
  --policy.action_slots='{"action.arm.position": {"origin_keys": [{"action": {"start": 0, "end": 6}}]}}' \
  --rename_map='{"observation.images.front": "observation.images.camera_top"}' \
  --policy.repo_id=your_repo_id
```

See the [LingBot-VLA 2.0 guide](https://huggingface.co/docs/lerobot/lingbot_vla_v2) for slot mappings,
cameras, RoboTwin evaluation and inference speed-ups.

```bibtex
@article{lingbotvla2,
      title={From Foundation to Application: Improving VLA Models in Practice},
      author={Wei Wu and Fangjing Wang and Fan Lu and He Sun and Shi Liu and Yunnan Wang and Yibin Yan and Yong Wang and Shuailei Ma and Xinyang Wang and Yibin Liu and Shuai Yang and Tianxiang Zhou and Kejia Zhang and Lei Zhou and Cheng Su and Nan Xue and Bin Tan and Han Zhang and Youchao Zhang and Fei Liao and Xing Zhu and Yujun Shen and Kecheng Zheng},
      journal={arXiv preprint arXiv:2607.06403},
      year={2026}
}
```
