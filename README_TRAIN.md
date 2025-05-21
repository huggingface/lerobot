# 0. 系统安装
    1. 运行程序需要切换到lerobot的conda环境：
        conda activate lerobot

# 1. 数据转换v1.6 -> v2.0
1. 首先确保数据格式已经转换至lerobot 1.6版本，再使用convert_raw_to_v2.0.py脚本转换至PI0可以用的2.0版本
2. convert_raw_to_v2.0.py在~/project/lerobot/lerobot/common/datasets路径中，使用以下command执行转换步骤：
```bash
python lerobot/common/datasets/convert_raw_to_v2.0.py  --input-dir /data/TR2/hugging_face/pick_and_place_0124_rf10 --output-dir /data/TR2/hugging_face/pick_place_0124_rf10_new
```

注意将input-dir以及output-dir换为实际路径, 如果有多个input dir需要多任务合并数据，直接在--input-dir后写上所有需要合并的数据路径，并在~/project/lerobot/lerobot/common/datasets路径中的task_map.json文件中按照--input-dir的顺序填写task idx，对应的episode range以及任务描述。episode range可以从原始数据文件夹中的videos中查看每个数据集的episode并手动计算episode range
    
# 2. 训练步骤

# 3. 推理步骤