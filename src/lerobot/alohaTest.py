# alohaTest.py
import os
import sys
# 如果源码放在项目根目录的 src/ 下，可以临时将 src 添加到路径
# sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.visualize_dataset import visualize_dataset


def main():
    # 创建 LeRobotDataset 实例，指定仓库 ID 和使用的解码后端
    dataset = LeRobotDataset(
        repo_id="lerobot/aloha_static_coffee",  # Hugging Face 上的数据集名称
        video_backend="pyav",                   # 强制使用 pyav 作为视频解码后端
    )

    # 调用可视化脚本函数，禁用本地 Rerun Viewer 启动，改为使用 Web 界面
    visualize_dataset(
        dataset,
        episode_index=0,
        batch_size=1,
        num_workers=0
        # web_port=8080, ws_port=8081,  # 自定义 WebSocket 端口
        # save=True,                     # 如果想将渲染结果保存到磁盘
    )

    # 运行后，终端会输出一个 URL，打开即可查看可视化内容


if __name__ == "__main__":
    main()