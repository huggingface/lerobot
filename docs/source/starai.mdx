# 如何搭建具身智能LeRobot-starai系列机械臂并完成自定义抓取任务



##  产品介绍

1. **开源 & 便于二次开发**
   本系列舵机由[华馨京科技](https://fashionrobo.com/)提供，是一套开源、便于二次开发的6+1自由度机器臂解决方案。
2. **支持 LeRobot 平台集成**
   专为与 [LeRobot 平台](https://github.com/huggingface/lerobot) 集成而设计。该平台提供 PyTorch 模型、数据集与工具，面向现实机器人任务的模仿学习（包括数据采集、仿真、训练与部署）。
3. **丰富的学习资源**
   提供全面的开源学习资源，包括环境搭建，安装与调试与自定义夹取任务案例帮助用户快速上手并开发机器人应用。
4. **兼容 Nvidia 平台**
   支持通过 reComputer Mini J4012 Orin NX 16GB 平台进行部署。



## 特点内容

- **零组装**:  即刻上手｜一开箱即踏入AI时代。
- 6+1自由度结构设计，470mm臂展，赋予无限操作可能。
- 配备2颗全金属无刷总线舵机，稳定驱动，轻松承重300g。
- 智能平行夹爪，最大开合66mm，模块化指尖，精准抓取不设限。
- 独家悬停控制系统，指尖一按，Leader Arm稳停于任意姿态。



## 规格参数

![image-20250709072845215](../../media/starai/1-114090080-fashionstar-star-arm-cello-violin.jpg)

| Item                 | Follower Arm \| Viola                             | Leder Arm \|Violin                                |
| -------------------- | ------------------------------------------------- | ------------------------------------------------- |
| Degrees of Freedom   | 6+1                                                 | 6+1                                               |
| Reach                | 470mm                                             | 470mm                                             |
| Span                 | 940mm                                             | 940mm                                             |
| Repeatability        | 2mm                                               | -                                                 |
| Working Payload      | 300g (with 70% Reach)                            | -                                                 |
| Servos               | RX8-U50H-M x2<br/>RA8-U25H-M x4<br/>RA8-U26H-M x1 | RX8-U50H-M x2<br/>RA8-U25H-M x4<br/>RA8-U26H-M x1 |
| Parallel Gripper Ki  | √                                                 | -                                                 |
| Wrist Rotate         | Yes                                               | Yes                                               |
| Hold at any Position | Yes                                               | Yes (with handle button)                          |
| Wrist Camera Mount   | √                                                 | -                                                 |
| Works with LeRobot   | √                                                 | √                                                 |
| Works with ROS 2     | √                                                 | /                                                 |
| Works with MoveIt    | √                                                 | /                                                 |
| Works with Gazebo    | √                                                 | /                                                 |
| Communication Hub    | UC-01                                             | UC-01                                             |
| Power Supply         | 12v/120w                                          | 12v/120w                                          |

有关舵机更多资讯，请访问以下链接。

[RA8-U25H-M](https://fashionrobo.com/actuator-u25/23396/)

[RX18-U100H-M](https://fashionrobo.com/actuator-u100/22853/)

[RX8-U50H-M](https://fashionrobo.com/actuator-u50/136/)







## 初始环境搭建

For Ubuntu X86:

- Ubuntu 22.04
- CUDA 12+
- Python 3.10
- Troch 2.6

## 安装与调试

### 安装LeRobot

需要根据你的 CUDA 版本安装 pytorch 和 torchvision 等环境。

1. 安装 Miniconda： 对于 Jetson：
    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
    chmod +x Miniconda3-latest-Linux-aarch64.sh
    ./Miniconda3-latest-Linux-aarch64.sh
    source ~/.bashrc
    ```
	或者，对于 X86 Ubuntu 22.04：

    ```bash
    mkdir -p ~/miniconda3
    cd miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    source ~/miniconda3/bin/activate
    conda init --all
    ```

2. 创建并激活一个新的 conda 环境用于 LeRobot

    ```bash
    conda create -y -n lerobot python=3.10 && conda activate lerobot
    ```

3. 克隆 LeRobot 仓库：

    ```bash
    git clone https://github.com/servodevelop/lerobot.git
    ```
	切换到develop分支

4. 使用 miniconda 时，在环境中安装 ffmpeg：

    ```bash
    conda install ffmpeg -c conda-forge
    ```
    这通常会为你的平台安装使用 libsvtav1 编码器编译的 ffmpeg 7.X。如果不支持 libsvtav1（可以通过 ffmpeg -encoders 查看支持的编码器），你可以：

    - 【适用于所有平台】显式安装 ffmpeg 7.X：

    ```bash
    conda install ffmpeg=7.1.1 -c conda-forge
    ```

5. 安装带有 fashionstar 电机依赖的 LeRobot：

    ```bash
    cd ~/lerobot && pip install -e ".[starai]"
    ```
6. 检查 Pytorch 和 Torchvision

    由于通过 pip 安装 LeRobot 环境时会卸载原有的 Pytorch 和 Torchvision 并安装 CPU 版本，因此需要在 Python 中进行检查。

    ```python
    import torch
    print(torch.cuda.is_available())
    ```

    如果输出结果为 False，需要根据[官网教程](https://pytorch.org/index.html)重新安装 Pytorch 和 Torchvision。

### 接线

https://github.com/user-attachments/assets/56130bd9-21ee-4ae4-9cac-3817ac4d659f



### 手臂端口设置

在终端输入以下指令来找到两个机械臂对应的端口号：

```bash
lerobot-find-port
```

例如：

1. 识别Leader时端口的示例输出（例如，在 Mac 上为 `/dev/tty.usbmodem575E0031751`，或在 Linux 上可能为 `/dev/ttyUSB0`） 
2. 识别Follower时端口的示例输出（例如，在 Mac 上为 `/dev/tty.usbmodem575E0032081`，或在 Linux 上可能为 `/dev/ttyUSB1`）

> [!NOTE]
>
> 如果识别不到ttyUSB0串口信息。尝试以下方法。
>
> 列出所有usb口。
>
> ```sh
> lsusb
> ```
>
> <img src="./../../media/starai/image-20241230112928879-1749511998299-1.png" alt="image-20241230112928879-1749511998299-1" style="zoom:80%;" />
>
> 识别成功，查看ttyusb的信息
>
> ```sh
> sudo dmesg | grep ttyUSB
> ```
>
> <img src="./../../media/starai/image-20241230113058856-1749512093309-2.png" alt="image-20241230113058856" style="zoom:80%;" />
>
> 最后一行显示断连，因为brltty在占用该USB设备号，移除掉就可以了
>
> ```sh
> sudo apt remove brltty
> ```
>
> <img src="./../../media/starai/image-20241230113211143-1749512102599-4.png" alt="image-20241230113211143" style="zoom: 80%;" />
>
> 最后，赋予权限
>
> ```sh
> sudo chmod 777 /dev/ttyUSB*
> ```
>



## 校准

如果是第一次校准，请对每个关节左右转动到对应位置。

如果是重新校准，按照命令提示输入字母c后按Enter键。

下面是参考值,通常情况下，真实的限位参考值的±10°范围内。

| 舵机ID  | 角度下限参考值 | 角度上限参考值 | 备注                               |
| ------- | -------------: | -------------: | ---------------------------------- |
| motor_0 |          -180° |           180° | 转动到限位处                       |
| motor_1 |           -90° |            90° | 转动到限位处                       |
| motor_2 |           -90° |            90° | 转动到限位处                       |
| motor_3 |          -180° |           180° | 没有限位，需转动到角度上下限参考值 |
| motor_4 |           -90° |            90° | 转动到限位处                       |
| motor_5 |          -180° |           180° | 没有限位，需转动到角度上下限参考值 |
| motor_6 |             0° |           100° | 转动到限位处                       |

### leader

> [!TIP]
>
> 将leader连接到/dev/ttyUSB0，或者修改下面的命令。



```bash
lerobot-calibrate     --teleop.type=starai_violin --teleop.port=/dev/ttyUSB0 --teleop.id=my_awesome_staraiviolin_arm
```

### follower

> [!TIP]
>
> 将follower连接到/dev/ttyUSB1，或者修改下面的命令。

```bash
lerobot-calibrate     --robot.type=starai_viola --robot.port=/dev/ttyUSB1 --robot.id=my_awesome_staraiviola_arm
```

## 遥操作


https://github.com/user-attachments/assets/23b3aa00-9889-48d3-ae2c-00ad50595e0a


将手臂移动至图上位置待机。

![image-20250717064511074](media/image-20250717064511074.png)



您已准备好遥操作您的机器人（不包括摄像头）！运行以下简单脚本：

```bash
lerobot-teleoperate \
    --robot.type=starai_viola \
    --robot.port=/dev/ttyUSB1 \
    --robot.id=my_awesome_staraiviola_arm \
    --teleop.type=starai_violin \
    --teleop.port=/dev/ttyUSB0 \
    --teleop.id=my_awesome_staraiviolin_arm
```
远程操作命令将自动检测下列参数:

1. 识别任何缺失的校准并启动校准程序。
2. 连接机器人和远程操作设备并开始远程操作。



程序启动后，悬停按钮依旧生效。



## 添加摄像头

https://github.com/user-attachments/assets/82650b56-96be-4151-9260-2ed6ab8b133f


在插入您的两个 USB 摄像头后，运行以下脚本以检查摄像头的端口号，切记摄像头避免插在USB Hub上，USB Hub速率太慢会导致读不到图像数据。

```bash
lerobot-find-cameras opencv # or realsense for Intel Realsense cameras
```

终端将打印出以下信息。以我的笔记本为例，笔记本摄像头为Camera0和Camera1，index_or_path分别为2和4。

```markdown
--- Detected Cameras ---
Camera #0:
  Name: OpenCV Camera @ /dev/video2
  Type: OpenCV
  Id: /dev/video2
  Backend api: V4L2
  Default stream profile:
    Format: 0.0
    Width: 640
    Height: 480
    Fps: 30.0
--------------------
Camera #1:
  Name: OpenCV Camera @ /dev/video4
  Type: OpenCV
  Id: /dev/video4
  Backend api: V4L2
  Default stream profile:
    Format: 0.0
    Width: 640
    Height: 360
    Fps: 30.0
--------------------

Finalizing image saving...
Image capture finished. Images saved to outputs/captured_images
```


确认外接摄像头后，将摄像头信息替换下方cameras信息您将能够在遥操作时在计算机上显示摄像头：

```bash
lerobot-teleoperate \
    --robot.type=starai_viola \
    --robot.port=/dev/ttyUSB1 \
    --robot.id=my_awesome_staraiviola_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --teleop.type=starai_violin \
    --teleop.port=/dev/ttyUSB0 \
    --teleop.id=my_awesome_staraiviolin_arm \
    --display_data=true
    
```

## 数据集制作采集


https://github.com/user-attachments/assets/8bb25714-783a-4f29-83dd-58b457aed80c

> [!TIP]
>
> 如果您想使用 Hugging Face Hub 的功能来上传您的数据集，请参考以下官方文章链接。本教程将不涉及这部分内容。
>
> [lmitation Learning for Robots](https://huggingface.co/docs/lerobot/il_robots?teleoperate_koch_camera=Command)
>
> 


一旦您熟悉了遥操作，您就可以开始您的第一个数据集。

记录 10 个回合：

```bash
lerobot-record \
    --robot.type=starai_viola \
    --robot.port=/dev/ttyUSB1 \
    --robot.id=my_awesome_staraiviola_arm \
    --robot.cameras="{ up: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30},front: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}}" \
    --teleop.type=starai_violin \
    --teleop.port=/dev/ttyUSB0 \
    --teleop.id=my_awesome_staraiviolin_arm \
    --display_data=true \
    --dataset.repo_id=starai/record-test \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=30 \
    --dataset.num_episodes=10 \
    --dataset.push_to_hub=False \
    --dataset.single_task="Grab the black cube"
```



### 记录功能

- record提供了一套用于在机器人操作过程中捕获和管理数据的工具:

#### 1.数据存储

- 数据使用 `LeRobotDataset` 格式存储，并在录制过程中存储在磁盘上。

#### 2.检查点和恢复

- 记录期间会自动创建检查点。
- 如果出现问题，可以通过使用 重新运行相同的命令来恢复。恢复录制时，必须设置为**要录制的额外剧集数**，而不是数据集中的目标总剧集数！`--resume=true` `--dataset.num_episodes`
- 要从头开始录制，请**手动删除**数据集目录。

#### 3.记录 参数 

使用命令行参数设置数据记录流：

--dataset.episode_time_s=60每个数据记录插曲的持续时间(默认值:60秒)。
--dataset.reset_time_s=60每集后重置环境的持续时间(默认:60秒)。
--dataset.num_episodes=50记录的总集数(默认值:50秒)。



#### 4.录制期间的键盘控制

使用键盘快捷键控制数据记录流：

- 按**右方向键(→)** ： 提前停止当前情节或重置时间,然后移动到下一个。

- 按**左方向键(←)** ：取消当前插曲并重新录制。
- 按**ESC**：立即停止会话,编码视频并上传数据集。



>[!TIP]
>
>在 Linux 上,如果左右箭头键和转义键在数据记录过程中没有任何效果,请确保已设置$DISPLAY环境变量。参见 pynput 限制。
>
>一旦你熟悉了数据记录,你就可以创建一个更大的数据集进行训练。一个好的开始任务是抓住一个物体在不同的位置,并把它放在一个垃圾箱。我们建议录制至少50集,每个地点10集。保持相机固定,并在整个录音中保持一致的抓握行为。还要确保你操纵的对象在相机上可见。一个好的经验法则是,你应该能够只看相机图像自己完成任务。







## 重播一个回合

播放已经录制好的动作，可以借此测试机器人动作的重复性。

```bash
lerobot-replay \
    --robot.type=starai_viola \
    --robot.port=/dev/ttyUSB1 \
    --robot.id=my_awesome_staraiviola_arm \
    --dataset.repo_id=starai/record-test \
    --dataset.episode=1 # choose the episode you want to replay
```



## 训练

要训练一个控制您机器人策略，以下是一个示例命令：

```bash
lerobot-train \
  --dataset.repo_id=starai/record-test \
  --policy.type=act \
  --output_dir=outputs/train/act_viola_test \
  --job_name=act_viola_test \
  --policy.device=cuda \
  --wandb.enable=False \
  --policy.repo_id=starai/my_policy
```

1. 我们提供了数据集作为参数。`dataset.repo_id=starai/record-test`
2. 我们为 .这将从 [`configuration_act.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py) 加载配置。重要的是，此策略将自动适应机器人的电机状态、电机动作和相机的数量已保存在您的数据集中。`policy.type=act` `laptop` `phone`
3. 我们提供了使用[权重和偏差](https://docs.wandb.ai/quickstart)来可视化训练图。这是可选的，但如果您使用它，请确保您已通过运行 登录。`wandb.enable=true` `wandb login`



要从某个检查点恢复训练。

```bash
lerobot-train \
  --config_path=outputs/train/act_viola_test/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

## 运行推理并评估

运行以下命令记录 10 个评估回合：

```bash
lerobot-record  \
  --robot.type=starai_viola \
  --robot.port=/dev/ttyUSB1 \
  --robot.cameras="{ up: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30},front: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}}" \
  --robot.id=my_awesome_staraiviola_arm \
  --display_data=false \
  --dataset.repo_id=starai/eval_record-test \
  --dataset.single_task="Put lego brick into the transparent box" \
  --policy.path=outputs/train/act_viola_test/checkpoints/last/pretrained_model
  # <- Teleop optional if you want to teleoperate in between episodes \
  # --teleop.type=starai_violin \
  # --teleop.port=/dev/ttyUSB0 \
  # --teleop.id=my_awesome_leader_arm \
```



## FAQ

- 如果使用本文档教程，请git clone本文档推荐的github仓库`https://github.com/servodevelop/lerobot.git`。

- 如果遥操作正常，而带Camera的遥操作无法显示图像界面，请参考[这里](https://github.com/huggingface/lerobot/pull/757/files)

- 如果在数据集遥操作过程中出现libtiff的问题，请更新libtiff版本。

  ```bash
  conda install libtiff==4.5.0  #for Ubuntu 22.04 is libtiff==4.5.1
  ```

  

- 执行完安装LeRobot可能会自动卸载gpu版本的pytorch，所以需要在手动安装torch-gpu。

- 对于Jetson，请先安装[Pytorch和Torchvsion](https://github.com/Seeed-Projects/reComputer-Jetson-for-Beginners/blob/main/3-Basic-Tools-and-Getting-Started/3.3-Pytorch-and-Tensorflow/README.md#installing-pytorch-on-recomputer-nvidia-jetson)再执行`conda install -y -c conda-forge ffmpeg`,否则编译torchvision的时候会出现ffmpeg版本不匹配的问题。

- 在3060的8G笔记本上训练ACT的50组数据的时间大概为6小时，在4090和A100的电脑上训练50组数据时间大概为2~3小时。

- 数据采集过程中要确保摄像头位置和角度和环境光线的稳定，并且减少摄像头采集到过多的不稳定背景和行人，否则部署的环境变化过大会导致机械臂无法正常抓取。

- 数据采集命令的num-episodes要确保采集数据足够，不可中途手动暂停，因为在数据采集结束后才会计算数据的均值和方差，这在训练中是必要的数据。

- 如果程序提示无法读取USB摄像头图像数据，请确保USB摄像头不是接在Hub上的，USB摄像头必须直接接入设备，确保图像传输速率快。

## 参考文档

矽递科技英文Wiki文档：[How to use the SO10xArm robotic arm in Lerobot | Seeed Studio Wiki]([如何在 Lerobot 中使用 SO100/101Arm 机器人手臂 | Seeed Studio Wiki](https://wiki.seeedstudio.com/cn/lerobot_so100m/))

Huggingface Project:[Lerobot](https://github.com/huggingface/lerobot/tree/main)

Huggingface:[LeRobot](https://huggingface.co/docs/lerobot/index)

ACT or ALOHA:[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://tonyzhaozh.github.io/aloha/)

VQ-BeT:[VQ-BeT: Behavior Generation with Latent Actions](https://sjlee.cc/vq-bet/)

Diffusion Policy:[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)

TD-MPC:[TD-MPC](https://www.nicklashansen.com/td-mpc/)



