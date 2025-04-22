#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import random

def plot_trajectory_comparison(gt_trajectory, pred_trajectory, title="Trajectory Comparison", save_path=None):
    """
    绘制真实轨迹和预测轨迹的3D对比图（仅XYZ）
    
    Args:
        gt_trajectory (torch.Tensor): 真实轨迹，形状为 [T, 3] 或 [B, T, 3]
        pred_trajectory (torch.Tensor): 预测轨迹，形状为 [T, 3] 或 [B, T, 3]
        title (str): 图表标题
        save_path (str): 保存图片的路径，如果为None则不保存
    """
    # 确保输入是numpy数组
    if isinstance(gt_trajectory, torch.Tensor):
        gt_trajectory = gt_trajectory.detach().cpu().numpy()
    if isinstance(pred_trajectory, torch.Tensor):
        pred_trajectory = pred_trajectory.detach().cpu().numpy()
    
    # 如果输入是批次数据，只取第一个样本
    if len(gt_trajectory.shape) == 3:
        gt_trajectory = gt_trajectory[0]
    if len(pred_trajectory.shape) == 3:
        pred_trajectory = pred_trajectory[0]
    
    # 只取XYZ坐标（前3个维度）
    gt_trajectory = gt_trajectory[:, :3]
    pred_trajectory = pred_trajectory[:, :3]
    
    # 创建3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制轨迹
    ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2], 
            'b-', label='Ground Truth', linewidth=2)
    ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2], 
            'r--', label='Predicted', linewidth=2)
    
    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 设置标题和图例
    ax.set_title(title)
    ax.legend()
    
    # 设置视角
    ax.view_init(elev=20, azim=45)
    
    # 自动调整坐标轴范围
    all_points = np.vstack([gt_trajectory, pred_trajectory])
    min_val = np.min(all_points, axis=0)
    max_val = np.max(all_points, axis=0)
    range_val = max_val - min_val
    padding = 0.1 * range_val
    
    ax.set_xlim(min_val[0] - padding[0], max_val[0] + padding[0])
    ax.set_ylim(min_val[1] - padding[1], max_val[1] + padding[1])
    ax.set_zlim(min_val[2] - padding[2], max_val[2] + padding[2])
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_epoch_trajectories(gt_trajectories, pred_trajectories, save_dir, epoch, num_samples=3):
    """
    为每个epoch绘制指定数量的轨迹对比图
    
    Args:
        gt_trajectories (torch.Tensor): 真实轨迹，形状为 [B, T, D]
        pred_trajectories (torch.Tensor): 预测轨迹，形状为 [B, T, D]
        save_dir (str): 保存图片的目录
        epoch (int): 当前epoch
        num_samples (int): 要绘制的样本数量
    """
    # 确保输入是numpy数组
    if isinstance(gt_trajectories, torch.Tensor):
        gt_trajectories = gt_trajectories.detach().cpu().numpy()
    if isinstance(pred_trajectories, torch.Tensor):
        pred_trajectories = pred_trajectories.detach().cpu().numpy()
    
    # 随机选择指定数量的样本
    batch_size = len(gt_trajectories)
    selected_indices = random.sample(range(batch_size), min(num_samples, batch_size))
    
    # 绘制选中的样本
    for i, idx in enumerate(selected_indices):
        save_path = f"{save_dir}/trajectory_epoch_{epoch}_sample_{i}.png"
        plot_trajectory_comparison(
            gt_trajectories[idx],
            pred_trajectories[idx],
            title=f"Trajectory Comparison (Epoch {epoch}, Sample {i})",
            save_path=save_path
        ) 