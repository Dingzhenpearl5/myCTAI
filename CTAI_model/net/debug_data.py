"""数据加载调试脚本 - 检查数据分布"""
import sys
sys.path.append("..")
import torch
from data_set import make
import numpy as np

data_path = 'C:/Users/Masoa/OneDrive/work/CTAI/src/data'
train_dataset, test_dataset = make.get_d1(data_path)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 检查第一个样本
x, y = train_dataset[0]
image = x[0]
mask = y[1]

print(f"\n=== 第一个训练样本 ===")
print(f"图像形状: {image.shape}")
print(f"图像范围: min={image.min():.4f}, max={image.max():.4f}, mean={image.mean():.4f}")
print(f"\nMask形状: {mask.shape}")
print(f"Mask范围: min={mask.min():.4f}, max={mask.max():.4f}, mean={mask.mean():.4f}")
print(f"Mask唯一值: {torch.unique(mask)}")
print(f"Mask非零像素数: {(mask > 0).sum().item()}/{mask.numel()}")
print(f"Mask非零像素比例: {(mask > 0).sum().item() / mask.numel() * 100:.2f}%")

# 检查多个样本
print("\n=== 前10个样本的Mask统计 ===")
for i in range(min(10, len(train_dataset))):
    x, y = train_dataset[i]
    mask = y[1]
    positive_ratio = (mask > 0).sum().item() / mask.numel() * 100
    print(f"样本 {i}: 非零像素比例 {positive_ratio:.2f}%, unique值={torch.unique(mask).tolist()}")
