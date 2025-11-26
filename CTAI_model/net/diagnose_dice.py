"""诊断 Dice = 0 的问题"""
import sys
sys.path.append("..")
import torch
import numpy as np
from data_set import make
from utils import dice_loss

print("=" * 60)
print("数据加载诊断")
print("=" * 60)

data_path = 'C:/Users/Masoa/OneDrive/work/CTAI/src/train'
train_dataset, test_dataset = make.get_d1(data_path)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 检查第一个训练样本
x, y = train_dataset[0]
image = x[0]  # [C, H, W]
mask = y[1]   # [H, W]

print(f"\n第一个样本:")
print(f"  图像 shape: {image.shape}")
print(f"  Mask shape: {mask.shape}")
print(f"  Mask unique值: {torch.unique(mask).tolist()}")
print(f"  Mask范围: [{mask.min():.4f}, {mask.max():.4f}]")
print(f"  Mask非零像素: {(mask > 0).sum().item()}/{mask.numel()}")
print(f"  肿瘤像素比例: {(mask > 0).sum().item() / mask.numel() * 100:.2f}%")

# 模拟 Dice 计算
mask_np = mask.numpy()
print(f"\n模拟 Dice 计算:")
print(f"  Mask (numpy): shape={mask_np.shape}, sum={mask_np.sum()}")

# 创建一个全0的预测（模拟当前情况）
pred_zeros = np.zeros_like(mask_np)
dice_zeros = dice_loss.dice(pred_zeros, mask_np)
print(f"  全0预测的 Dice: {dice_zeros}")

# 创建一个与mask相同的预测
pred_perfect = mask_np.copy()
dice_perfect = dice_loss.dice(pred_perfect, mask_np)
print(f"  完美预测的 Dice: {dice_perfect}")

#创建一个随机预测
np.random.seed(42)
pred_random = (np.random.random(mask_np.shape) > 0.5).astype(float)
dice_random = dice_loss.dice(pred_random, mask_np)
print(f"  随机预测的 Dice: {dice_random}")

print(f"\n检查多个样本:")
for i in range(min(5, len(train_dataset))):
    x, y = train_dataset[i]
    mask = y[1].numpy()
    pos_ratio = (mask > 0).sum() / mask.size * 100
    unique_vals = np.unique(mask)
    print(f"样本{i}: 肿瘤比例={pos_ratio:.2f}%, unique值={unique_vals.tolist()}, sum={mask.sum():.0f}")
