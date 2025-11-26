"""
诊断TransUNet训练问题
"""
import sys
sys.path.append("..")

import torch
from data_set import make
from net import transunet
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*60)
print("TransUNet诊断")
print("="*60)

# 加载数据
print("\n1. 加载数据集...")
train_dataset, _, _ = make.get_d1('C:/Users/Masoa/OneDrive/work/CTAI/src/train')
print(f"训练集大小: {len(train_dataset)}")

# 获取一个样本
x, mask = train_dataset[0]
images = x[0].unsqueeze(0).to(device)  # (1, 1, 512, 512)
targets = mask[1].unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 512, 512)

print(f"输入shape: {images.shape}")
print(f"目标shape: {targets.shape}")
print(f"目标值范围: [{targets.min():.3f}, {targets.max():.3f}]")
print(f"目标中有肿瘤的像素: {targets.sum().item():.0f} / {targets.numel()}")

# 初始化模型
print("\n2. 初始化TransUNet...")
model = transunet.TransUNet(
    img_size=512,
    patch_size=16,
    in_channels=1,
    out_channels=1,
    embed_dim=768,
    depth=12,
    num_heads=12
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params / 1e6:.2f}M")

# 前向传播
print("\n3. 前向传播测试...")
model.eval()
with torch.no_grad():
    outputs = model(images)

print(f"输出shape: {outputs.shape}")
print(f"输出值范围: [{outputs.min():.6f}, {outputs.max():.6f}]")
print(f"输出均值: {outputs.mean():.6f}")
print(f"输出std: {outputs.std():.6f}")

# 检查输出分布
output_np = outputs.cpu().numpy().flatten()
print(f"\n输出值分布:")
print(f"  < 0.1: {(output_np < 0.1).sum()} ({(output_np < 0.1).sum() / len(output_np) * 100:.1f}%)")
print(f"  0.1-0.3: {((output_np >= 0.1) & (output_np < 0.3)).sum()} ({((output_np >= 0.1) & (output_np < 0.3)).sum() / len(output_np) * 100:.1f}%)")
print(f"  0.3-0.5: {((output_np >= 0.3) & (output_np < 0.5)).sum()} ({((output_np >= 0.3) & (output_np < 0.5)).sum() / len(output_np) * 100:.1f}%)")
print(f"  0.5-0.7: {((output_np >= 0.5) & (output_np < 0.7)).sum()} ({((output_np >= 0.5) & (output_np < 0.7)).sum() / len(output_np) * 100:.1f}%)")
print(f"  > 0.7: {(output_np >= 0.7).sum()} ({(output_np >= 0.7).sum() / len(output_np) * 100:.1f}%)")

# 计算损失
print("\n4. 损失计算...")
criterion_bce = torch.nn.BCELoss()

def dice_loss_fn(pred, target, smooth=1.0):
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice_score = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return 1 - dice_score

bce_loss = criterion_bce(outputs, targets)
dice_loss = dice_loss_fn(outputs, targets)

print(f"BCE Loss: {bce_loss.item():.6f}")
print(f"Dice Loss: {dice_loss.item():.6f}")
print(f"Dice Score: {(1 - dice_loss.item()):.6f}")

# 二值化后的Dice
pred_binary = (outputs >= 0.5).float()
dice_loss_binary = dice_loss_fn(pred_binary, targets)
print(f"二值化后Dice Score: {(1 - dice_loss_binary.item()):.6f}")

# 检查梯度
print("\n5. 梯度检查...")
model.train()
outputs = model(images)
loss = criterion_bce(outputs, targets) * 0.3 + dice_loss_fn(outputs, targets) * 1.0
loss.backward()

# 检查部分参数的梯度
grad_norms = []
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_norms.append(grad_norm)
        if 'transformer' in name and 'weight' in name:
            print(f"  {name}: grad_norm={grad_norm:.6f}")

print(f"\n梯度统计:")
print(f"  均值: {np.mean(grad_norms):.6f}")
print(f"  最大: {np.max(grad_norms):.6f}")
print(f"  最小: {np.min(grad_norms):.6f}")

print("\n="*60)
print("诊断完成")
print("="*60)
