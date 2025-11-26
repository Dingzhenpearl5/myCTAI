"""
使用新的患者级别数据划分测试原始UNet+AttentionGate模型
"""

import sys
sys.path.append("..")

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_set import make
from net import unet
from utils import dice_loss

# 配置
class Config:
    model_path = '../model_Unet.pth'  # 之前训练好的UNet模型
    train_dataset_path = '../../src/train'
    batch_size = 2
    rate = 0.5  # 二值化阈值

config = Config()

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}\n")

# 加载数据集（使用新的患者级别划分）
print("="*60)
print("加载数据集（患者级别划分）...")
print("="*60)
train_dataset, val_dataset, test_dataset = make.get_d1(config.train_dataset_path)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

print(f"每epoch步数: {len(train_loader)}\n")

# 加载模型
print("="*60)
print("加载UNet+AttentionGate模型...")
print("="*60)
from unet import Unet
model = Unet(1, 1).to(device)

try:
    checkpoint = torch.load(config.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"成功加载模型: {config.model_path}")
except Exception as e:
    print(f"加载模型失败: {e}")
    print("将使用未训练的模型进行测试")

total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params / 1e6:.2f}M\n")

# 测试函数
def evaluate(data_loader, dataset_name="测试"):
    model.eval()
    epoch_dice = 0
    sample_count = 0
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f"评估{dataset_name}集")
        for x, mask in pbar:
            # 提取数据
            images = x[0].to(device)  # (B, 1, 512, 512)
            mask_batch = mask[1]  # (B, 512, 512)
            
            # 前向传播
            outputs = model(images)
            
            # 计算每个样本的Dice
            pred_np = outputs.cpu().squeeze(1).numpy()  # (B, 512, 512)
            target_np = mask_batch.cpu().numpy()  # (B, 512, 512)
            
            batch_size = pred_np.shape[0]
            for i in range(batch_size):
                pred_binary = (pred_np[i] >= config.rate).astype(np.float32)
                target_binary = target_np[i]
                dice_score = dice_loss.dice(pred_binary, target_binary)
                epoch_dice += dice_score
                sample_count += 1
            
            pbar.set_postfix({'Dice': f'{epoch_dice/sample_count:.4f}'})
    
    avg_dice = epoch_dice / sample_count if sample_count > 0 else 0
    return avg_dice, sample_count

# 评估训练集
print("="*60)
print("评估训练集表现...")
print("="*60)
train_dice, train_count = evaluate(train_loader, "训练")
print(f"\n训练集统计:")
print(f"  样本数: {train_count}")
print(f"  平均Dice: {train_dice:.4f} ({train_dice*100:.2f}%)")

# 评估验证集
print("\n" + "="*60)
print("评估验证集表现...")
print("="*60)
val_dice, val_count = evaluate(val_loader, "验证")
print(f"\n验证集统计:")
print(f"  样本数: {val_count}")
print(f"  平均Dice: {val_dice:.4f} ({val_dice*100:.2f}%)")

# 评估测试集
print("\n" + "="*60)
print("评估测试集表现...")
print("="*60)
test_dice, test_count = evaluate(test_loader, "测试")
print(f"\n测试集统计:")
print(f"  样本数: {test_count}")
print(f"  平均Dice: {test_dice:.4f} ({test_dice*100:.2f}%)")

# 总结
print("\n" + "="*60)
print("UNet+AttentionGate 在新数据划分上的表现")
print("="*60)
print(f"训练集Dice: {train_dice:.4f} ({train_dice*100:.2f}%)")
print(f"验证集Dice: {val_dice:.4f} ({val_dice*100:.2f}%)")
print(f"测试集Dice: {test_dice:.4f} ({test_dice*100:.2f}%)")
print(f"Train-Val Gap: {abs(train_dice - val_dice):.4f} ({abs(train_dice - val_dice)*100:.2f}%)")
print(f"Train-Test Gap: {abs(train_dice - test_dice):.4f} ({abs(train_dice - test_dice)*100:.2f}%)")
print("="*60)

# 与之前的结果对比
print("\n对比分析:")
print("之前(随机划分):")
print("  训练集Dice: ~92.57%")
print("  测试集Dice: ~85.42%")
print("  Train-Test Gap: ~7.15%")
print("\n现在(患者级别划分 6:2:2):")
print(f"  训练集Dice: {train_dice*100:.2f}%")
print(f"  验证集Dice: {val_dice*100:.2f}%")
print(f"  测试集Dice: {test_dice*100:.2f}%")
print(f"  Train-Val Gap: {abs(train_dice - val_dice)*100:.2f}%")
print(f"  Train-Test Gap: {abs(train_dice - test_dice)*100:.2f}%")
print("\n结论:")
if test_dice < 0.85:
    print("  - 测试集性能下降是正常的,因为新划分避免了数据泄露")
    print("  - 这个结果更能反映模型对新患者的真实泛化能力")
else:
    print("  - 模型在新划分上仍然保持良好性能")
    print("  - 说明之前的训练确实学到了通用特征")
