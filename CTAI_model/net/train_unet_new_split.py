"""
使用新的6:2:2患者级别划分重新训练UNet+AttentionGate
模型将保存为: model_Unet_622split.pth
"""

import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
import os
from datetime import datetime

from data_set import make
from unet import Unet
from utils import dice_loss
from utils.augmentation import get_training_augmentation

# 配置
class Config:
    # 数据配置
    train_dataset_path = '../../src/train'
    batch_size = 2
    num_workers = 0
    
    # 数据增强配置
    use_augmentation = True
    augmentation_mode = 'medium'  # 'light', 'medium', 'heavy'
    
    # 训练配置
    learning_rate = 1e-4
    num_epochs = 50
    
    # 损失函数权重
    dice_weight = 1.0
    bce_weight = 0.3
    
    # 保存配置
    save_dir = '../checkpoints'
    model_name = 'unet_622split'
    
    # 早停配置
    patience = 10
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# 创建保存目录
os.makedirs(config.save_dir, exist_ok=True)

print("="*60)
print("UNet+AttentionGate 训练 (6:2:2新划分)")
print("="*60)
print(f"设备: {config.device}")
print(f"批大小: {config.batch_size}")
print(f"学习率: {config.learning_rate}")
print(f"总Epochs: {config.num_epochs}")
print(f"模型保存名: {config.model_name}")
print(f"数据增强: {'启用' if config.use_augmentation else '禁用'}")
if config.use_augmentation:
    print(f"增强模式: {config.augmentation_mode}")
print("="*60)

# 加载数据集
print("\n加载数据集...")

# 创建数据增强
train_transform = None
if config.use_augmentation:
    train_transform = get_training_augmentation(mode=config.augmentation_mode)
    print(f"✓ 训练集数据增强已启用 ({config.augmentation_mode}模式)")

train_dataset, val_dataset, test_dataset = make.get_d1(
    config.train_dataset_path,
    train_transform=train_transform,
    val_transform=None  # 验证集不使用增强
)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers)

print(f"每epoch步数: {len(train_loader)}\n")

# 初始化模型
print("初始化模型...")
model = Unet(1, 1).to(config.device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数量: {total_params / 1e6:.2f}M")
print(f"可训练参数: {trainable_params / 1e6:.2f}M\n")

# 损失函数和优化器
def dice_loss_fn(pred, target, smooth=1.0):
    """Dice Loss for segmentation"""
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice_score = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return 1 - dice_score

criterion_bce = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

# 训练历史
history = {
    'train_loss': [],
    'train_dice': [],
    'val_dice': [],
    'bce_loss': [],
    'dice_loss': [],
    'lr': []
}

best_val_dice = 0
patience_counter = 0

def train_one_epoch(epoch):
    model.train()
    epoch_loss = 0
    epoch_dice = 0
    epoch_bce = 0
    epoch_dice_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
    for x, mask in pbar:
        # 提取数据
        images = x[0].to(config.device)  # (B, 1, 512, 512)
        mask_batch = mask[1].to(config.device)  # (B, 512, 512)
        
        # 前向传播
        outputs = model(images)
        outputs = outputs.squeeze(1)  # (B, 512, 512)
        
        # 计算损失
        loss_dice = dice_loss_fn(outputs, mask_batch)
        loss_bce = criterion_bce(outputs, mask_batch)
        loss = config.dice_weight * loss_dice + config.bce_weight * loss_bce
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算Dice分数
        with torch.no_grad():
            pred_np = outputs.cpu().numpy()
            target_np = mask_batch.cpu().numpy()
            batch_dice = 0
            for i in range(pred_np.shape[0]):
                pred_binary = (pred_np[i] >= 0.5).astype(np.float32)
                batch_dice += dice_loss.dice(pred_binary, target_np[i])
            batch_dice /= pred_np.shape[0]
        
        # 累计统计
        epoch_loss += loss.item()
        epoch_dice += batch_dice
        epoch_bce += loss_bce.item()
        epoch_dice_loss += loss_dice.item()
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Dice': f'{batch_dice:.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    # 返回平均值
    n_batches = len(train_loader)
    return epoch_loss / n_batches, epoch_dice / n_batches, epoch_bce / n_batches, epoch_dice_loss / n_batches

def validate():
    model.eval()
    epoch_dice = 0
    sample_count = 0
    
    with torch.no_grad():
        for x, mask in val_loader:
            # 提取数据
            images = x[0].to(config.device)
            mask_batch = mask[1]
            
            # 前向传播
            outputs = model(images)
            
            # 计算Dice
            pred_np = outputs.cpu().squeeze(1).numpy()
            target_np = mask_batch.cpu().numpy()
            
            for i in range(pred_np.shape[0]):
                pred_binary = (pred_np[i] >= 0.5).astype(np.float32)
                dice_score = dice_loss.dice(pred_binary, target_np[i])
                epoch_dice += dice_score
                sample_count += 1
    
    avg_dice = epoch_dice / sample_count if sample_count > 0 else 0
    return avg_dice

def test():
    model.eval()
    epoch_dice = 0
    sample_count = 0
    
    with torch.no_grad():
        for x, mask in test_loader:
            # 提取数据
            images = x[0].to(config.device)
            mask_batch = mask[1]
            
            # 前向传播
            outputs = model(images)
            
            # 计算Dice
            pred_np = outputs.cpu().squeeze(1).numpy()
            target_np = mask_batch.cpu().numpy()
            
            for i in range(pred_np.shape[0]):
                pred_binary = (pred_np[i] >= 0.5).astype(np.float32)
                dice_score = dice_loss.dice(pred_binary, target_np[i])
                epoch_dice += dice_score
                sample_count += 1
    
    avg_dice = epoch_dice / sample_count if sample_count > 0 else 0
    return avg_dice

# 开始训练
print("开始训练...\n")
start_time = time.time()

for epoch in range(config.num_epochs):
    # 训练一个epoch
    train_loss, train_dice, train_bce, train_dice_loss = train_one_epoch(epoch)
    
    # 验证
    val_dice = validate()
    
    # 更新学习率
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # 记录历史
    history['train_loss'].append(train_loss)
    history['train_dice'].append(train_dice)
    history['val_dice'].append(val_dice)
    history['bce_loss'].append(train_bce)
    history['dice_loss'].append(train_dice_loss)
    history['lr'].append(current_lr)
    
    # 打印统计
    print("="*60)
    print(f"Epoch {epoch+1}/{config.num_epochs} 统计")
    print("="*60)
    print(f"训练损失: {train_loss:.4f} (BCE: {train_bce:.4f}, Dice: {train_dice_loss:.4f})")
    print(f"训练Dice: {train_dice:.4f}")
    print(f"验证Dice: {val_dice:.4f}")
    print(f"学习率:   {current_lr:.6f}")
    print("="*60 + "\n")
    
    # 保存最佳模型
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        patience_counter = 0
        
        best_model_path = os.path.join(config.save_dir, f'{config.model_name}_best.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ 保存最佳模型 (验证Dice: {val_dice:.4f})\n")
    else:
        patience_counter += 1
        print(f"验证Dice未提升 ({patience_counter}/{config.patience})\n")
    
    # 早停
    if patience_counter >= config.patience:
        print(f"早停触发! 验证Dice已{config.patience}个epoch未提升")
        break

# 训练结束
training_time = time.time() - start_time

# 加载最佳模型并在测试集上评估
print("\n" + "="*60)
print("加载最佳模型进行最终测试...")
print("="*60)
best_model_path = os.path.join(config.save_dir, f'{config.model_name}_best.pth')
model.load_state_dict(torch.load(best_model_path))
test_dice = test()

print(f"最佳验证Dice: {best_val_dice:.4f}")
print(f"最终测试Dice: {test_dice:.4f}")
print("="*60)

# 保存最终模型
final_model_path = os.path.join(config.save_dir, f'{config.model_name}_final.pth')
torch.save(model.state_dict(), final_model_path)

# 保存训练报告
report_path = os.path.join(config.save_dir, f'{config.model_name}_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("UNet+AttentionGate 训练报告 (6:2:2新划分)\n")
    f.write("="*60 + "\n\n")
    f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"训练时长: {training_time/3600:.2f} 小时\n")
    f.write(f"总Epochs: {epoch+1}\n\n")
    
    f.write("## 数据集划分\n")
    f.write(f"- 训练集大小: {len(train_dataset)}\n")
    f.write(f"- 验证集大小: {len(val_dataset)}\n")
    f.write(f"- 测试集大小: {len(test_dataset)}\n")
    f.write(f"- 划分比例: 6:2:2\n")
    f.write(f"- 数据增强: {'启用' if config.use_augmentation else '禁用'}\n")
    if config.use_augmentation:
        f.write(f"- 增强模式: {config.augmentation_mode}\n")
    f.write("\n")
    
    f.write("## 模型配置\n")
    f.write(f"- 模型: UNet+AttentionGate\n")
    f.write(f"- 总参数量: {total_params/1e6:.2f}M\n")
    f.write(f"- 批大小: {config.batch_size}\n")
    f.write(f"- 学习率: {config.learning_rate}\n\n")
    
    f.write("## 性能指标\n")
    f.write(f"- 最佳验证Dice: {best_val_dice:.4f} ({best_val_dice*100:.2f}%)\n")
    f.write(f"- 最终测试Dice: {test_dice:.4f} ({test_dice*100:.2f}%)\n")
    f.write(f"- 最终训练Dice: {history['train_dice'][-1]:.4f} ({history['train_dice'][-1]*100:.2f}%)\n\n")
    
    f.write("## 保存的模型文件\n")
    f.write(f"- 最佳模型: {best_model_path}\n")
    f.write(f"- 最终模型: {final_model_path}\n")

print(f"\n✅ 训练完成! 报告已保存到: {report_path}")
print(f"✅ 最佳模型: {best_model_path}")
print(f"✅ 最终模型: {final_model_path}")
