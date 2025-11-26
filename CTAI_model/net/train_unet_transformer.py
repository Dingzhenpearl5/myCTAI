"""
UNet + Transformer 训练脚本
更适合小数据集的混合架构
"""

import sys
sys.path.append("..")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from data_set import make
from net import unet_transformer
from utils import dice_loss
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import os

# 配置
class Config:
    # 数据配置
    train_dataset_path = '../../src/train'
    rate = 0.50
    
    # 模型配置
    num_transformer_layers = 3  # 瓶颈层Transformer层数
    model_save_name = 'unet_transformer_622split'
    
    # 训练配置
    epochs = 50
    batch_size = 2
    learning_rate = 1e-4
    weight_decay = 1e-4
    
    # 损失函数权重
    dice_weight = 1.0
    bce_weight = 0.3
    
    # 混合精度
    use_amp = True
    
    # Early Stopping
    patience = 10
    min_delta = 0.001
    
    # 保存路径
    save_dir = '../checkpoints'
    log_dir = '../logs'

config = Config()
os.makedirs(config.save_dir, exist_ok=True)
os.makedirs(config.log_dir, exist_ok=True)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

print(f"{'='*60}")
print("UNet + Transformer 训练启动")
print(f"{'='*60}")
print(f"设备: {device}")
print(f"Transformer层数: {config.num_transformer_layers}")
print(f"批大小: {config.batch_size}")
print(f"学习率: {config.learning_rate}")
print(f"总Epochs: {config.epochs}")
print(f"{'='*60}\n")

# 加载数据集
print("加载数据集...")
train_dataset, val_dataset, test_dataset = make.get_d1(config.train_dataset_path)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")
print(f"每epoch步数: {len(train_loader)}\n")

# 初始化模型
print("初始化模型...")
model = unet_transformer.UNetWithTransformer(
    in_channels=1,
    out_channels=1,
    num_transformer_layers=config.num_transformer_layers
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params / 1e6:.2f}M\n")

# 损失函数
criterion_bce = nn.BCELoss().to(device)

def dice_loss_fn(pred, target, smooth=1.0):
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice_score = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return 1 - dice_score

def combined_loss(pred, target):
    with torch.cuda.amp.autocast(enabled=False):
        pred_fp32 = pred.float()
        target_fp32 = target.float()
        loss_bce = criterion_bce(pred_fp32, target_fp32)
    
    loss_dice = dice_loss_fn(pred, target)
    total_loss = config.bce_weight * loss_bce + config.dice_weight * loss_dice
    return total_loss, loss_bce.item(), loss_dice.item()

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
scaler = GradScaler() if config.use_amp else None

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_dice = 0
        self.early_stop = False
        
    def __call__(self, dice):
        if dice < self.best_dice + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_dice = dice
            self.counter = 0
        return self.early_stop

early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)

# 训练记录
history = {
    'epoch': [],
    'train_loss': [],
    'train_dice': [],
    'val_dice': [],
    'learning_rate': [],
    'bce_loss': [],
    'dice_loss': []
}

best_dice = 0.0

# 训练函数
def train_one_epoch(epoch):
    model.train()
    epoch_loss = 0
    epoch_dice = 0
    epoch_bce = 0
    epoch_dice_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}")
    
    for batch_idx, (x, mask) in enumerate(pbar):
        images = x[0].to(device)
        targets = mask[1].unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        
        if config.use_amp:
            with autocast():
                outputs = model(images)
                loss, loss_bce, loss_dice = combined_loss(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss, loss_bce, loss_dice = combined_loss(outputs, targets)
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            pred_np = outputs.cpu().detach().squeeze(1).numpy()
            target_np = targets.cpu().detach().squeeze(1).numpy()
            
            if pred_np.ndim == 2:
                pred_np = pred_np[np.newaxis, ...]
            if target_np.ndim == 2:
                target_np = target_np[np.newaxis, ...]
            
            batch_dice = 0
            for i in range(pred_np.shape[0]):
                pred_binary = (pred_np[i] >= config.rate).astype(np.float32)
                target_binary = target_np[i]
                batch_dice += dice_loss.dice(pred_binary, target_binary)
            batch_dice /= pred_np.shape[0]
        
        epoch_loss += loss.item()
        epoch_dice += batch_dice
        epoch_bce += loss_bce
        epoch_dice_loss += loss_dice
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Dice': f'{batch_dice:.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    avg_loss = epoch_loss / len(train_loader)
    avg_dice = epoch_dice / len(train_loader)
    avg_bce = epoch_bce / len(train_loader)
    avg_dice_loss = epoch_dice_loss / len(train_loader)
    
    return avg_loss, avg_dice, avg_bce, avg_dice_loss

# 验证函数
def validate():
    model.eval()
    epoch_dice = 0
    sample_count = 0
    
    with torch.no_grad():
        for x, mask in val_loader:
            images = x[0].to(device)
            mask_batch = mask[1]
            
            outputs = model(images)
            
            pred_np = outputs.cpu().squeeze(1).numpy()
            target_np = mask_batch.cpu().numpy()
            
            batch_size = pred_np.shape[0]
            for i in range(batch_size):
                pred_binary = (pred_np[i] >= config.rate).astype(np.float32)
                epoch_dice += dice_loss.dice(pred_binary, target_np[i])
                sample_count += 1
    
    avg_dice = epoch_dice / sample_count if sample_count > 0 else 0
    return avg_dice

# 测试函数
def test():
    model.eval()
    epoch_dice = 0
    sample_count = 0
    
    with torch.no_grad():
        for x, mask in test_loader:
            images = x[0].to(device)
            mask_batch = mask[1]
            
            outputs = model(images)
            
            pred_np = outputs.cpu().squeeze(1).numpy()
            target_np = mask_batch.cpu().numpy()
            
            batch_size = pred_np.shape[0]
            for i in range(batch_size):
                pred_binary = (pred_np[i] >= config.rate).astype(np.float32)
                epoch_dice += dice_loss.dice(pred_binary, target_np[i])
                sample_count += 1
    
    avg_dice = epoch_dice / sample_count if sample_count > 0 else 0
    return avg_dice

# 主训练循环
print("开始训练...\n")
start_time = datetime.now()

for epoch in range(1, config.epochs + 1):
    train_loss, train_dice, train_bce, train_dice_loss = train_one_epoch(epoch)
    val_dice = validate()
    scheduler.step()
    
    current_lr = optimizer.param_groups[0]['lr']
    
    history['epoch'].append(epoch)
    history['train_loss'].append(train_loss)
    history['train_dice'].append(train_dice)
    history['val_dice'].append(val_dice)
    history['learning_rate'].append(current_lr)
    history['bce_loss'].append(train_bce)
    history['dice_loss'].append(train_dice_loss)
    
    print(f"\n{'='*60}")
    print(f"Epoch {epoch}/{config.epochs} 统计")
    print(f"{'='*60}")
    print(f"训练损失: {train_loss:.4f} (BCE: {train_bce:.4f}, Dice: {train_dice_loss:.4f})")
    print(f"训练Dice: {train_dice:.4f}")
    print(f"验证Dice: {val_dice:.4f}")
    print(f"学习率:   {current_lr:.6f}")
    print(f"{'='*60}\n")
    
    if val_dice > best_dice:
        best_dice = val_dice
        best_model_path = f'{config.save_dir}/{config.model_save_name}_best.pth'
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ 保存最佳模型 (验证Dice: {best_dice:.4f})\n")
    
    if early_stopping(val_dice):
        print(f"⚠️ Early Stopping触发")
        break

end_time = datetime.now()
training_time = (end_time - start_time).total_seconds()

print(f"\n{'='*60}")
print("最终评估...")
print(f"{'='*60}")
test_dice = test()

print(f"\n{'='*60}")
print(f"✨ 训练完成!")
print(f"{'='*60}")
print(f"总耗时: {training_time/3600:.2f} 小时")
print(f"最佳验证Dice: {best_dice:.4f} ({best_dice*100:.2f}%)")
print(f"最终测试Dice: {test_dice:.4f} ({test_dice*100:.2f}%)")
print(f"{'='*60}\n")

# 保存模型
final_model_path = f'{config.save_dir}/{config.model_save_name}_final.pth'
torch.save(model.state_dict(), final_model_path)

# 保存历史
history_path = f'{config.log_dir}/{config.model_save_name}_history.json'
with open(history_path, 'w') as f:
    json.dump(history, f, indent=4)

print(f"📦 最佳模型: {best_model_path}")
print(f"📦 最终模型: {final_model_path}\n")
print("🎉 训练完成!")
