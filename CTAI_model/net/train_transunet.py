"""
TransUNet训练脚本
在原有UNet+AttentionGate基础上升级为TransUNet架构
"""

import sys
sys.path.append("..")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练

from data_set import make
from net import transunet
from utils import dice_loss
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# ==================== 配置参数 ====================
class Config:
    # 数据配置
    train_dataset_path = 'C:/Users/Masoa/OneDrive/work/CTAI/src/train'
    rate = 0.50  # 二值化阈值
    
    # 模型配置
    model_type = 'transunet_lite'  # 改用Lite版本，更容易训练
    model_save_name = 'transunet_lite_622split'  # 保存的模型名称
    img_size = 512
    patch_size = 16
    embed_dim = 768  # 384 for lite version
    depth = 12       # 6 for lite version
    num_heads = 12   # 6 for lite version
    
    # 训练配置
    epochs = 50
    batch_size = 2  # RTX 3050 4GB显存,建议2
    learning_rate = 1e-4  # 降低学习率,更稳定
    weight_decay = 1e-4
    
    # 学习率调度器
    scheduler_type = 'cosine'  # 'cosine' 或 'plateau'
    
    # 损失函数权重（与UNet保持一致）
    dice_weight = 1.0
    bce_weight = 0.3
    
    # 混合精度训练
    use_amp = True  # 显存不足时必须开启
    
    # Early Stopping
    patience = 10
    min_delta = 0.001
    
    # 保存路径
    save_dir = '../checkpoints'
    log_dir = '../logs'


config = Config()

# ==================== 设备配置 ====================
torch.set_num_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

print(f"{'='*60}")
print("TransUNet训练启动")
print(f"{'='*60}")
print(f"设备: {device}")
print(f"模型类型: {config.model_type}")
print(f"批大小: {config.batch_size}")
print(f"学习率: {config.learning_rate}")
print(f"总Epochs: {config.epochs}")
print(f"混合精度: {'启用' if config.use_amp else '禁用'}")
print(f"{'='*60}\n")

# ==================== 数据加载 ====================
print("加载数据集...")
train_dataset, val_dataset, test_dataset = make.get_d1(config.train_dataset_path)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

print(f"每epoch步数: {len(train_loader)}\n")

# ==================== 模型初始化 ====================
print("初始化模型...")
if config.model_type == 'transunet_lite':
    model = transunet.TransUNetLite(
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_channels=1,
        out_channels=1
    ).to(device)
else:
    model = transunet.TransUNet(
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_channels=1,
        out_channels=1,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads
    ).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数量: {total_params / 1e6:.2f}M")
print(f"可训练参数: {trainable_params / 1e6:.2f}M\n")

# ==================== 损失函数 ====================
# 注意: 模型输出已经过Sigmoid,使用BCE Loss
# 但混合精度训练需要使用稳定的版本
criterion_bce = nn.BCELoss().to(device)

def dice_loss_fn(pred, target, smooth=1.0):
    """Dice Loss - 修复版本"""
    # 展平为1D向量进行计算
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice_score = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return 1 - dice_score

def combined_loss(pred, target):
    """组合损失: BCE + Dice (兼容混合精度训练)"""
    # 为了兼容混合精度,先转为float32再计算BCE
    with torch.cuda.amp.autocast(enabled=False):
        pred_fp32 = pred.float()
        target_fp32 = target.float()
        loss_bce = criterion_bce(pred_fp32, target_fp32)
    
    # Dice Loss可以直接用混合精度
    loss_dice = dice_loss_fn(pred, target)
    total_loss = config.bce_weight * loss_bce + config.dice_weight * loss_dice
    return total_loss, loss_bce.item(), loss_dice.item()

# ==================== 优化器和调度器 ====================
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=config.learning_rate,
    weight_decay=config.weight_decay
)

if config.scheduler_type == 'cosine':
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
else:
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

# 混合精度scaler
scaler = GradScaler() if config.use_amp else None

# ==================== Early Stopping ====================
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

# ==================== 训练记录 ====================
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

# ==================== 训练函数 ====================
def train_one_epoch(epoch):
    model.train()
    epoch_loss = 0
    epoch_dice = 0
    epoch_bce = 0
    epoch_dice_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}")
    
    for batch_idx, (x, mask) in enumerate(pbar):
        # DataLoader已经自动组好batch了
        # x是list: [images_batch, patient_ids, filenames]
        # mask是list: [filenames, masks_batch]
        # images_batch shape: (B, 1, 512, 512)
        # masks_batch shape: (B, 512, 512)
        
        images = x[0].to(device)  # (B, 1, 512, 512)
        targets = mask[1].unsqueeze(1).to(device)  # (B, 512, 512) -> (B, 1, 512, 512)
        
        optimizer.zero_grad()
        
        # 混合精度前向传播
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
        
        # 计算Dice
        with torch.no_grad():
            pred_np = outputs.cpu().detach().squeeze(1).numpy()
            target_np = targets.cpu().detach().squeeze(1).numpy()
            
            # 确保是3D数组(batch, H, W)
            if pred_np.ndim == 2:
                pred_np = pred_np[np.newaxis, ...]
            if target_np.ndim == 2:
                target_np = target_np[np.newaxis, ...]
            
            # 计算batch平均Dice
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
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Dice': f'{batch_dice:.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    # 计算平均值
    avg_loss = epoch_loss / len(train_loader)
    avg_dice = epoch_dice / len(train_loader)
    avg_bce = epoch_bce / len(train_loader)
    avg_dice_loss = epoch_dice_loss / len(train_loader)
    
    return avg_loss, avg_dice, avg_bce, avg_dice_loss

# ==================== 验证函数 ====================
def validate():
    """在验证集上评估模型"""
    model.eval()
    epoch_dice = 0
    sample_count = 0
    
    with torch.no_grad():
        for x, mask in val_loader:
            # DataLoader已经自动组好batch
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
                epoch_dice += dice_loss.dice(pred_binary, target_np[i])
                sample_count += 1
    
    avg_dice = epoch_dice / sample_count if sample_count > 0 else 0
    return avg_dice

# ==================== 测试函数 ====================
def test():
    """在测试集上评估模型（仅在训练完成后使用）"""
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

# ==================== 主训练循环 ====================
print("开始训练...\n")
start_time = datetime.now()

for epoch in range(1, config.epochs + 1):
    # 训练
    train_loss, train_dice, train_bce, train_dice_loss = train_one_epoch(epoch)
    
    # 验证
    val_dice = validate()
    
    # 学习率调度
    if config.scheduler_type == 'cosine':
        scheduler.step()
    else:
        scheduler.step(val_dice)
    
    current_lr = optimizer.param_groups[0]['lr']
    
    # 记录历史
    history['epoch'].append(epoch)
    history['train_loss'].append(train_loss)
    history['train_dice'].append(train_dice)
    history['val_dice'].append(val_dice)
    history['learning_rate'].append(current_lr)
    history['bce_loss'].append(train_bce)
    history['dice_loss'].append(train_dice_loss)
    
    # 打印统计
    print(f"\n{'='*60}")
    print(f"Epoch {epoch}/{config.epochs} 统计")
    print(f"{'='*60}")
    print(f"训练损失: {train_loss:.4f} (BCE: {train_bce:.4f}, Dice: {train_dice_loss:.4f})")
    print(f"训练Dice: {train_dice:.4f}")
    print(f"验证Dice: {val_dice:.4f}")
    print(f"学习率:   {current_lr:.6f}")
    print(f"{'='*60}\n")
    
    # 保存最佳模型 (基于验证集)
    if val_dice > best_dice:
        best_dice = val_dice
        best_model_path = f'{config.save_dir}/{config.model_save_name}_best.pth'
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ 保存最佳模型 (验证Dice: {best_dice:.4f})\n")
    
    # Early Stopping检查 (基于验证集)
    if early_stopping(val_dice):
        print(f"⚠️ Early Stopping触发 (Patience: {config.patience})")
        break

# 训练结束
end_time = datetime.now()
training_time = (end_time - start_time).total_seconds()

# 在训练完成后,在测试集上最终评估
print(f"\n{'='*60}")
print("在测试集上进行最终评估...")
print(f"{'='*60}")
test_dice = test()

print(f"\n{'='*60}")
print(f"✨ 训练完成!")
print(f"{'='*60}")
print(f"总耗时: {training_time/3600:.2f} 小时")
print(f"最佳验证Dice: {best_dice:.4f}")
print(f"最终测试Dice: {test_dice:.4f}")
print(f"{'='*60}\n")

# ==================== 保存结果 ====================
# 保存最终模型
final_model_path = f'{config.save_dir}/{config.model_save_name}_final.pth'
torch.save(model.state_dict(), final_model_path)

# 保存训练历史
history_path = f'{config.log_dir}/{config.model_save_name}_history.json'
with open(history_path, 'w') as f:
    json.dump(history, f, indent=4)

# ==================== 绘制训练曲线 ====================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss曲线
axes[0, 0].plot(history['epoch'], history['train_loss'], label='Total Loss', linewidth=2)
axes[0, 0].plot(history['epoch'], history['bce_loss'], label='BCE Loss', linewidth=2, alpha=0.7)
axes[0, 0].plot(history['epoch'], history['dice_loss'], label='Dice Loss', linewidth=2, alpha=0.7)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Dice曲线
axes[0, 1].plot(history['epoch'], history['train_dice'], label='Train Dice', linewidth=2)
axes[0, 1].plot(history['epoch'], history['val_dice'], label='Val Dice', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Dice Score')
axes[0, 1].set_title('Dice Score Progression')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=0.80, color='r', linestyle='--', alpha=0.5, label='Target (0.80)')

# 学习率曲线
axes[1, 0].plot(history['epoch'], history['learning_rate'], linewidth=2, color='green')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Learning Rate')
axes[1, 0].set_title('Learning Rate Schedule')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True, alpha=0.3)

# 性能对比
axes[1, 1].bar(['Train Dice', 'Val Dice', 'Test Dice'], 
               [history['train_dice'][-1], history['val_dice'][-1], test_dice],
               color=['#4CAF50', '#2196F3', '#FF9800'])
axes[1, 1].set_ylabel('Dice Score')
axes[1, 1].set_title('Final Performance')
axes[1, 1].set_ylim([0, 1])
axes[1, 1].axhline(y=0.80, color='r', linestyle='--', alpha=0.5)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_path = f'{config.log_dir}/{config.model_save_name}_curves.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"📈 训练曲线已保存到 {plot_path}\n")

# ==================== 生成训练报告 ====================
report = f"""
# TransUNet训练报告 - {config.model_save_name}

## 训练配置
- 模型类型: {config.model_type}
- 参数量: {total_params / 1e6:.2f}M
- 批大小: {config.batch_size}
- 初始学习率: {config.learning_rate}
- 总Epochs: {epoch}
- 训练时长: {training_time/3600:.2f} 小时
- 数据划分: 6:2:2 (随机患者级别)

## 性能指标
- 最佳验证Dice: {best_dice:.4f} ({best_dice*100:.2f}%)
- 最终测试Dice: {test_dice:.4f} ({test_dice*100:.2f}%)
- 最终训练Dice: {history['train_dice'][-1]:.4f}
- 过拟合程度: {abs(history['train_dice'][-1] - test_dice):.4f}

## 损失函数
- BCE权重: {config.bce_weight}
- Dice权重: {config.dice_weight}
- 最终BCE Loss: {history['bce_loss'][-1]:.4f}
- 最终Dice Loss: {history['dice_loss'][-1]:.4f}

## 与Baseline对比 (UNet+AttentionGate)
- Baseline Dice: 需要在新划分上重新训练后对比
- TransUNet Dice: {best_dice:.4f}

## 保存的模型
- 最佳模型: {config.save_dir}/{config.model_save_name}_best.pth
- 最终模型: {config.save_dir}/{config.model_save_name}_final.pth

## 训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

report_path = f'{config.log_dir}/{config.model_save_name}_report.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print(f"📄 训练报告已保存到 {report_path}\n")
print(f"📦 最佳模型: {config.save_dir}/{config.model_save_name}_best.pth")
print(f"📦 最终模型: {final_model_path}\n")
print("🎉 所有任务完成!")
