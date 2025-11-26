"""
实时监控TransUNet训练进度
"""
import os
import json
import time
from datetime import datetime

log_dir = r"C:\Users\Masoa\OneDrive\work\CTAI\CTAI_model\logs"
history_file = os.path.join(log_dir, "transunet_622split_history.json")

print("=" * 70)
print("TransUNet训练监控")
print("=" * 70)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"监控文件: {history_file}")
print("=" * 70)
print()

last_epoch = 0
start_time = time.time()

while True:
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            if len(history['epoch']) > last_epoch:
                last_epoch = len(history['epoch'])
                current_epoch = history['epoch'][-1]
                
                train_loss = history['train_loss'][-1]
                train_dice = history['train_dice'][-1]
                val_dice = history['val_dice'][-1]
                lr = history['learning_rate'][-1]
                bce_loss = history['bce_loss'][-1]
                dice_loss_val = history['dice_loss'][-1]
                
                elapsed = time.time() - start_time
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                
                print(f"\n{'=' * 70}")
                print(f"📊 Epoch {current_epoch}/50 完成 | 已用时: {hours}h {minutes}m")
                print(f"{'=' * 70}")
                print(f"训练损失: {train_loss:.4f} (BCE: {bce_loss:.4f}, Dice: {dice_loss_val:.4f})")
                print(f"训练Dice: {train_dice:.4f} ({train_dice*100:.2f}%)")
                print(f"验证Dice: {val_dice:.4f} ({val_dice*100:.2f}%)")
                print(f"学习率:   {lr:.6f}")
                print(f"{'=' * 70}")
                
                # 显示最佳结果
                best_val_dice = max(history['val_dice'])
                best_epoch = history['val_dice'].index(best_val_dice) + 1
                print(f"✨ 最佳验证Dice: {best_val_dice:.4f} ({best_val_dice*100:.2f}%) @ Epoch {best_epoch}")
                print()
                
                # 预估剩余时间
                if current_epoch > 1:
                    avg_time_per_epoch = elapsed / current_epoch
                    remaining_epochs = 50 - current_epoch
                    remaining_time = avg_time_per_epoch * remaining_epochs
                    remaining_hours = int(remaining_time // 3600)
                    remaining_minutes = int((remaining_time % 3600) // 60)
                    print(f"⏱️  预计剩余时间: {remaining_hours}h {remaining_minutes}m")
                    print()
        except Exception as e:
            pass
    
    time.sleep(10)  # 每10秒检查一次
