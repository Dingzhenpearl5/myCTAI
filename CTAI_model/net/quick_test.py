"""快速数据加载测试"""
import sys
sys.path.append("..")
from data_set import make

print("开始加载数据...")
train_dataset_path = 'C:/Users/Masoa/OneDrive/work/CTAI/src/train'

try:
    train_dataset, test_dataset = make.get_d1(train_dataset_path)
    print(f"✅ 数据加载成功!")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 检查第一个样本
    x, y = train_dataset[0]
    print(f"\n第一个样本:")
    print(f"  图像: {x[0].shape}, range=[{x[0].min():.3f}, {x[0].max():.3f}]")
    print(f"  Mask: {y[1].shape}, range=[{y[1].min():.3f}, {y[1].max():.3f}]")
    print(f"  Mask unique值: {y[1].unique().tolist()}")
    print(f"  肿瘤像素比例: {(y[1] > 0).sum().item() / y[1].numel() * 100:.2f}%")
    
except Exception as e:
    print(f"❌ 加载失败: {e}")
    import traceback
    traceback.print_exc()
