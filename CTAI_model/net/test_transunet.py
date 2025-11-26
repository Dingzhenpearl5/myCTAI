"""
TransUNet快速测试脚本
用于验证模型是否可以正常运行和计算显存占用
"""

import torch
import sys
sys.path.append('..')
from net.transunet import TransUNet, TransUNetLite

def test_model(model_type='transunet'):
    print(f"\n{'='*60}")
    print(f"测试模型: {model_type}")
    print(f"{'='*60}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    if device.type == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 创建模型
    if model_type == 'transunet_lite':
        model = TransUNetLite(img_size=512, patch_size=16, in_channels=1, out_channels=1)
    else:
        model = TransUNet(
            img_size=512,
            patch_size=16,
            in_channels=1,
            out_channels=1,
            embed_dim=768,
            depth=12,
            num_heads=12
        )
    
    model = model.to(device)
    model.eval()
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 模型统计:")
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数: {trainable_params / 1e6:.2f}M")
    
    # 测试前向传播
    print(f"\n🔬 测试前向传播...")
    
    try:
        # 测试batch_size=1
        x1 = torch.randn(1, 1, 512, 512).to(device)
        
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            y1 = model(x1)
        
        print(f"✅ Batch Size 1 成功")
        print(f"   输入: {x1.shape}")
        print(f"   输出: {y1.shape}")
        
        if device.type == 'cuda':
            mem_used = torch.cuda.max_memory_allocated() / 1e9
            print(f"   显存占用: {mem_used:.2f} GB")
        
        # 测试batch_size=2
        print(f"\n🔬 测试 Batch Size 2...")
        x2 = torch.randn(2, 1, 512, 512).to(device)
        
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            y2 = model(x2)
        
        print(f"✅ Batch Size 2 成功")
        print(f"   输入: {x2.shape}")
        print(f"   输出: {y2.shape}")
        
        if device.type == 'cuda':
            mem_used = torch.cuda.max_memory_allocated() / 1e9
            print(f"   显存占用: {mem_used:.2f} GB")
            
            # 估算训练时显存(约为推理的2-3倍)
            estimated_train_mem = mem_used * 2.5
            print(f"   预估训练显存: {estimated_train_mem:.2f} GB")
            
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            if estimated_train_mem > total_mem * 0.9:
                print(f"   ⚠️  警告: 显存可能不足,建议:")
                print(f"       1. 使用 TransUNetLite 版本")
                print(f"       2. 启用混合精度训练 (use_amp=True)")
                print(f"       3. 减小 batch_size 到 1")
            else:
                print(f"   ✅ 显存充足,可以正常训练")
        
        print(f"\n✅ 所有测试通过!")
        
    except RuntimeError as e:
        print(f"\n❌ 错误: {str(e)}")
        if 'out of memory' in str(e):
            print(f"\n💡 建议:")
            print(f"   1. 使用 TransUNetLite: python test_transunet.py --lite")
            print(f"   2. 在训练脚本中设置 use_amp=True")
            print(f"   3. 减小 batch_size")
        return False
    
    print(f"\n{'='*60}\n")
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lite', action='store_true', help='测试轻量级版本')
    args = parser.parse_args()
    
    model_type = 'transunet_lite' if args.lite else 'transunet'
    
    success = test_model(model_type)
    
    if success:
        print("🎉 TransUNet模型测试成功!")
        print("\n📝 下一步:")
        print("   1. 安装依赖: pip install -r requirements_transunet.txt")
        print("   2. 开始训练: python train_transunet.py")
    else:
        print("⚠️  请根据建议调整配置后重试")
