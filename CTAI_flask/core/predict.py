import os
import sys
import cv2
import torch
import core.net.unet as net
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

import os

rate = 0.5


def predict(dataset, model):
    """
    使用模型进行预测
    """
    print(f"[Predict] 开始预测...")
    
    global res, img_y, mask_arrary
    
    try:
        with torch.no_grad():
            print(f"[Predict] 准备数据...")
            x = dataset[0][0].to(device)
            file_name = dataset[1]
            print(f"[Predict] 文件名: {file_name}, 输入shape: {x.shape}")

            # 打印输入数值范围，帮助排查归一化/数值问题
            try:
                x_min = float(x.min().cpu().numpy())
                x_max = float(x.max().cpu().numpy())
            except Exception:
                x_min, x_max = None, None
            print(f"[Predict] 输入范围: min={x_min}, max={x_max}")

            print(f"[Predict] 开始模型推理...")
            y = model(x)
            print(f"[Predict] 推理完成，输出shape: {y.shape}")

            # 打印输出统计信息
            try:
                y_min = float(y.min().cpu().numpy())
                y_max = float(y.max().cpu().numpy())
            except Exception:
                y_min, y_max = None, None
            print(f"[Predict] 输出范围: min={y_min}, max={y_max}")

            print(f"[Predict] 后处理中...")
            img_y = torch.squeeze(y).cpu().numpy()
            # 若输出是概率(0-1)，按阈值二值化；否则直接按非零判断
            if y_max is not None and y_max <= 1.0:
                bin_mask = (img_y >= rate).astype('uint8')
            else:
                bin_mask = (img_y != 0).astype('uint8')

            # 将二值掩码扩展到0-255并确保uint8
            img_y_out = (bin_mask * 255).astype('uint8')

            # 打印唯一值以确认是否有正例
            unique_vals = np.unique(img_y_out)
            print(f"[Predict] mask 唯一值: {unique_vals}")

            mask_path = f'./tmp/mask/{file_name}_mask.png'
            cv2.imwrite(mask_path, img_y_out, (cv2.IMWRITE_PNG_COMPRESSION, 0))
            print(f"[Predict] ✅ 预测完成，mask保存至: {mask_path}")
            
    except Exception as e:
        print(f"[Predict] ❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    # 写保存模型
    # train()
    predict()
