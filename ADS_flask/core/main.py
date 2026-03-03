from core import process, predict, get_feature
import time
import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def c_main(path, model, progress_callback=None):
    """
    主处理函数
    :param path: DCM文件路径
    :param model: 模型对象
    :param progress_callback: 进度回调函数 callback(percentage, message)
    """
    print(f"\n{'='*60}")
    print(f"[Main]开始处理: {path}")
    start_time = time.time()
    
    def emit(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)
        print(f"[Progress] {pct}% - {msg}")
    
    try:
        # 1. 预处理
        emit(20, '预处理图像...')
        print(f"[Main]Step 1/4: 预处理图像...")
        t1 = time.time()
        image_data = process.pre_process(path)
        print(f"[Main]预处理完成 ({time.time()-t1:.2f}秒)")
        
        # 2. 模型预测
        emit(40, '模型推理中...')
        print(f"[Main]Step 2/4: 模型预测...")
        
        heatmap_generated = False
        pid = image_data[1]
        
        if model is not None:
            t2 = time.time()
            predict_result = predict.predict(image_data, model)
            if isinstance(predict_result, dict) and 'heatmap_path' in predict_result:
                heatmap_generated = True
            print(f"[Main] 预测完成 ({time.time()-t2:.2f}秒)")
        else:
            # 强制使用不需要模拟数据
             raise RuntimeError("系统错误: AI诊断模型未加载，无法进行预测。")
            
        
        # 3. 后处理
        emit(70, '后处理生成轮廓...')
        print(f"[Main]Step 3/4: 后处理...")
        t3 = time.time()
        process.last_process(image_data[1])
        print(f"[Main]后处理完成 ({time.time()-t3:.2f}秒)")
        
        # 4. 特征提取
        emit(90, '提取特征数据...')
        print(f"[Main]Step 4/4: 特征提取...")
        t4 = time.time()
        
        # 总是提取真实特征
        if model is None:
             raise RuntimeError("AI模型未就绪")
             
        image_info = get_feature.main(image_data[1])
        print(f"[Main]特征提取完成 ({time.time()-t4:.2f}秒)")
        
        # 添加热力图标记
        if heatmap_generated:
            image_info['has_heatmap'] = True
        
        total_time = time.time() - start_time
        print(f"[Main]全部完成! 总耗时: {total_time:.2f}秒")
        print(f"{'='*60}\n")
        
        return image_data[1] + '.png', image_info
        
    except Exception as e:
        print(f"[Main]处理失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    pass
