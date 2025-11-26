from core import process, predict, get_feature
import time


def c_main(path, model):
    print(f"\n{'='*60}")
    print(f"[Main] 开始处理: {path}")
    start_time = time.time()
    
    try:
        # 1. 预处理
        print(f"[Main] Step 1/4: 预处理图像...")
        t1 = time.time()
        image_data = process.pre_process(path)
        print(f"[Main] ✅ 预处理完成 ({time.time()-t1:.2f}秒)")
        
        # 2. 模型预测
        print(f"[Main] Step 2/4: 模型预测...")
        t2 = time.time()
        predict.predict(image_data, model)
        print(f"[Main] ✅ 预测完成 ({time.time()-t2:.2f}秒)")
        
        # 3. 后处理
        print(f"[Main] Step 3/4: 后处理...")
        t3 = time.time()
        process.last_process(image_data[1])
        print(f"[Main] ✅ 后处理完成 ({time.time()-t3:.2f}秒)")
        
        # 4. 特征提取
        print(f"[Main] Step 4/4: 特征提取...")
        t4 = time.time()
        image_info = get_feature.main(image_data[1])
        print(f"[Main] ✅ 特征提取完成 ({time.time()-t4:.2f}秒)")
        
        total_time = time.time() - start_time
        print(f"[Main] 🎉 全部完成! 总耗时: {total_time:.2f}秒")
        print(f"{'='*60}\n")
        
        return image_data[1] + '.png', image_info
        
    except Exception as e:
        print(f"[Main] ❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    pass
