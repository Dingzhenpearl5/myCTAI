"""
测试Flask后端是否正常运行
"""
import requests
import os

# 测试后端健康检查
backend_url = "http://127.0.0.1:5003"

print("="*60)
print("测试Flask后端连接")
print("="*60)

try:
    # 测试根路径
    response = requests.get(backend_url, timeout=5)
    print(f"✅ 后端正在运行!")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.text[:200] if response.text else 'OK'}")
except requests.exceptions.ConnectionError:
    print(f"❌ 无法连接到后端: {backend_url}")
    print("请确保Flask服务器正在运行")
    exit(1)
except Exception as e:
    print(f"❌ 错误: {e}")
    exit(1)

print("\n" + "="*60)
print("测试文件上传功能")
print("="*60)

# 检查测试DICOM文件
test_dcm = r"C:\Users\Masoa\OneDrive\work\CTAI\CTAI_flask\data\20014.dcm"
if not os.path.exists(test_dcm):
    print(f"❌ 测试文件不存在: {test_dcm}")
    exit(1)

print(f"使用测试文件: {test_dcm}")

try:
    # 上传DICOM文件
    with open(test_dcm, 'rb') as f:
        files = {'file': (os.path.basename(test_dcm), f, 'application/dicom')}
        response = requests.post(f"{backend_url}/upload", files=files, timeout=30)
    
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 上传成功!")
        print(f"响应数据: {result}")
        
        if result.get('status') == 1:
            print(f"\n处理结果:")
            print(f"  图像URL: {result.get('image_url')}")
            print(f"  标注URL: {result.get('draw_url')}")
            print(f"  图像信息: {result.get('image_info')}")
        else:
            print(f"⚠️ 处理失败")
    else:
        print(f"❌ 上传失败: {response.text}")
        
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("测试完成")
print("="*60)
