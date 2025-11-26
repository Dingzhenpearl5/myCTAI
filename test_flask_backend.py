"""
测试Flask后端功能
使用DICOM文件测试推理接口
"""
import requests
import os

# 配置
FLASK_URL = "http://127.0.0.1:5003"
TEST_DICOM_PATH = r"C:\Users\Masoa\OneDrive\work\CTAI\CTAI_flask\data\20014.dcm"

print("="*60)
print("Flask后端测试")
print("="*60)

# 1. 检查服务器状态
print("\n1. 检查服务器状态...")
try:
    response = requests.get(FLASK_URL, timeout=5)
    if response.status_code == 200:
        print(f"✅ 服务器在线: {FLASK_URL}")
    else:
        print(f"⚠️ 服务器响应异常: {response.status_code}")
except Exception as e:
    print(f"❌ 无法连接到服务器: {e}")
    print("请确保Flask服务器正在运行 (python app.py)")
    exit(1)

# 2. 检查测试文件
print(f"\n2. 检查测试文件...")
if not os.path.exists(TEST_DICOM_PATH):
    print(f"❌ 测试文件不存在: {TEST_DICOM_PATH}")
    print("请提供一个有效的DICOM文件路径")
    exit(1)
else:
    file_size = os.path.getsize(TEST_DICOM_PATH) / 1024  # KB
    print(f"✅ 测试文件: {os.path.basename(TEST_DICOM_PATH)} ({file_size:.1f} KB)")

# 3. 上传文件进行推理
print(f"\n3. 上传文件进行推理...")
try:
    with open(TEST_DICOM_PATH, 'rb') as f:
        files = {'file': (os.path.basename(TEST_DICOM_PATH), f, 'application/dicom')}
        response = requests.post(f"{FLASK_URL}/upload", files=files, timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 推理成功!")
        print(f"   状态: {result.get('status')}")
        print(f"   图像URL: {result.get('image_url')}")
        print(f"   分割URL: {result.get('draw_url')}")
        print(f"   特征信息: {result.get('image_info')}")
    else:
        print(f"❌ 推理失败: HTTP {response.status_code}")
        print(f"   响应: {response.text}")
        
except Exception as e:
    print(f"❌ 上传/推理出错: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("测试完成")
print("="*60)
