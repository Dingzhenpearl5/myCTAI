import requests

# 测试20016.dcm文件
file_path = r'C:\Users\Masoa\OneDrive\work\CTAI\CTAI_flask\data\20016.dcm'
url = 'http://127.0.0.1:5003/upload'

print(f"测试文件: {file_path}")
with open(file_path, 'rb') as f:
    response = requests.post(url, files={'file': f})

print(f"状态码: {response.status_code}")
print(f"响应数据: {response.json()}")

if response.status_code == 200:
    data = response.json()
    print("\n处理结果:")
    print(f"  图像URL: {data.get('image_url')}")
    print(f"  标注URL: {data.get('draw_url')}")
    print(f"  图像信息: {data.get('image_info')}")
