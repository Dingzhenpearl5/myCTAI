"""
简单测试Flask后端
"""
import requests

url = "http://127.0.0.1:5003"

print("测试Flask服务器...")
try:
    response = requests.get(url, timeout=5)
    print(f"✅ 服务器响应: HTTP {response.status_code}")
    if response.status_code == 302:
        print("   重定向到:", response.headers.get('Location'))
except Exception as e:
    print(f"❌ 错误: {e}")
