import requests
import traceback

url = 'http://127.0.0.1:5003/upload'
test_file = r'C:\Users\Masoa\OneDrive\work\CTAI\CTAI_flask\data\20014.dcm'

print("="*60)
print("调试文件上传")
print("="*60)

try:
    with open(test_file, 'rb') as f:
        files = {'file': (test_file.split('\\')[-1], f, 'application/dicom')}
        response = requests.post(url, files=files)
        
        print(f"状态码: {response.status_code}")
        print(f"响应头: {response.headers}")
        print(f"\n响应内容:")
        print(response.text)
        
        if response.status_code == 200:
            print("\n✅ 上传成功!")
            print(response.json())
        else:
            print(f"\n❌ 上传失败 (状态码: {response.status_code})")
            
except Exception as e:
    print(f"❌ 请求失败: {e}")
    traceback.print_exc()
