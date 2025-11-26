"""
临时debug启动脚本 - 用于查看详细错误
"""
import sys
import os

# 切换到CTAI_flask目录
os.chdir(r'C:\Users\Masoa\OneDrive\work\CTAI\CTAI_flask')
sys.path.insert(0, r'C:\Users\Masoa\OneDrive\work\CTAI\CTAI_flask')

# 导入app
from app import app, init_model
from flask import current_app

print("🔧 初始化模型...")
with app.app_context():
    current_app.model = init_model()

print("🚀 启动Flask...")
print("📍 服务器地址: http://127.0.0.1:5003")

# 不使用debug模式，但启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)
app.logger.setLevel(logging.DEBUG)

try:
    app.run(host='127.0.0.1', port=5003, debug=False, use_reloader=False)
except Exception as e:
    print(f"\n❌ Flask启动失败: {e}")
    import traceback
    traceback.print_exc()
    input("按Enter键退出...")
