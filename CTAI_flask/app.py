# import datetime
# import logging as rel_log
# import os
# import shutil
# from datetime import timedelta

# import torch
# from flask import *

# import core.main
# import core.net.unet as net

# UPLOAD_FOLDER = r'./uploads'

# ALLOWED_EXTENSIONS = set(['dcm'])
# app = Flask(__name__)
# app.secret_key = 'secret!'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# werkzeug_logger = rel_log.getLogger('werkzeug')
# werkzeug_logger.setLevel(rel_log.ERROR)

# # 解决缓存刷新问题
# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)


# # 添加header解决跨域
# @app.after_request
# def after_request(response):
#     response.headers['Access-Control-Allow-Origin'] = '*'
#     response.headers['Access-Control-Allow-Credentials'] = 'true'
#     response.headers['Access-Control-Allow-Methods'] = 'POST'
#     response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
#     return response


# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# @app.route('/')
# def hello_world():
#     return redirect(url_for('static', filename='./index.html'))


# @app.route('/upload', methods=['GET', 'POST'])
# def upload_file():
#     file = request.files['file']
#     print(datetime.datetime.now(), file.filename)
#     if file and allowed_file(file.filename):
#         src_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(src_path)
#         shutil.copy(src_path, './tmp/ct')
#         image_path = os.path.join('./tmp/ct', file.filename)
#         # print(image_path)
#         pid, image_info = core.main.c_main(image_path, current_app.model)
#         return jsonify({'status': 1,
#                         'image_url': 'http://127.0.0.1:5003/tmp/image/' + pid,
#                         'draw_url': 'http://127.0.0.1:5003/tmp/draw/' + pid,
#                       'image_info': image_info
#                        })


#     return jsonify({'status': 0})


# @app.route("/download", methods=['GET'])
# def download_file():
#     # 需要知道2个参数, 第1个参数是本地目录的path, 第2个参数是文件名(带扩展名)
#     return send_from_directory('data', 'testfile.zip', as_attachment=True)


# # show photo
# @app.route('/tmp/<path:file>', methods=['GET'])
# def show_photo(file):
#     # print(file)
#     if request.method == 'GET':
#         if file is None:
#             pass
#         else:
#             image_data = open(f'tmp/{file}', "rb").read()
#             response = make_response(image_data)
#             response.headers['Content-Type'] = 'image/png'
#             return response
#     else:
#         pass


# def init_model():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = net.Unet(1, 1).to(device)
#     if torch.cuda.is_available():
#         model.load_state_dict(torch.load("./core/net/model.pth"))
#     else:
#         model.load_state_dict(torch.load("./core/net/model.pth", map_location='cpu'))
#     model.eval()
#     return model


# if __name__ == '__main__':
#     with app.app_context():
#         current_app.model = init_model()
#     app.run(host='127.0.0.1', port=5003, debug=True)
import datetime
import logging as rel_log
import os
import shutil
from datetime import timedelta

import torch
from flask import *

import core.main
import core.net.unet as net

UPLOAD_FOLDER = r'./uploads'
ALLOWED_EXTENSIONS = set(['dcm'])
app = Flask(__name__)
app.secret_key = 'secret!'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

werkzeug_logger = rel_log.getLogger('werkzeug')
werkzeug_logger.setLevel(rel_log.ERROR)

# 解决缓存刷新问题
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)


# 添加header解决跨域
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    return response


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def hello_world():
    return redirect(url_for('static', filename='./index.html'))


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    print(f"\n{'='*60}")
    print(f"[Upload] 收到上传请求")
    
    try:
        file = request.files['file']
        print(f"[Upload] 文件名: {file.filename}")
        print(f"[Upload] 时间: {datetime.datetime.now()}")
        
        if file and allowed_file(file.filename):
            # 保存文件
            src_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            print(f"[Upload] 保存到: {src_path}")
            file.save(src_path)
            
            # 复制到临时目录
            shutil.copy(src_path, './tmp/ct')
            image_path = os.path.join('./tmp/ct', file.filename)
            
            # 处理图像
            print(f"[Upload] 开始处理图像...")
            pid, image_info = core.main.c_main(image_path, current_app.model)
            
            result = {
                'status': 1,
                'image_url': 'http://127.0.0.1:5003/tmp/image/' + pid,
                'draw_url': 'http://127.0.0.1:5003/tmp/draw/' + pid,
                'image_info': image_info
            }
            print(f"[Upload] ✅ 处理成功!")
            print(f"{'='*60}\n")
            return jsonify(result)
        else:
            print(f"[Upload] ❌ 文件格式不支持")
            return jsonify({'status': 0, 'error': '仅支持.dcm文件'})
            
    except Exception as e:
        print(f"[Upload] ❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 0, 'error': str(e)})



@app.route("/download", methods=['GET'])
def download_file():
    return send_from_directory('data', 'testfile.zip', as_attachment=True)


@app.route('/tmp/<path:file>', methods=['GET'])
def show_photo(file):
    if request.method == 'GET':
        if file is None:
            pass
        else:
            image_data = open(f'tmp/{file}', "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response
    else:
        pass


# 加载真实的UNet模型
def init_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net.Unet(1, 1).to(device)
    
    model_path = "core/net/model.pth"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件未找到: {model_path}")
        print("请确保已将训练好的模型复制到此路径")
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
        print(f"✅ 模型已加载 (GPU): {model_path}")
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"✅ 模型已加载 (CPU): {model_path}")
    
    model.eval()
    return model


if __name__ == '__main__':
    try:
        import logging
        logging.basicConfig(level=logging.INFO)
        
        print("🔧 开始初始化模型...")
        with app.app_context():
            current_app.model = init_model()
        print("🚀 启动Flask服务器...")
        print("📍 服务器地址: http://127.0.0.1:5003")
        app.run(host='127.0.0.1', port=5003, debug=False, use_reloader=False)
    except Exception as e:
        print(f"❌ Flask启动失败: {e}")
        import traceback
        traceback.print_exc()


