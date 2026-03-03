"""
认证相关路由
"""
import datetime
import hashlib
import uuid
from flask import Blueprint, request, jsonify

from extensions import db
from models import User, Token, LoginAttempt
from utils import success_response, error_response, token_required, log_audit

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/login', methods=['POST', 'OPTIONS'])
def login():
    """用户登录接口"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 1})
    
    try:
        data = request.get_json()
        username = data.get('username', '')
        password = data.get('password', '')
        ip_address = request.remote_addr
        
        print(f"[Login]用户 {username} 从 {ip_address} 尝试登录")
        
        if not username or not password:
            return error_response('用户名和密码不能为空')
        
        # 检查登录失败次数(15分钟内)
        fifteen_minutes_ago = datetime.datetime.now() - datetime.timedelta(minutes=15)
        failed_attempts = LoginAttempt.query.filter(
            LoginAttempt.username == username,
            LoginAttempt.success == False,
            LoginAttempt.attempt_time > fifteen_minutes_ago
        ).count()
        
        if failed_attempts >= 5:
            return error_response('登录失败次数过多,请15分钟后再试')
        
        # 验证用户
        user = User.query.filter_by(username=username).first()
        if not user:
            db.session.add(LoginAttempt(username=username, success=False, ip_address=ip_address))
            db.session.commit()
            return error_response('用户名或密码错误')
        
        # 检查用户是否被禁用
        if user.status == 'disabled':
            return error_response('账户已被禁用，请联系管理员')
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        if user.password != password_hash:
            db.session.add(LoginAttempt(username=username, success=False, ip_address=ip_address))
            db.session.commit()
            return error_response('用户名或密码错误')
        
        # 生成token(24小时有效期)
        token = str(uuid.uuid4())
        expire_time = datetime.datetime.now() + datetime.timedelta(hours=24)
        new_token = Token(
            token=token,
            username=user.username,
            login_time=datetime.datetime.now(),
            expire_time=expire_time
        )
        db.session.add(new_token)
        
        # 记录成功登录
        db.session.add(LoginAttempt(username=username, success=True, ip_address=ip_address))
        db.session.commit()
        
        # 记录审计日志
        log_audit('login', 'auth', target=username, detail={'role': user.role}, username=username)
        
        print(f"[Login] 用户 {username} 登录成功")
        
        return success_response({
            'token': token,
            'username': user.username,
            'name': user.name,
            'role': user.role,
            'permissions': user.get_permissions(),
            'expire_time': expire_time.strftime('%Y-%m-%d %H:%M:%S')
        }, '登录成功')
        
    except Exception as e:
        print(f"[Login] 登录失败: {e}")
        return error_response(str(e))


@auth_bp.route('/refresh-token', methods=['POST', 'OPTIONS'])
@token_required
def refresh_token():
    """刷新Token接口"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 1})
    
    try:
        old_token = request.headers.get('Authorization')
        if old_token and old_token.startswith('Bearer '):
            old_token = old_token[7:]
        
        old_token_record = Token.query.filter_by(token=old_token).first()
        if old_token_record:
            username = old_token_record.username
            db.session.delete(old_token_record)
            
            # 生成新token
            new_token = str(uuid.uuid4())
            expire_time = datetime.datetime.now() + datetime.timedelta(hours=24)
            token_record = Token(
                token=new_token,
                username=username,
                login_time=datetime.datetime.now(),
                expire_time=expire_time
            )
            db.session.add(token_record)
            db.session.commit()
            
            print(f"[Token] 用户 {username} 刷新了Token")
            
            return success_response({
                'token': new_token,
                'expire_time': expire_time.strftime('%Y-%m-%d %H:%M:%S')
            }, 'Token刷新成功')
        
        return error_response('Token不存在')
        
    except Exception as e:
        print(f"[Token] 刷新失败: {e}")
        return error_response(str(e))


@auth_bp.route('/check-auth', methods=['GET', 'OPTIONS'])
@token_required
def check_auth():
    """检查认证状态"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 1})
    
    try:
        user = request.current_user
        return success_response({
            'username': user.username,
            'name': user.name,
            'role': user.role,
            'permissions': user.get_permissions()
        })
    except Exception as e:
        print(f"[Auth] 验证失败: {e}")
        return error_response(str(e))


@auth_bp.route('/logout', methods=['POST', 'OPTIONS'])
@token_required
def logout():
    """用户登出接口"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 1})
    
    try:
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            token = token[7:]
        
        if token:
            token_record = Token.query.filter_by(token=token).first()
            if token_record:
                username = token_record.username
                log_audit('logout', 'auth', target=username, username=username)
                db.session.delete(token_record)
                db.session.commit()
                print(f"[Logout] 用户 {username} 已登出")
        
        return success_response(message='登出成功')
        
    except Exception as e:
        return error_response(str(e))
