"""
诊断相关路由
"""
import os
import json
import shutil
import datetime
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app

from extensions import db, socketio, emit_progress
from models import DiagnosisRecord, Patient, Token
from utils import (
    success_response, error_response, paginate_response,
    token_required, get_pagination_params, log_audit, allowed_file
)
import config
import core.main

diagnosis_bp = Blueprint('diagnosis', __name__)


@diagnosis_bp.route('/upload', methods=['GET', 'POST'])
@token_required
def upload_file():
    """上传CT图像接口 - 仅上传和预处理，不进行预测"""
    print(f"\n{'='*60}")
    print(f"[Upload] 收到上传请求")
    
    try:
        if 'file' not in request.files:
            return error_response('未选择文件')
        
        file = request.files['file']
        
        if file.filename == '':
            return error_response('未选择文件')
        
        print(f"[Upload] 文件名: {file.filename}")
        print(f"[Upload] 时间: {datetime.datetime.now()}")
        
        if file and allowed_file(file.filename, config.ALLOWED_EXTENSIONS):
            # 保存文件
            src_path = os.path.join(config.UPLOAD_FOLDER, file.filename)
            print(f"[Upload] 保存到: {src_path}")
            file.save(src_path)
            
            # 复制到临时目录
            tmp_ct_dir = os.path.join(config.BASE_DIR, 'tmp', 'ct')
            shutil.copy(src_path, tmp_ct_dir)
            image_path = os.path.join(tmp_ct_dir, file.filename)
            
            # 仅进行预处理，生成预览图
            print(f"[Upload] 预处理图像...")
            from core import process
            image_data = process.pre_process(image_path)
            pid = image_data[1]
            
            result = {
                'image_url': f'{config.SERVER_URL}/tmp/image/{pid}.png',
                'message': '上传成功，请点击开始诊断'
            }
            
            # 记录审计日志
            log_audit('upload', 'diagnosis', target=file.filename)
            
            print(f"[Upload] 上传成功!")
            print(f"{'='*60}\n")
            return success_response(result, '上传成功')
        else:
            print(f"[Upload] 文件格式不支持")
            return error_response('仅支持.dcm文件')
            
    except Exception as e:
        print(f"[Upload] 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return error_response(str(e))


@diagnosis_bp.route('/predict', methods=['POST', 'OPTIONS'])
@token_required
def predict_image():
    """AI预测接口 - 对已上传的图像进行预测分析"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 1})
    
    print(f"\n{'='*60}")
    print(f"[Predict] 收到预测请求")
    
    try:
        data = request.get_json()
        image_url = data.get('imageUrl', '')
        
        # 从URL中提取文件名
        if '/tmp/image/' in image_url:
            filename = image_url.split('/tmp/image/')[-1].replace('.png', '')
        else:
            return error_response('无效的图像URL')
        
        print(f"[Predict] 处理文件: {filename}")
        
        emit_progress(10, '正在准备分析...')
        
        # 检查原始dcm文件是否存在
        dcm_path = Path(config.BASE_DIR) / 'tmp' / 'ct' / f'{filename}.dcm'
        if not dcm_path.exists():
            socketio.emit('error', '原始图像文件不存在')
            return error_response('原始图像文件不存在')
        
        # 执行预测
        print(f"[Predict] 开始AI分析...")
        pid, image_info = core.main.c_main(str(dcm_path), current_app.model, emit_progress)
        
        emit_progress(100, '分析完成')
        
        result = {
            'image_url': f'{config.SERVER_URL}/tmp/image/{pid}',
            'draw_url': f'{config.SERVER_URL}/tmp/draw/{pid}',
            'image_info': image_info
        }
        
        # 添加热力图URL
        if image_info.get('has_heatmap'):
            # pid 包含 .png，需去掉扩展名用于 heatmap 路径 (假设 heatmap 是 .png)
            # 实际上 pid = "filename.png"
            file_stem = pid.replace('.png', '')
            result['heatmap_url'] = f'{config.SERVER_URL}/tmp/heatmap/{file_stem}_heatmap.png'
        
        # 保存诊断记录到数据库
        record_id = None
        try:
            patient_id = data.get('patientId')
            doctor_username = None
            if hasattr(request, 'current_user') and request.current_user:
                doctor_username = request.current_user.username
            
            # 从新格式 {key: [中文名, 数值]} 中提取数值
            def _val(info, key):
                v = info.get(key, 0)
                return v[1] if isinstance(v, list) else v

            record = DiagnosisRecord(
                patient_id=patient_id,
                doctor_username=doctor_username,
                filename=f'{filename}.dcm',
                image_url=result['image_url'],
                draw_url=result['draw_url'],
                area=_val(image_info, 'area'),
                perimeter=_val(image_info, 'perimeter'),
                features=json.dumps(image_info, ensure_ascii=False)
            )
            db.session.add(record)
            db.session.commit()
            record_id = record.id
            print(f"[Predict] 诊断记录已保存，ID: {record.id}")
        except Exception as save_err:
            print(f"[Predict] 保存诊断记录失败: {save_err}")
        
        result['record_id'] = record_id
        
        # 通过 Socket 发送结果
        socketio.emit('result', {
            'url2': result['draw_url'],
            'feature_list': image_info,
            'area': _val(image_info, 'area'),
            'perimeter': _val(image_info, 'perimeter'),
            'record_id': record_id
        })
        
        print(f"[Predict] 预测完成!")
        print(f"{'='*60}\n")
        return success_response(result)
        
    except Exception as e:
        print(f"[Predict] 预测失败: {e}")
        socketio.emit('error', str(e))
        import traceback
        traceback.print_exc()
        return error_response(str(e))


@diagnosis_bp.route('/analyze', methods=['POST', 'OPTIONS'])
@token_required
def analyze_condition():
    """AI病情分析接口 (接入 DeepSeek)"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 1})
    
    try:
        data = request.get_json()
        features = data.get('features', {})
        
        # 简单校验
        if not features:
            return error_response('缺少特征数据')
        
        print(f"[Analyze] 收到特征数据: {features}")
        print(f"[Analyze] 开始调用 DeepSeek 分析...")
        
        from core.analyze import analyze_diagnosis
        result = analyze_diagnosis(features)
        
        # 添加分析时间
        result['analysisTime'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"[Analyze] 分析完成: {result.get('conclusion')}")
        return success_response(result)
        
    except Exception as e:
        print(f"[Analyze] 分析失败: {e}")
        return error_response(str(e))


@diagnosis_bp.route('/diagnosis/history', methods=['GET', 'OPTIONS'])
def get_diagnosis_history():
    """获取诊断历史记录"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 1})
    
    try:
        patient_id = request.args.get('patient_id')
        page, per_page = get_pagination_params()
        
        query = DiagnosisRecord.query.order_by(DiagnosisRecord.created_at.desc())
        
        if patient_id:
            query = query.filter_by(patient_id=patient_id)
        
        # 分页
        total = query.count()
        records = query.offset((page - 1) * per_page).limit(per_page).all()
        
        history_list = [record.to_dict() for record in records]
        
        return paginate_response(history_list, total, page, per_page)
        
    except Exception as e:
        return error_response(str(e))


@diagnosis_bp.route('/diagnosis/<int:record_id>', methods=['GET', 'OPTIONS'])
def get_diagnosis_detail(record_id):
    """获取单条诊断记录详情"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 1})
    
    try:
        record = DiagnosisRecord.query.get(record_id)
        if not record:
            return error_response('记录不存在', 404)
        
        data = record.to_dict(include_patient=True)
        # 添加额外的患者信息
        if record.patient:
            data['patient_gender'] = record.patient.gender
            data['patient_age'] = record.patient.age
            data['patient_phone'] = record.patient.phone
        if record.doctor:
            data['doctor_name'] = record.doctor.name
            
        return success_response(data)
        
    except Exception as e:
        return error_response(str(e))


@diagnosis_bp.route('/diagnosis/<int:record_id>/record', methods=['POST', 'OPTIONS'])
@token_required
def save_doctor_record(record_id):
    """保存医生的诊断记录"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 1})
    
    try:
        record = DiagnosisRecord.query.get(record_id)
        if not record:
            return error_response('诊断记录不存在', 404)
        
        data = request.get_json()
        
        if 'doctor_diagnosis' in data:
            record.doctor_diagnosis = data['doctor_diagnosis']
        if 'doctor_suggestion' in data:
            record.doctor_suggestion = data['doctor_suggestion']
        if 'diagnosis_conclusion' in data:
            record.diagnosis_conclusion = data['diagnosis_conclusion']
        
        if not record.doctor_username and hasattr(request, 'current_user'):
            record.doctor_username = request.current_user.username
        
        db.session.commit()
        
        # 记录审计日志
        log_audit('update', 'diagnosis', target=record_id)
        
        print(f"[Diagnosis] 诊断记录 {record_id} 已更新医生记录")
        
        return success_response({
            'id': record.id,
            'doctor_diagnosis': record.doctor_diagnosis,
            'doctor_suggestion': record.doctor_suggestion,
            'diagnosis_conclusion': record.diagnosis_conclusion,
            'updated_at': record.updated_at.strftime('%Y-%m-%d %H:%M:%S') if record.updated_at else None
        }, '诊断记录保存成功')
        
    except Exception as e:
        print(f"[Diagnosis] 保存医生记录失败: {e}")
        return error_response(str(e))


@diagnosis_bp.route('/statistics', methods=['GET', 'OPTIONS'])
def get_statistics():
    """获取系统统计数据"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 1})
    
    try:
        from models import User
        from datetime import timedelta
        
        total_diagnoses = DiagnosisRecord.query.count()
        total_patients = Patient.query.count()
        total_users = User.query.count()
        
        # 今日诊断数
        today = datetime.datetime.now().date()
        today_start = datetime.datetime.combine(today, datetime.time.min)
        today_diagnoses = DiagnosisRecord.query.filter(
            DiagnosisRecord.created_at >= today_start
        ).count()
        
        # 最近7天每日诊断数
        daily_stats = []
        for i in range(6, -1, -1):
            day = today - timedelta(days=i)
            day_start = datetime.datetime.combine(day, datetime.time.min)
            day_end = datetime.datetime.combine(day, datetime.time.max)
            count = DiagnosisRecord.query.filter(
                DiagnosisRecord.created_at >= day_start,
                DiagnosisRecord.created_at <= day_end
            ).count()
            daily_stats.append({
                'date': day.strftime('%m-%d'),
                'count': count
            })
        
        # 最近诊断记录
        recent_records = DiagnosisRecord.query.order_by(
            DiagnosisRecord.created_at.desc()
        ).limit(5).all()
        
        recent_diagnoses = []
        for record in recent_records:
            patient_name = '未知患者'
            part = '直肠'
            if record.patient:
                patient_name = record.patient.name
                part = record.patient.part or '直肠'
            
            if record.created_at:
                time_diff = datetime.datetime.now() - record.created_at
                if time_diff.days > 0:
                    time_str = f"{time_diff.days}天前"
                elif time_diff.seconds < 3600:
                    time_str = f"{time_diff.seconds // 60}分钟前"
                else:
                    time_str = f"{time_diff.seconds // 3600}小时前"
            else:
                time_str = '未知'
            
            recent_diagnoses.append({
                'id': record.id,
                'patientName': patient_name,
                'part': part,
                'time': time_str,
                'status': '已完成'
            })
        
        return success_response({
            'total_diagnoses': total_diagnoses,
            'total_patients': total_patients,
            'total_users': total_users,
            'today_diagnoses': today_diagnoses,
            'daily_stats': daily_stats,
            'trend': daily_stats,
            'recent_diagnoses': recent_diagnoses
        })
        
    except Exception as e:
        return error_response(str(e))
