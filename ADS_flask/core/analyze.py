"""
DeepSeek AI 分析模块
"""
import requests
import json
import config

def analyze_diagnosis(image_features, risk_level="未知"):
    """
    使用 DeepSeek API 分析诊断结果
    
    Args:
        image_features (dict): 图像特征数据
        risk_level (str): 初步风险等级
        
    Returns:
        dict: AI分析结果
    """
    if not config.DEEPSEEK_API_KEY:
        return {
            "error": "未配置 DeepSeek API Key",
            "conclusion": "无法进行AI分析",
            "description": "请在配置文件中设置 DEEPSEEK_API_KEY"
        }

    # 构建提示词 - 支持英文 key、中文 key、[中文名, 数值] 格式
    # 中文 key → 英文 key 映射
    _cn_to_en = {
        '肿瘤面积': 'area',
        '肿瘤周长': 'perimeter',
        '似圆度':   'ellipse',
        '灰度均值': 'mean',
        '灰度标准差': 'std',
    }
    # 统一转为英文 key
    normalized = {}
    for k, v in image_features.items():
        en_key = _cn_to_en.get(k, k)
        if isinstance(v, list):
            normalized[en_key] = v[1] if len(v) > 1 else v[0]
        else:
            normalized[en_key] = v

    area      = normalized.get('area', '未知')
    perimeter = normalized.get('perimeter', '未知')
    ellipse   = normalized.get('ellipse', '未知')
    gray_mean = normalized.get('mean', '未知')
    gray_std  = normalized.get('std', '未知')
    
    prompt = f"""
    你是一个专业的直肠肿瘤辅助诊断AI助手。请根据以下CT影像分析特征，生成一份专业的医疗诊断分析报告。
    
    **影像特征数据:**
    - 肿瘤面积: {area} 像素
    - 肿瘤周长: {perimeter} 像素
    - 似圆度: {ellipse} (越接近0越圆，负值表示接近圆形)
    - 灰度均值: {gray_mean}
    - 灰度方差: {gray_std}
    
    **请生成以下 JSON 格式的回复 (不要包含 markdown 格式):**
    {{
        "conclusion": "简短的诊断结论 (如: 疑似直肠恶性肿瘤)",
        "riskLevel": "风险等级 (低/中/高)",
        "description": "详细的病情描述 (100-200字，分析形态、密度等特征)",
        "suggestions": ["建议1", "建议2", "建议3"]
    }}
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个严谨的医疗辅助诊断AI，输出必须是合法的JSON格式。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    try:
        response = requests.post(config.DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # 清理可能的 markdown 标记
            content = content.replace('```json', '').replace('```', '').strip()
            
            return json.loads(content)
        else:
            print(f"[DeepSeek] API Error: {response.text}")
            return {
                "conclusion": "AI分析服务暂时不可用",
                "riskLevel": "未知",
                "description": f"API请求失败: {response.status_code}",
                "suggestions": ["请稍后重试"]
            }
            
    except Exception as e:
        print(f"[DeepSeek] Exception: {e}")
        return {
            "conclusion": "AI分析发生错误",
            "riskLevel": "未知",
            "description": f"系统错误: {str(e)}",
            "suggestions": ["联系管理员检查日志"]
        }