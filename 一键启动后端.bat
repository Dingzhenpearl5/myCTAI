@echo off
chcp 65001 >nul
title 直肠肿瘤辅助诊断系统 - Backend

cd /d "%~dp0"

if not exist "ADS_flask\app.py" (
    echo [错误] 请在项目根目录运行此脚本
    pause
    exit
)

echo 正在激活 design39 环境...
call conda activate design39 || (
    echo [失败] 无法激活环境 design39
    pause
    exit
)

echo 启动 Flask ...
cd ADS_flask
python app.py

echo 服务已结束
pause