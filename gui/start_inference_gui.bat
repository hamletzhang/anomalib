@echo off
chcp 65001 > nul
title PatchCore 推理界面启动器

echo ========================================
echo  PatchCore 异常检测推理工具
echo ========================================
echo.

:: 检查是否在conda环境中
if "%CONDA_DEFAULT_ENV%"=="" (
    echo ⚠️  警告: 未检测到conda环境
    echo 正在尝试激活anomalib环境...
    call conda activate anomalib
    if errorlevel 1 (
        echo ❌ 无法激活anomalib环境
        echo 请确保已安装anaconda并创建了anomalib环境
        pause
        exit /b 1
    )
) else (
    echo ✅ 当前conda环境: %CONDA_DEFAULT_ENV%
)

echo.
echo 🚀 启动推理界面...
echo.

:: 运行推理GUI启动脚本
python run_inference_gui.py

:: 如果出错，暂停以查看错误信息
if errorlevel 1 (
    echo.
    echo ❌ 启动失败，请检查错误信息
    pause
)

echo.
echo 👋 程序已退出
pause 