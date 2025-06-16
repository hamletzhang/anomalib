@echo off
chcp 65001 >nul
echo ============================================================
echo PatchCore 训练 GUI 工具启动脚本
echo ============================================================

echo.
echo 激活anaconda环境...
call conda activate anomalib
if errorlevel 1 (
    echo ❌ 无法激活anomalib环境
    echo 请确保已创建并配置anomalib conda环境
    pause
    exit /b 1
)

echo ✅ 环境激活成功

echo.
echo 启动GUI界面...
python run_gui.py

echo.
echo 程序已退出，按任意键关闭窗口...
pause 