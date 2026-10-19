@echo off
setlocal
REM ============================================================
REM  JAKA Zu 7 follower console launcher  (double-click to run)
REM  Runs with the "jaka" conda env Python 3.10,
REM  because the jkrc SDK only works on Python <= 3.10.
REM ============================================================

REM --- Change these if your paths / controller IP differ ---
set "PYTHON_EXE=C:\Users\Administrator\miniconda3\envs\jaka\python.exe"
set "WORK_DIR=E:\A_practice3\lerobot-main\lerobot-main\src\lerobot\robots\jaka_zu7_follower"
set "JAKA_IP=10.5.5.100"
REM -----------------------------------------------------------

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python not found: %PYTHON_EXE%
    echo Please edit PYTHON_EXE in this .bat to point to your jaka env python.exe.
    pause
    exit /b 1
)

if not exist "%WORK_DIR%\jaka_interactive.py" (
    echo [ERROR] jaka_interactive.py not found under: %WORK_DIR%
    echo Please edit WORK_DIR in this .bat.
    pause
    exit /b 1
)

cd /d "%WORK_DIR%"
"%PYTHON_EXE%" jaka_interactive.py --ip %JAKA_IP%

echo.
pause
