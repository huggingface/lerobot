@echo off
REM Simple script to run the trained model on the robot
REM This handles dataset cleanup automatically

echo Cleaning up previous test dataset...
rmdir /s /q "C:\Users\picip\.cache\huggingface\lerobot\Zarax\eval_test" 2>NUL

echo Starting robot with trained model...
lerobot-record --config_path ./config/eval/zarax_eval_simple.yaml

echo.
echo Robot stopped.
