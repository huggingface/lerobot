@echo off
REM Script pour supprimer facilement les datasets LeRobot
REM Double-cliquez sur ce fichier ou appelez-le depuis le terminal

cd /d "%~dp0"
call conda activate lerobot 2>nul
if errorlevel 1 (
    echo Activation de l'environnement conda 'lerobot' echouee
    echo Tentative d'execution avec Python par defaut...
)

python delete_datasets.py %*

if errorlevel 1 (
    echo.
    echo Une erreur s'est produite.
    pause
)
