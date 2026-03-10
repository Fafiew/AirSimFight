@echo off
REM Setup script for AirSimFight (Windows)

echo AirSimFight Setup
echo =================

REM Create directories
echo Creating directories...
if not exist models mkdir models
if not exist logs mkdir logs
if not exist scenarios mkdir scenarios
if not exist config mkdir config

REM Install Python dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo Setup complete!
echo.
echo To run pretraining:
echo   python -m rl.trainer_pretrain --config config\default.yaml --device cpu --timesteps 1000000 --output models\attacker_pretrained --safety-check --force
echo.
echo To run dual training:
echo   python -m rl.trainer_dual --config config\default.yaml --device cuda --timesteps 2000000 --attacker-model models\attacker_pretrained.zip --fine-tune-attacker
echo.
echo To visualize:
echo   python tools\visualize.py --scenario scenarios\example_visualization.json
echo.
echo To run tests:
echo   pytest -q
