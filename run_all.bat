@echo off
REM Complete pipeline execution script for Windows
REM This script runs the entire Academic Performance Predictor pipeline

echo ==========================================
echo Academic Performance Predictor Pipeline
echo ==========================================
echo.

REM Check if virtual environment is activated
if "%VIRTUAL_ENV%"=="" (
    echo ERROR: Virtual environment not activated!
    echo Please run: venv\Scripts\activate
    exit /b 1
)

echo Step 1/6: Generating synthetic data...
python src\synthetic_data_generator.py --num_students 500
if errorlevel 1 exit /b 1
echo √ Data generation complete
echo.

echo Step 2/6: Preprocessing data...
python src\data_preprocessing.py --input data\raw\student_activity_logs.csv
if errorlevel 1 exit /b 1
echo √ Preprocessing complete
echo.

echo Step 3/6: Training baseline models...
python src\train_baselines.py --models both --save_models
if errorlevel 1 exit /b 1
echo √ Baseline training complete
echo.

echo Step 4/6: Training CNN-GRU model...
python src\train_cnn_gru.py --epochs 50 --batch_size 32
if errorlevel 1 exit /b 1
echo √ CNN-GRU training complete
echo.

echo Step 5/6: Evaluating all models...
python src\evaluate_models.py
if errorlevel 1 exit /b 1
echo √ Evaluation complete
echo.

echo Step 6/6: Running inference (Early Warning System)...
python src\run_inference.py --input data\raw\student_activity_logs.csv
if errorlevel 1 exit /b 1
echo √ Inference complete
echo.

echo ==========================================
echo Pipeline execution complete!
echo ==========================================
echo.
echo Results saved to:
echo   - Model checkpoints: outputs\checkpoints\
echo   - Metrics: outputs\metrics\
echo   - Predictions: outputs\predictions\
echo.
