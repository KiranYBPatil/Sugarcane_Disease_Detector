@echo off
title Sugarcane Disease WebApp Starter

:: --- CONFIGURATION ---
set PROJECT_DIR=C:\Users\Kiran Patil\sugarcane-disease-webapp1
set CONDA_ENV_NAME=sugarcane_env
:: *** CRITICAL FIX: CHANGED PATH TO THE ANACONDA INSTALLATION THAT CONTAINS sugarcane_env ***
set ANACONDA_ROOT=C:\Users\Kiran Patil\anaconda3
set BACKEND_PORT=8000
set FRONTEND_PORT=5173

echo ==============================================
echo Starting Sugarcane Disease Detection System...
echo ==============================================

:: --- 1. INITIALIZE CONDA BASE ENVIRONMENT (Must use the path where the environment lives) ---
echo Initializing Conda Base at: %ANACONDA_ROOT%
call "%ANACONDA_ROOT%\Scripts\activate.bat"

:: --- 2. ACTIVATE TARGET ENVIRONMENT ---
echo Activating target environment: %CONDA_ENV_NAME%
call conda activate %CONDA_ENV_NAME%

if errorlevel 1 (
    echo.
    echo ERROR: Could not activate Conda environment "%CONDA_ENV_NAME%".
    echo Please ensure the environment exists and the ANACONDA_ROOT path is correct.
    pause
    goto :eof
)

:: --- START BACKEND (FastAPI) in a new CMD window ---
:: This starts the uvicorn server, ensuring it also activates the correct environment.
echo Starting FastAPI Backend (Port %BACKEND_PORT%)...
start "Backend Server" cmd /k "cd /d "%PROJECT_DIR%\backend\app" && call "%ANACONDA_ROOT%\Scripts\activate.bat" && call conda activate %CONDA_ENV_NAME% && uvicorn main:app --reload --host 0.0.0.0 --port %BACKEND_PORT%"

:: Give the backend a few seconds to fully initialize
timeout /t 5 /nobreak >nul

:: --- START FRONTEND (React) in a new CMD window ---
echo Starting React Frontend (Port %FRONTEND_PORT%)...
start "Frontend App" cmd /k "cd /d "%PROJECT_DIR%\frontend" && npm run dev"

:: --- OPEN BROWSER ---
timeout /t 3 /nobreak >nul
echo Opening browser tabs...
start "" "http://localhost:%FRONTEND_PORT%"

echo.
echo ==============================================
echo ✅ Backend running on: http://localhost:%BACKEND_PORT%
echo 🌐 Frontend running on: http://localhost:%FRONTEND_PORT%
echo ==============================================
echo Both CMD windows will keep running separately. Close them to stop the servers.
echo.
pause
