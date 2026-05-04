@echo off
REM Tribunal IA Portugal V4 — Arranque Windows

SET PORT=8501
IF NOT "%1"=="" SET PORT=%1

echo.
echo ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
echo   🏛️  TRIBUNAL IA PORTUGAL V4
echo ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
echo.

cd /d "%~dp0"

IF NOT EXIST ".env" (
    echo ⚠️  Ficheiro .env nao encontrado.
    copy .env.example .env
    echo    .env criado. Edita com a tua chave OpenRouter antes de continuar.
    pause
)

echo 🔍 A verificar dependencias...
python -c "import streamlit" 2>NUL
IF ERRORLEVEL 1 (
    echo 📦 A instalar dependencias...
    pip install -r requirements.txt
)

mkdir data\leis 2>NUL
mkdir data\jurisprudencia 2>NUL
mkdir data\precedentes 2>NUL
mkdir output_atas 2>NUL
mkdir logs 2>NUL
mkdir src\cache 2>NUL
mkdir src\cache\data 2>NUL
mkdir src\historico 2>NUL
mkdir src\historico\data 2>NUL

echo.
echo 🚀 A iniciar na porta %PORT%...
echo    URL: http://localhost:%PORT%
echo    Para parar: Ctrl+C
echo.

streamlit run app.py ^
    --server.port %PORT% ^
    --server.headless true ^
    --browser.gatherUsageStats false

pause
