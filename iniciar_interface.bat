@echo off
REM Tribunal IA Portugal V6 — Arranque Windows
SET PORT=8501
IF NOT "%1"=="" SET PORT=%1

echo.
echo ================================================================
echo   🏛  TRIBUNAL IA PORTUGAL V6  🇵🇹
echo   RAG Hibrido + Reranking - LangGraph - FastAPI - TEDH
echo ================================================================
echo.

cd /d "%~dp0"

IF NOT EXIST ".env" (
    echo ⚠️  A criar .env...
    copy .env.example .env
    echo    Edita .env com a tua OPENROUTER_API_KEY
    pause
)

echo 🔍 A verificar dependencias...
python -c "import streamlit, pydantic_settings, httpx" 2>NUL || (
    echo 📦 A instalar dependencias base...
    pip install -r requirements.txt
)

REM Criar todas as pastas necessarias
for %%d in (data\leis data\jurisprudencia data\precedentes data\tedh output_atas logs src\cache\data src\historico\data) do (
    if not exist "%%d" mkdir "%%d"
)

echo.
echo 🚀 A iniciar na porta %PORT%...
echo    Streamlit : http://localhost:%PORT%
echo    Para a API: python api_server.py
echo.

streamlit run app.py ^
    --server.port %PORT% ^
    --server.headless true ^
    --browser.gatherUsageStats false ^
    --theme.primaryColor "#1a3a5c" ^
    --theme.backgroundColor "#ffffff" ^
    --theme.secondaryBackgroundColor "#f4f6f9" ^
    --theme.textColor "#1a1a1a"

pause
