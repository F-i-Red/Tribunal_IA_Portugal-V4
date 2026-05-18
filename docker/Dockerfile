FROM python:3.12-slim

LABEL maintainer="Tribunal IA Portugal V6"
LABEL description="Simulador judicial com RAG Híbrido, LangGraph e FastAPI"

# Dependências sistema para PyMuPDF e compilação
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libmupdf-dev curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependências Python (em camadas para cache eficiente)
COPY requirements.txt .

# Instalar base primeiro (mais estável)
RUN pip install --no-cache-dir \
    httpx python-dotenv pydantic>=2.7.0 pydantic-settings tenacity \
    streamlit typer rich structlog fastapi uvicorn[standard] \
    python-multipart PyMuPDF reportlab \
    pytest pytest-asyncio

# sentence-transformers (pesado — camada separada)
RUN pip install --no-cache-dir sentence-transformers

# LangGraph
RUN pip install --no-cache-dir langgraph langchain-core

# Copiar código
COPY src/ ./src/
COPY app.py main.py api_server.py verificar.py gerir_base.py ./

# Pastas persistentes (montadas via volumes)
RUN mkdir -p \
    data/leis data/jurisprudencia data/precedentes data/tedh \
    output_atas logs src/cache/data src/historico/data

# Não correr como root
RUN useradd -m -u 1000 tribunal && chown -R tribunal:tribunal /app
USER tribunal

EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default: Streamlit. Para API: docker run ... python api_server.py
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false", \
     "--theme.primaryColor=#1a3a5c", \
     "--theme.backgroundColor=#ffffff", \
     "--theme.secondaryBackgroundColor=#f4f6f9"]
