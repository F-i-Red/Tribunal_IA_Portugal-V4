FROM python:3.12-slim

LABEL maintainer="Tribunal IA Portugal V5"
LABEL description="Simulador judicial — Direito Português"

# Dependências do sistema para PyMuPDF e ReportLab
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libmupdf-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY src/ ./src/
COPY app.py main.py verificar.py gerir_base.py ./

# Criar pastas com dados persistentes (montadas via volumes)
RUN mkdir -p \
    data/leis \
    data/jurisprudencia \
    data/precedentes \
    output_atas \
    logs \
    src/cache/data \
    src/historico/data

# Não correr como root
RUN useradd -m -u 1000 tribunal && chown -R tribunal:tribunal /app
USER tribunal

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false", \
     "--theme.primaryColor=#1a3a5c", \
     "--theme.backgroundColor=#ffffff", \
     "--theme.secondaryBackgroundColor=#f4f6f9"]
