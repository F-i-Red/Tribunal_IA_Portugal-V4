#!/bin/bash
# Tribunal IA Portugal V6 — Arranque da interface web
# Uso: ./iniciar_interface.sh [porta]
set -e
PORT=${1:-8501}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "🏛️  TRIBUNAL IA PORTUGAL V6  🇵🇹"
echo "  RAG Híbrido + Reranking · LangGraph · FastAPI · TEDH"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# .env
if [ ! -f ".env" ]; then
    echo "⚠️  A criar .env a partir de .env.example..."
    cp .env.example .env
    echo "   Edita .env com a tua OPENROUTER_API_KEY (ou usa BACKEND=ollama)"
fi

# Python
if ! command -v python3 &>/dev/null; then
    echo "❌ Python 3 não encontrado."
    exit 1
fi

# Dependências obrigatórias
echo "🔍 A verificar dependências..."
python3 -c "import streamlit, pydantic_settings, httpx, tenacity" 2>/dev/null || {
    echo "📦 A instalar dependências base..."
    pip install streamlit pydantic-settings httpx tenacity python-dotenv
}

# sentence-transformers (opcional mas recomendado)
python3 -c "import sentence_transformers" 2>/dev/null || {
    echo "ℹ️  sentence-transformers não instalado."
    echo "   Para RAG híbrido: pip install sentence-transformers"
    echo "   (a iniciar com RAG BM25 como fallback)"
}

# LangGraph (opcional)
python3 -c "import langgraph" 2>/dev/null || {
    echo "ℹ️  LangGraph não instalado — usando orquestração imperativa."
    echo "   Para LangGraph: pip install langgraph langchain-core"
}

# Criar pastas
mkdir -p data/leis data/jurisprudencia data/precedentes data/tedh
mkdir -p output_atas logs src/cache/data src/historico/data

# Info RAG
LEIS=$(find data/leis -name "*.txt" 2>/dev/null | wc -l | tr -d ' ')
TEDH=$(find data/tedh -name "*.txt" 2>/dev/null | wc -l | tr -d ' ')
[ "$LEIS" -eq 0 ] && echo "ℹ️  data/leis/ vazia — RAG sem contexto jurídico PT"
[ "$TEDH" -eq 0 ] && echo "ℹ️  data/tedh/ vazia — análise TEDH inactiva"

# Ollama check
BACKEND=$(grep "^BACKEND=" .env 2>/dev/null | cut -d'=' -f2)
if [ "$BACKEND" = "ollama" ]; then
    echo ""
    echo "🖥️  Modo Ollama (soberania de dados)"
    curl -s http://localhost:11434/api/tags >/dev/null 2>&1 \
        && echo "   ✅ Ollama disponível" \
        || echo "   ⚠️  Ollama não está em execução. Inicia com: ollama serve"
fi

echo ""
echo "🚀 A iniciar na porta $PORT..."
echo "   Streamlit : http://localhost:$PORT"
echo "   Para a API: python api_server.py"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

streamlit run app.py \
    --server.port "$PORT" \
    --server.headless true \
    --browser.gatherUsageStats false \
    --theme.primaryColor "#1a3a5c" \
    --theme.backgroundColor "#ffffff" \
    --theme.secondaryBackgroundColor "#f4f6f9" \
    --theme.textColor "#1a1a1a"
