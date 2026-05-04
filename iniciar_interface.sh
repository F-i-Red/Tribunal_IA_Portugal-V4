#!/bin/bash
# Tribunal IA Portugal V5 — Arranque da interface web
set -e
PORT=${1:-8501}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "🏛️  TRIBUNAL IA PORTUGAL V5  🇵🇹"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -f ".env" ]; then
    echo "⚠️  Criando .env a partir de .env.example..."
    cp .env.example .env
    echo "   Edita .env com a tua OPENROUTER_API_KEY (ou configura BACKEND=ollama)"
fi

if ! command -v python3 &>/dev/null; then
    echo "❌ Python 3 não encontrado."
    exit 1
fi

echo "🔍 A verificar dependências..."
python3 -c "import streamlit, pydantic_settings, httpx" 2>/dev/null || {
    echo "📦 A instalar dependências..."
    pip install -r requirements.txt
}

# Criar pastas necessárias
mkdir -p data/leis data/jurisprudencia data/precedentes
mkdir -p output_atas logs src/cache/data src/historico/data

LEIS_COUNT=$(find data/leis -name "*.txt" 2>/dev/null | wc -l | tr -d ' ')
if [ "$LEIS_COUNT" -eq 0 ]; then
    echo ""
    echo "ℹ️  data/leis/ está vazia — RAG sem contexto jurídico local."
    echo "   Adiciona ficheiros .txt (Codigo_Penal.txt, etc.) para melhor qualidade."
fi

# Verificar Ollama se configurado
BACKEND=$(grep "^BACKEND=" .env 2>/dev/null | cut -d'=' -f2)
if [ "$BACKEND" = "ollama" ]; then
    echo ""
    echo "🖥️  Modo Ollama (local)"
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "   ✅ Ollama disponível"
    else
        echo "   ⚠️  Ollama não está em execução. Inicia com: ollama serve"
    fi
fi

echo ""
echo "🚀 A iniciar na porta $PORT..."
echo "   URL: http://localhost:$PORT"
echo "   Para parar: Ctrl+C"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

streamlit run app.py \
    --server.port "$PORT" \
    --server.headless true \
    --browser.gatherUsageStats false \
    --theme.primaryColor "#1a3a5c" \
    --theme.backgroundColor "#ffffff" \
    --theme.secondaryBackgroundColor "#f4f6f9" \
    --theme.textColor "#1a1a1a"
