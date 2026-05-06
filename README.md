# 🏛️ Tribunal IA Portugal — V6

Simulador judicial com RAG Híbrido, LangGraph, FastAPI, modo contraditório e análise TEDH.

> ⚠️ **Aviso Legal:** Fins exclusivamente educativos. Não constitui parecer jurídico.
> Para situações reais: [Ordem dos Advogados de Portugal](https://www.oa.pt)

---

## 🆕 V6 — O que é novo

| Funcionalidade | V5 | V6 |
|---|---|---|
| RAG BM25 | ✅ | ✅ |
| **RAG Híbrido BM25 + Embeddings** | ❌ | ✅ activo por defeito |
| **Cross-encoder Reranking** | ❌ | ✅ |
| **Modelo PT multilingual (multilingual-e5)** | ❌ | ✅ |
| **LangGraph orquestração** | ❌ | ✅ com fallback imperativo |
| **Modo Contraditório** | ❌ | ✅ utilizador como advogado |
| **FastAPI REST backend** | ❌ | ✅ |
| **Multi-idioma TEDH / ECHR** | ❌ | ✅ |
| **Metadata filtering (diploma + instância)** | ✅ | ✅ melhorado |
| Histórico de casos | ✅ | ✅ |
| Exportação PDF | ✅ | ✅ |
| Docker + Ollama | ✅ | ✅ melhorado |

---

## ⚡ Início Rápido

```bash
git clone https://github.com/F-i-Red/Tribunal_IA_Portugal-V6
cd Tribunal_IA_Portugal-V6
pip install -r requirements.txt
cp .env.example .env
# Edita .env com OPENROUTER_API_KEY
streamlit run app.py
```

---

## 🤖 Backends

### OpenRouter (cloud)
```env
BACKEND=openrouter
OPENROUTER_API_KEY=sk-or-...
MODELO=openrouter/free          # testes gratuitos
MODELO=google/gemini-2.0-flash-001  # produção
```

### Ollama (local — soberania de dados total)
```env
BACKEND=ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODELO=llama3.3:70b
```
```bash
ollama serve && ollama pull llama3.3:70b
```

---

## 🔬 RAG Híbrido V6

Pipeline de 3 fases:

```
Query
  ↓
[1] BM25 → top_k candidatos (lexical, rápido)
  ↓
[2] Embeddings multilinguais → score semântico
    + Fusão RRF (Reciprocal Rank Fusion)
  ↓
[3] Cross-encoder Reranker → top_n final (qualidade máxima)
```

**Modelos de embeddings recomendados para Português:**

| Modelo | Qualidade | Tamanho |
|--------|-----------|---------|
| `intfloat/multilingual-e5-large-instruct` | ⭐⭐⭐⭐⭐ | 560MB |
| `intfloat/multilingual-e5-base` | ⭐⭐⭐⭐ | 280MB |
| `neuralmind/bert-base-portuguese-cased` | ⭐⭐⭐⭐ | 440MB |
| `paraphrase-multilingual-MiniLM-L12-v2` | ⭐⭐⭐ | 118MB |

```env
RAG_MODO=hibrido
RAG_EMBEDDING_MODELO=intfloat/multilingual-e5-large-instruct
RAG_RERANKING=true
RAG_RERANKER_MODELO=cross-encoder/ms-marco-MiniLM-L-6-v2
RAG_TOP_K=15    # candidatos antes do reranking
RAG_TOP_N=6     # resultado final
```

---

## 🔀 LangGraph

O pipeline judicial é agora um **grafo declarativo** em vez de código imperativo:

```
anonimizar → rag → detetive → acusacao → defesa
→ juiz_rigoroso → juiz_garantista → juiz_equilibrado
→ consistencia → tedh → finalizar
```

Fallback automático para orquestração imperativa se LangGraph não estiver instalado.

```env
ORQUESTRACAO=langgraph   # ou: imperativo
```

---

## ⚔️ Modo Contraditório

Novo em V6. O utilizador intervém como **Advogado de Defesa**:

1. Sistema gera Instrução + Acusação normalmente
2. Utilizador lê a acusação e escreve os seus argumentos
3. Sistema avalia juridicamente o argumento (feedback imediato)
4. `DefesaAgent` incorpora os argumentos do utilizador
5. Pipeline continua com defesa enriquecida

Disponível na interface Streamlit (passo 4) e na CLI (`--contraditorio`).

---

## 🌍 Multi-idioma / TEDH

Adiciona ficheiros `.txt` com jurisprudência ECHR em inglês em `data/tedh/`:

```
data/tedh/
├── ECHR_Article6_FairTrial.txt
├── ECHR_Article8_Privacy.txt
├── ECHR_Article1_Protocol1_Property.txt
└── ...
```

Fonte: https://hudoc.echr.coe.int

O sistema analisa automaticamente o caso português à luz da jurisprudência europeia e avalia o risco de queixa a Estrasburgo.

---

## 🌐 API REST (FastAPI)

```bash
python api_server.py [--host 0.0.0.0] [--port 8000]
# Docs: http://localhost:8000/docs
```

**Endpoints:**

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | `/saude` | Health check |
| `GET` | `/instancias` | Lista tribunais |
| `POST` | `/instrucao` | Gera perguntas de instrução |
| `POST` | `/processar` | Processa caso completo |
| `POST` | `/contraditorio` | Submete argumento de defesa |
| `GET` | `/ata/{id}/pdf` | Download PDF |
| `GET` | `/ata/{id}/txt` | Download TXT |
| `GET` | `/historico` | Lista casos |
| `GET` | `/rag/stats` | Estatísticas RAG |

---

## 🐳 Docker

```bash
# Streamlit apenas
cd docker && docker compose up -d

# Com API REST
docker compose --profile api up -d

# Com Ollama local
docker compose --profile ollama up -d

# Completo
docker compose --profile api --profile ollama up -d

# Instalar modelo Ollama
docker exec tribunal_ollama ollama pull llama3.3:70b
```

---

## 🏛️ Instâncias Judiciais (11)

| Código | Tribunal | Matéria |
|--------|----------|---------|
| `TIC` | T. Instrução Criminal | Penal — Instrução |
| `TCCR` | Tribunal Criminal | Penal — Julgamento |
| `TCIC` | T. Central Instrução Criminal | Grande Criminalidade |
| `TC_CIVEL` | Tribunal Cível | Direito Privado |
| `TFM` | T. Família e Menores | Família |
| `TRAB` | Tribunal do Trabalho | Laboral |
| `TAF` | T. Administrativo e Fiscal | Administrativo |
| `TCOM` | Tribunal de Comércio | Comercial |
| `TR` | Tribunal da Relação | 2ª Instância |
| `STJ` | Supremo Tribunal de Justiça | 3ª Instância |
| `TC` | Tribunal Constitucional | Constitucional |

---

## 🏗️ Arquitectura V6

```
src/
├── agents/         → 9 agentes (+ TEDHAgent + ContraditórioFeedbackAgent)
├── api/            → FastAPI REST backend
├── cache/          → Cache semântico thread-safe
├── contraditorio/  → Modo contraditório (gestor de sessões)
├── export/         → PDF (ReportLab) + leitura PDF (PyMuPDF)
├── historico/      → Histórico persistente com pesquisa
├── pipeline/       → CaseProcessor (LangGraph + fallback) + 11 instâncias
├── prompts/        → Todos os prompts centralizados + TEDH + contraditório
├── rag/            → BM25 + Embeddings + RRF + Reranking + multilíngue
└── utils/          → Config · Brain (OpenRouter+Ollama) · Logger · Anonymizer
```

---

## 🧪 Testes

```bash
# Todos (sem API key necessária)
pytest tests/ -v

# Por módulo
pytest tests/test_rag_v6.py -v          # RAG híbrido + TEDH
pytest tests/test_contraditorio.py -v   # modo contraditório
pytest tests/test_config_v6.py -v       # configuração
pytest tests/test_historico.py -v       # histórico
pytest tests/test_anonymizer.py -v      # RGPD
pytest tests/test_instancias.py -v      # detecção de tribunais
```

---

## 📋 CLI

```bash
# Processar caso
python main.py processar "Fui despedido sem justa causa..."
python main.py processar --instancia TRAB --contraditorio
python main.py processar --modelo google/gemini-2.0-flash-001

# Histórico
python main.py historico --query "furto" --instancia TIC

# RAG
python main.py rag --stats
python main.py rag --pesquisar "despedimento ilícito"
python main.py rag --pesquisar "fair trial" --lingua en

# API
python main.py api --port 8000

# Diagnóstico
python verificar.py --rag --tedh --api
```

---

## 🔒 RGPD & Soberania

- Anonimização automática antes de qualquer chamada externa
- Com `BACKEND=ollama`: **zero dados saem do servidor**
- Hash SHA-256 + watermark em todos os documentos
- Pseudónimos determinísticos por caso (`[PESSOA_4821]`, `[NIF_REMOVIDO]`)

---

## 🚀 Roadmap V7

- [ ] Autenticação multi-tenant para .gov (OAuth2 / SAML)
- [ ] Interface React/Next.js separada (consumindo a FastAPI)
- [ ] Embeddings fine-tuned em corpus jurídico PT
- [ ] Análise de risco de recurso (probabilidade de reversão em 2ª instância)
- [ ] Integração com bases de dados de jurisprudência nacionais
- [ ] Suporte a documentos Word (.docx) como provas
