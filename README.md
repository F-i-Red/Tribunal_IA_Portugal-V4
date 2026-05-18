# 🏛️ Tribunal IA Portugal — V7

Simulador judicial com RAG híbrido PT-específico, LangGraph, FastAPI JWT, Prometheus, modo contraditório, análise TEDH e cadeia de auditoria imutável.

> ⚠️ Aviso Legal: Fins exclusivamente educativos. Não constitui parecer jurídico.
> Este sistema é APOIO COGNITIVO — não DECISÃO SOBERANA.
> Para situações reais: www.oa.pt

---

## 🆕 V7 vs V6

| Funcionalidade | V6 | V7 |
|---|---|---|
| RAG BM25 + Embeddings + PORTULAN | ✅ | ✅ |
| LangGraph + FastAPI JWT + Prometheus | ✅ | ✅ |
| Cadeia de hash imutável (Git jurídico) | ❌ | ✅ |
| Threat model / detecção prompt injection | ❌ | ✅ |
| Voto de vencido formal | ❌ | ✅ |
| Provenance log (rastreabilidade RAG) | ❌ | ✅ |
| Declaração de separação de papéis | ❌ | ✅ |

---

## ⚡ Início Rápido

```bash
pip install -r requirements.txt
cp .env.example .env
# Edita .env com OPENROUTER_API_KEY
streamlit run app.py        # Interface web
python api_server.py        # API REST (docs: /docs)
python verificar.py         # Diagnóstico
```

---

## 🔐 Separação de Papéis

O sistema PODE: resumir factos, identificar legislação, simular argumentos, evidenciar incerteza.

O sistema NÃO PODE: determinar culpa, substituir magistrados, produzir efeitos jurídicos vinculativos.

---

## 🔗 Auditoria — Git Jurídico

Cada caso gera um bloco na cadeia de hash SHA-256 encadeada:

```
Bloco 0: [genesis → hash_caso_001]
Bloco 1: [hash_caso_001 → hash_caso_002]
```

Qualquer alteração retroactiva quebra a cadeia — verificável publicamente.

Componentes:
- CadeiaAuditoria — hash encadeado, append-only
- ProvenanceLog — rastreia quais fragmentos RAG influenciaram cada decisão
- VotoVencido — quando 1 de 3 juízes diverge, formaliza como voto de vencido
- validar_input() — detecta 10 padrões de prompt injection

---

## 🔬 RAG V7 — Modelos PT-específicos

| Modelo | Tamanho | Nota |
|--------|---------|------|
| paraphrase-multilingual-MiniLM-L12-v2 | 118MB | Default — rápido |
| intfloat/multilingual-e5-large-instruct | 560MB | Melhor qualidade |
| neuralmind/bert-base-portuguese-cased | 440MB | PT-nativo |
| PORTULAN/serafim-pt-small-100m-lingua-pt | 400MB | PT-nativo |

```env
RAG_MODO=bm25         # sem downloads (default)
RAG_MODO=hibrido      # com embeddings
```

---

## 🌐 API REST

```bash
python api_server.py --port 8000
# Docs: http://localhost:8000/docs
```

Endpoints principais: /auth/token, /processar, /instrucao, /contraditorio, /historico, /rag/stats, /metrics (Prometheus)

---

## 🧪 Testes

```bash
pytest tests/ -q    # 178 testes, 19 skipped (FastAPI opcional)
pytest tests/test_auditoria.py -v    # cadeia hash, threat model, dissenso
pytest tests/test_rag_v7.py -v       # PORTULAN, Cohere, RRF
pytest tests/test_api_e2e.py -v      # E2E com mock LLM
```

---

## 🏗️ Arquitectura V7

```
src/
├── agents/         → 9 agentes tipados
├── api/            → FastAPI + JWT + rate limiting
├── auditoria/      → Cadeia hash + provenance + threat model  ← NOVO V7
├── cache/          → Cache semântico thread-safe
├── contraditorio/  → Modo contraditório
├── export/         → PDF + leitura PDF
├── historico/      → Histórico persistente
├── observability/  → Prometheus + OpenTelemetry
├── pipeline/       → LangGraph + 11 instâncias + validação input
├── prompts/        → Prompts PT + EN + contraditório
├── rag/            → BM25 + PORTULAN/E5 + RRF + Cohere
└── utils/          → Config · Brain · Logger · Anonymizer
```

---

## 🐳 Docker

```bash
cd docker && docker compose up -d
docker compose --profile api --profile ollama up -d
```

---

## 🚀 Roadmap V8

- Frontend React/Next.js
- Fine-tuning embeddings em corpus jurídico PT
- OAuth2/SAML para .gov
- Integração DGAJ/Citius (requer acesso institucional)
- Constitutional sandbox com dataset jurídico real
