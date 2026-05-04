# 🏛️ Tribunal IA Portugal — V4

Simulador judicial de alta fidelidade para o Direito Português.
Concebido para fins educativos, académicos e de prova de conceito.

> ⚠️ **Aviso Legal:** Não constitui parecer jurídico nem decisão judicial.
> Para situações reais: [Ordem dos Advogados de Portugal](https://www.oa.pt)

---
**O modelo de inteligência artificial grátis comete erros e não consegue o que um modelo pago consegue fazer. Aqui usei um modelo grátis. É apenas para prova de conceito.**
---

## 🆕 V4 — Tudo o que foi implementado

| Funcionalidade | V3 | V4 |
|---|---|---|
| Pydantic Settings v2 | ✅ | ✅ |
| Agentes com herança | ✅ | ✅ |
| Prompts centralizados | ✅ | ✅ |
| Thread-safe Brain + Cache | ✅ | ✅ |
| **Ollama (soberania de dados)** | ❌ | ✅ |
| **RAG com metadata filtering** | ❌ | ✅ |
| **Relatório de consistência** | ❌ | ✅ |
| **Grau de incerteza jurídica** | ❌ | ✅ |
| **Upload de PDF como prova** | ❌ | ✅ |
| **Exportação de ata em PDF** | ❌ | ✅ |
| **Histórico de casos** | ❌ | ✅ |
| **Docker + docker-compose** | ❌ | ✅ |
| **CLI com typer + rich** | ❌ | ✅ |
| **Logger structlog** | ❌ | ✅ |
| Passo 2 - documentos no wizard | ❌ | ✅ |
| Wizard 5 passos | ❌ | ✅ |
| 7 suites de testes | ❌ | ✅ |

---

## ⚡ Início Rápido

### 1. Instalar
```bash
git clone https://github.com/F-i-Red/Tribunal_IA_Portugal-V4
cd Tribunal_IA_Portugal-V4
pip install -r requirements.txt
```

### 2. Configurar
```bash
cp .env.example .env
# Edita .env
```

### 3. Iniciar
```bash
# Interface web
streamlit run app.py
# ou
./iniciar_interface.sh

# CLI
python main.py processar

# Diagnóstico
python verificar.py
```

---

## 🤖 Backends Suportados

### OpenRouter (cloud)
```env
BACKEND=openrouter
OPENROUTER_API_KEY=sk-or-...
MODELO=openrouter/auto        # gratuito — testes
MODELO=google/gemini-2.0-flash-001  # pago — produção
```

Chave gratuita: https://openrouter.ai/keys

### Ollama (local — soberania de dados)
```env
BACKEND=ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODELO=llama3.3:70b
```

```bash
# Instalar Ollama: https://ollama.ai
ollama serve
ollama pull llama3.3:70b
```

**Recomendado para .gov** — os dados nunca saem do servidor.

---

## 🐳 Docker

### OpenRouter
```bash
cd docker
docker compose up -d
```

### Com Ollama local
```bash
cd docker
docker compose --profile ollama up -d
# Instalar modelo no container:
docker exec tribunal_ollama ollama pull llama3.3:70b
```

---

## 📚 RAG — Base de Conhecimento

```
data/
├── leis/           → Códigos e leis (Codigo_Penal.txt, etc.)
├── jurisprudencia/ → Acórdãos e jurisprudência
└── precedentes/    → Precedentes relevantes
```

Modos:
- `bm25` — BM25 puro (sem dependências extra, default)
- `hibrido` — BM25 + sentence-transformers (requer `pip install sentence-transformers`)
- `api` — BM25 + embeddings via OpenRouter

Metadata filtering automático por diploma e instância judicial.

```bash
# Gerir base
python gerir_base.py --stats
python gerir_base.py --reindexar
python gerir_base.py --pesquisar "furto qualificado" --instancia TIC
python gerir_base.py --historico
```

---

## 🏛️ Instâncias Judiciais

| Código | Tribunal | Matéria |
|--------|----------|---------|
| `TIC` | T. Instrução Criminal | Penal — Instrução |
| `TCCR` | Tribunal Criminal | Penal — Julgamento |
| `TCIC` | T. Central Instrução Criminal | Grande Criminalidade |
| `TC_CIVEL` | Tribunal Cível | Direito Privado |
| `TFM` | T. Família e Menores | Família / Menores |
| `TRAB` | Tribunal do Trabalho | Laboral |
| `TAF` | T. Administrativo e Fiscal | Administrativo |
| `TCOM` | Tribunal de Comércio | Comercial / Insolvência |
| `TR` | Tribunal da Relação | 2ª Instância |
| `STJ` | Supremo Tribunal de Justiça | 3ª Instância |
| `TC` | Tribunal Constitucional | Constitucional |

---

## 🔄 Fluxo V4

```
Caso (texto + PDFs opcionais)
        ↓
Anonimização RGPD
        ↓
RAG (BM25 + metadata filtering por instância e diploma)
        ↓
Instrução (perguntas específicas ao caso via LLM)
        ↓
┌──────────────────────────┐
│ Agente Detetive          │
│ Agente Acusação/MP       │  → sequencial (dependências)
│ Agente Defesa            │
└──────────────────────────┘
        ↓
┌─────────────────────────────┐
│ Juiz Rigoroso               │
│ Juiz Garantista             │  → paralelo (pagos) / sequencial (free/Ollama)
│ Juiz Equilibrado            │
└─────────────────────────────┘
        ↓
Agente Consistência + Incerteza
        ↓
Validação de citações (RAG)
        ↓
Ata TXT + PDF (ReportLab)
        ↓
Histórico de casos
```

---

## 📊 Relatório de Consistência e Incerteza

Novo em V4. Para cada caso, o sistema gera:

- **Convergências** — factos em que as 3 sentenças concordam (alta certeza)
- **Divergências** — onde as sentenças diferem (revela discricionariedade)
- **Pontos factuais mais frágeis** — onde a prova é questionada
- **Grau de incerteza global** — Baixo / Médio / Alto / Muito Alto
- **Recomendação ao cidadão** em linguagem acessível

---

## 🔒 RGPD & Soberania de Dados

- Anonimização automática antes de qualquer chamada API externa
- Entidades: nomes, emails, telefones, NIFs, CCs, NISSs, moradas, IBANs, processos
- Pseudónimos determinísticos por hash (`[PESSOA_4821]`, `[NIF_REMOVIDO]`)
- Com Ollama: **zero dados saem do servidor** — soberania total
- Hash + watermark em todos os documentos

---

## 🧪 Testes

```bash
# Todos (não requer API key)
pytest tests/ -v

# Por categoria
pytest tests/test_anonymizer.py -v     # anonimização
pytest tests/test_rag_v5.py -v         # RAG + metadata filtering
pytest tests/test_historico.py -v      # histórico de casos
pytest tests/test_config_v5.py -v      # configuração
pytest tests/test_instancias.py -v     # detecção de instâncias
pytest tests/test_export.py -v         # exportação PDF
```

---

## 🏗️ Arquitectura V4

```
src/
├── agents/         → BaseAgent + 6 agentes especializados
├── cache/          → Cache semântico thread-safe
├── export/         → PDF (ReportLab) + leitura de PDF (PyMuPDF)
├── historico/      → Histórico persistente com pesquisa
├── pipeline/       → CaseProcessor + 11 instâncias judiciais
├── prompts/        → Todos os prompts centralizados
├── rag/            → BM25 + metadata filtering + validador
└── utils/          → Config · Brain · Logger · Anonymizer
```

---

## 🚀 Roadmap V5

- [ ] Embeddings híbridos com sentence-transformers PT
- [ ] Interface multi-idioma (EN para comparação com ECHR)
- [ ] Modo contraditório interactivo
- [ ] API REST (FastAPI) para integração
- [ ] Autenticação e multi-tenant para .gov
- [ ] Observability (OpenTelemetry + Prometheus)
