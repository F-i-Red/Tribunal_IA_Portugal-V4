"""
Tribunal IA Portugal V5 — Interface Streamlit
Wizard 5 passos + Histórico + Upload PDF + Exportação PDF
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# ── Página ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tribunal IA Portugal V5",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {
    --azul: #1a3a5c; --azul-claro: #2d6a9f;
    --verde: #1e7e34; --vermelho: #c0392b; --cinza: #f4f6f9;
}
.main-title { font-size: 2.2rem; font-weight: 800; color: var(--azul); }
.sub-title  { font-size: 1rem; color: #555; margin-bottom: 1.2rem; }
.disclaimer {
    background: #fff8e1; border-left: 4px solid #f39c12;
    border-radius: 4px; padding: 0.7rem 1rem; margin-bottom: 1rem; font-size: 0.88rem;
}
.badge-free  { display:inline-block; background:#d4edda; color:#155724;
               border:1px solid #c3e6cb; border-radius:20px; padding:2px 10px; font-size:.8rem; }
.badge-paid  { display:inline-block; background:#cce5ff; color:#004085;
               border:1px solid #b8daff; border-radius:20px; padding:2px 10px; font-size:.8rem; }
.badge-local { display:inline-block; background:#e8d5f5; color:#5a1e8c;
               border:1px solid #d0a8f0; border-radius:20px; padding:2px 10px; font-size:.8rem; }
.step-done    { background:#1e7e34; color:#fff; border-radius:20px; padding:4px 12px;
                font-weight:700; text-align:center; font-size:.82rem; }
.step-active  { background:#1a3a5c; color:#fff; border-radius:20px; padding:4px 12px;
                font-weight:700; text-align:center; font-size:.82rem; }
.step-waiting { background:#e9ecef; color:#666; border-radius:20px; padding:4px 12px;
                text-align:center; font-size:.82rem; }
.sentenca-r { border-left:5px solid #c0392b; background:#fff5f5;
              border-radius:6px; padding:.8rem 1rem; margin-bottom:.8rem; }
.sentenca-g { border-left:5px solid #1e7e34; background:#f5fff7;
              border-radius:6px; padding:.8rem 1rem; margin-bottom:.8rem; }
.sentenca-e { border-left:5px solid #2d6a9f; background:#f0f5ff;
              border-radius:6px; padding:.8rem 1rem; margin-bottom:.8rem; }
.incerteza-alto    { background:#fff0f0; border-left:4px solid #c0392b; padding:.5rem .8rem; border-radius:4px; }
.incerteza-medio   { background:#fffbe6; border-left:4px solid #f39c12; padding:.5rem .8rem; border-radius:4px; }
.incerteza-baixo   { background:#f0fff4; border-left:4px solid #1e7e34; padding:.5rem .8rem; border-radius:4px; }
</style>
""", unsafe_allow_html=True)

# ── Modelos ───────────────────────────────────────────────────────────
MODELOS_FREE = {
    "openrouter/free (router gratuito — recomendado)": "openrouter/free",
    "openrouter/auto (router automático)":             "openrouter/auto",
    "LLaMA 3.3 70B (grátis)":                         "meta-llama/llama-3.3-70b-instruct:free",
    "DeepSeek R1 Raciocínio (grátis)":                "deepseek/deepseek-r1:free",
    "Gemini Flash Experimental (grátis)":              "google/gemini-2.0-flash-exp:free",
    "Qwen 2.5 72B (grátis)":                          "qwen/qwen-2.5-72b-instruct:free",
    "Mistral 7B (grátis, leve)":                      "mistralai/mistral-7b-instruct:free",
}
MODELOS_PAGOS = {
    "Gemini 2.0 Flash ⭐ recomendado ($0.10/1M)":    "google/gemini-2.0-flash-001",
    "Gemini 2.5 Flash ($0.15/1M)":                   "google/gemini-2.5-flash",
    "Claude Haiku 4.5 ($1.00/1M)":                   "anthropic/claude-haiku-4-5",
    "Claude Sonnet 4.6 — qualidade máxima ($3/1M)":  "anthropic/claude-sonnet-4.6",
    "GPT-4.1 Mini ($0.40/1M)":                       "openai/gpt-4.1-mini",
    "DeepSeek Chat V3 ($0.27/1M)":                   "deepseek/deepseek-chat-v3-0324",
}
MODELOS_OLLAMA_SUGERIDOS = [
    "llama3.3:70b", "qwen2.5:72b", "deepseek-r1:32b",
    "mistral-nemo:12b", "llama3.1:8b",
]
NOME_PARA_MODELO = {**MODELOS_FREE, **MODELOS_PAGOS}
MODELO_PARA_NOME = {v: k for k, v in NOME_PARA_MODELO.items()}


def init_state():
    defaults = {
        "step": 1,
        "case_description": "",
        "instancia": None,
        "auto_detect": True,
        "perguntas": None,
        "respostas": {},
        "materiais": "",
        "pdf_docs": [],
        "resultado": None,
        "erro": None,
        "backend": "openrouter",
        "modelo_selecionado": "openrouter/free",
        "ollama_modelo": "llama3.3:70b",
        "ollama_url": "http://localhost:11434",
        "modo_economico": True,
        "tab_activa": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def reset_all():
    preservar = {}
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init_state()
    for k, v in preservar.items():
        st.session_state[k] = v
    from src.utils.brain import reset_brain
    from src.utils.config import reset_config
    reset_brain()
    reset_config()
    st.rerun()


def aplicar_modelo():
    """Aplica o modelo selecionado via variáveis de ambiente e reinicia brain."""
    from src.utils.brain import reset_brain
    from src.utils.config import reset_config
    os.environ["BACKEND"] = st.session_state.backend
    if st.session_state.backend == "ollama":
        os.environ["OLLAMA_MODELO"] = st.session_state.ollama_modelo
        os.environ["OLLAMA_URL"] = st.session_state.ollama_url
    else:
        os.environ["MODELO"] = st.session_state.modelo_selecionado
    reset_config()
    reset_brain()


def is_free() -> bool:
    if st.session_state.backend == "ollama":
        return True
    m = st.session_state.modelo_selecionado
    return (
        m.endswith(":free")
        or "free" in m.lower()
        or m in ("openrouter/auto", "openrouter/free")
    )


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚖️ Tribunal IA Portugal V5")
    st.caption("Simulador judicial — República Portuguesa")
    st.divider()

    st.markdown("#### 🤖 Motor de IA")
    backend = st.radio(
        "Backend:",
        ["☁️ OpenRouter (cloud)", "🖥️ Ollama (local/soberania)"],
        index=0 if st.session_state.backend == "openrouter" else 1,
        label_visibility="collapsed",
    )
    novo_backend = "ollama" if "Ollama" in backend else "openrouter"
    if novo_backend != st.session_state.backend:
        st.session_state.backend = novo_backend
        aplicar_modelo()

    if st.session_state.backend == "openrouter":
        tipo = st.radio("Tipo:", ["🆓 Gratuitos", "💳 Pagos"], horizontal=True,
                        label_visibility="collapsed")
        opcoes = list(MODELOS_FREE.keys()) if "Gratuitos" in tipo else list(MODELOS_PAGOS.keys())
        nome_atual = MODELO_PARA_NOME.get(st.session_state.modelo_selecionado, opcoes[0])
        if nome_atual not in opcoes:
            nome_atual = opcoes[0]
        escolhido = st.selectbox("Modelo:", opcoes, index=opcoes.index(nome_atual))
        novo_modelo = NOME_PARA_MODELO[escolhido]
        if novo_modelo != st.session_state.modelo_selecionado:
            st.session_state.modelo_selecionado = novo_modelo
            aplicar_modelo()

        badge_cls = "badge-free" if is_free() else "badge-paid"
        badge_txt = "🆓 GRÁTIS" if is_free() else "💳 PAGO"
        st.markdown(f'<span class="{badge_cls}">{badge_txt}</span>', unsafe_allow_html=True)
        if is_free():
            st.caption("⏱️ Modelos gratuitos são mais lentos. Cada agente ~30-90s.")

    else:
        # Ollama
        novo_ollama_url = st.text_input("URL Ollama:", value=st.session_state.ollama_url)
        if novo_ollama_url != st.session_state.ollama_url:
            st.session_state.ollama_url = novo_ollama_url
            aplicar_modelo()

        ollama_sugestoes = MODELOS_OLLAMA_SUGERIDOS
        novo_ollama_mod = st.selectbox(
            "Modelo Ollama:",
            ollama_sugestoes + ["outro"],
            index=ollama_sugestoes.index(st.session_state.ollama_modelo)
            if st.session_state.ollama_modelo in ollama_sugestoes else len(ollama_sugestoes),
        )
        if novo_ollama_mod == "outro":
            novo_ollama_mod = st.text_input("Modelo personalizado:", value=st.session_state.ollama_modelo)
        if novo_ollama_mod != st.session_state.ollama_modelo:
            st.session_state.ollama_modelo = novo_ollama_mod
            aplicar_modelo()

        st.markdown('<span class="badge-local">🖥️ LOCAL — Soberania de dados</span>', unsafe_allow_html=True)
        st.caption("Os dados nunca saem do teu servidor.")

    st.divider()
    st.markdown("#### ⚙️ Opções")
    novo_eco = st.toggle("💰 Modo Económico", value=st.session_state.modo_economico,
                         help="Reduz tokens por chamada. Recomendado em gratuitos.")
    if novo_eco != st.session_state.modo_economico:
        st.session_state.modo_economico = novo_eco

    st.divider()

    # Histórico rápido
    st.markdown("#### 📋 Histórico")
    try:
        from src.utils.config import get_config
        from src.historico import get_historico
        cfg = get_config()
        if cfg.historico_enabled:
            hist = get_historico()
            stats = hist.estatisticas()
            st.caption(f"**{stats['total']}** casos processados")
            recentes = hist.pesquisar(limite=3)
            for r in recentes:
                with st.expander(f"📄 {r.id}", expanded=False):
                    st.caption(f"**{r.instancia_nome}**")
                    st.caption(r.resumo[:100] + "...")
                    st.caption(f"Incerteza: **{r.grau_incerteza}**")
    except Exception:
        st.caption("—")

    st.divider()
    try:
        from src.cache import get_cache
        from src.cache import limpar_cache
        stats_c = get_cache().estatisticas()
        st.markdown("#### 📊 Cache")
        st.caption(f"**{stats_c['entradas']}** entradas em cache")
        if st.button("🗑️ Limpar cache", use_container_width=True):
            limpar_cache(0)
            st.success("Cache limpo")
    except Exception:
        pass

    st.divider()
    st.markdown(
        '<div style="font-size:.72rem;color:#aaa;text-align:center;">'
        '⚠️ Fins educativos apenas<br>'
        '<a href="https://www.oa.pt" target="_blank">Ordem dos Advogados de Portugal</a>'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Header ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏛️ Tribunal IA Portugal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Simulador judicial · Direito Português 🇵🇹 · V5</div>', unsafe_allow_html=True)
st.markdown("""
<div class="disclaimer">
<strong>⚠️ Aviso Legal:</strong> Ferramenta de simulação educativa.
Não constitui parecer jurídico nem decisão judicial.
Para situações reais: <a href="https://www.oa.pt" target="_blank">Ordem dos Advogados de Portugal</a>.
</div>
""", unsafe_allow_html=True)

# Verificar config
try:
    os.environ.setdefault("BACKEND", st.session_state.backend)
    os.environ.setdefault("MODELO", st.session_state.modelo_selecionado)
    os.environ.setdefault("OLLAMA_MODELO", st.session_state.ollama_modelo)
    os.environ.setdefault("OLLAMA_URL", st.session_state.ollama_url)
    from src.utils.config import get_config
    cfg = get_config()
except Exception as e:
    st.error(f"❌ Configuração: {e}")
    if "openrouter" in str(e).lower():
        st.code("OPENROUTER_API_KEY=a_tua_chave\nMODELO=openrouter/auto", language="bash")
        st.info("Obtém chave gratuita em https://openrouter.ai/keys  |  Ou usa Ollama local (sem chave)")
    st.stop()

# Progress steps
step = st.session_state.step
labels = ["1 · Caso", "2 · Documentos", "3 · Instrução", "4 · Processo", "5 · Resultado"]
cols = st.columns(5)
for i, label in enumerate(labels):
    with cols[i]:
        if i + 1 < step:
            st.markdown(f'<div class="step-done">✅ {label}</div>', unsafe_allow_html=True)
        elif i + 1 == step:
            st.markdown(f'<div class="step-active">▶ {label}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="step-waiting">{label}</div>', unsafe_allow_html=True)
st.divider()


# ════════════════════════════════════════════════════════════════════════
# PASSO 1 — CASO
# ════════════════════════════════════════════════════════════════════════
if step == 1:
    st.markdown("### 📝 Descreve o caso")
    st.info("Usa linguagem comum. Quanto mais detalhe, melhor o resultado. Podes também carregar documentos no passo seguinte.")

    case_input = st.text_area(
        "Caso jurídico:",
        value=st.session_state.case_description,
        height=220,
        placeholder=(
            "Ex: Fui despedido da empresa XYZ sem justa causa após 8 anos de trabalho. "
            "Recebi uma carta a dizer que era por motivos económicos, mas contrataram outra pessoa "
            "passado 2 semanas. Tenho o contrato de trabalho, os recibos de vencimento e emails..."
        ),
        label_visibility="collapsed",
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        st.session_state.auto_detect = st.checkbox("🔎 Detectar tribunal automaticamente",
                                                    value=st.session_state.auto_detect)
    with c2:
        if not st.session_state.auto_detect:
            from src.pipeline.instancias import INSTANCIAS
            opcoes_inst = {f"{k} — {v.nome}": k for k, v in INSTANCIAS.items()}
            escolha = st.selectbox("Tribunal:", list(opcoes_inst.keys()))
            st.session_state.instancia = opcoes_inst[escolha]

    if st.button("▶ Avançar", type="primary", disabled=not case_input.strip()):
        st.session_state.case_description = case_input
        if st.session_state.auto_detect:
            from src.pipeline.instancias import detectar_instancia_por_keywords
            st.session_state.instancia = detectar_instancia_por_keywords(case_input)
        st.session_state.perguntas = None
        st.session_state.respostas = {}
        st.session_state.pdf_docs = []
        st.session_state.step = 2
        st.rerun()


# ════════════════════════════════════════════════════════════════════════
# PASSO 2 — DOCUMENTOS (PDF Upload)
# ════════════════════════════════════════════════════════════════════════
elif step == 2:
    from src.pipeline.instancias import INSTANCIAS
    inst = INSTANCIAS[st.session_state.instancia]
    st.markdown(f"### 📎 Documentos e Provas — {inst.nome}")
    st.caption("Carrega documentos PDF relevantes (contratos, relatórios, certidões, etc.) "
               "para enriquecer a análise. Esta etapa é opcional.")

    uploaded = st.file_uploader(
        "Documentos PDF (opcional):",
        type=["pdf"],
        accept_multiple_files=True,
        help="Máximo 5 ficheiros. O conteúdo é processado localmente e anonimizado.",
    )

    docs_processados = []
    if uploaded:
        from src.export import extrair_texto_pdf
        from src.utils.brain import get_brain
        from src.agents import PDFExtractorAgent
        from src.utils.logger import get_logger

        with st.spinner("A processar documentos PDF..."):
            for f in uploaded[:5]:  # máximo 5
                bytes_pdf = f.read()
                texto, tipo = extrair_texto_pdf(bytes_pdf)
                if texto and not texto.startswith("PyMuPDF"):
                    try:
                        ag = PDFExtractorAgent(get_brain(), get_logger())
                        resumo = ag.executar(texto, tipo)
                        docs_processados.append(resumo)
                        st.success(f"✅ {f.name} ({tipo}) — processado")
                    except Exception as e:
                        st.warning(f"⚠️ {f.name}: não foi possível analisar — {str(e)[:100]}")
                        docs_processados.append(f"[Documento: {f.name}]\n{texto[:1000]}")
                elif texto.startswith("PyMuPDF"):
                    st.warning("PyMuPDF não instalado. Instala com: `pip install PyMuPDF`")
                    docs_processados.append(f"[Ficheiro: {f.name} — texto não extraído]")

    st.session_state.pdf_docs = docs_processados

    if docs_processados:
        st.info(f"📄 {len(docs_processados)} documento(s) prontos para integrar na análise.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅ Voltar"):
            st.session_state.step = 1
            st.rerun()
    with c2:
        if st.button("▶ Avançar para Instrução", type="primary"):
            st.session_state.step = 3
            st.rerun()


# ════════════════════════════════════════════════════════════════════════
# PASSO 3 — INSTRUÇÃO
# ════════════════════════════════════════════════════════════════════════
elif step == 3:
    from src.pipeline.instancias import INSTANCIAS
    inst = INSTANCIAS[st.session_state.instancia]
    st.markdown(f"### 🔍 Instrução — {inst.nome}")
    st.caption(f"Matéria: {inst.materia} | Diploma: {inst.diploma_principal}")

    col_info, col_saltar = st.columns([3, 1])
    with col_info:
        st.info(
            "O sistema vai gerar perguntas **específicas a este caso concreto**. "
            "As respostas enriquecem a análise de todas as peças processuais."
        )
    with col_saltar:
        if st.button("⏭ Saltar instrução", use_container_width=True):
            st.session_state.perguntas = {"perguntas": [], "introducao": ""}
            st.session_state.step = 4
            st.session_state.resultado = None
            st.rerun()

    # Gerar perguntas com timeout
    if st.session_state.perguntas is None:
        TIMEOUT = 90
        _res: dict = {"perguntas": None, "erro": None}

        # Capturar ANTES do thread — st.session_state não é acessível dentro de threads
        _case_desc   = str(st.session_state.get("case_description", ""))
        _instancia   = str(st.session_state.get("instancia", "TIC"))

        def _gerar():
            try:
                from src.pipeline.case_processor import CaseProcessor
                proc = CaseProcessor()
                _res["perguntas"] = proc.gerar_perguntas_instrucao(
                    _case_desc, _instancia
                )
            except Exception as ex:
                _res["erro"] = str(ex)

        t = threading.Thread(target=_gerar, daemon=True)
        t.start()
        with st.spinner("A analisar o caso e a gerar perguntas de instrução..."):
            t.join(timeout=TIMEOUT)

        if t.is_alive():
            st.session_state.perguntas = {"perguntas": [], "introducao": "", "_timeout": True}
        elif _res["erro"]:
            st.session_state.perguntas = {"perguntas": [], "introducao": "", "_erro": _res["erro"]}
        else:
            st.session_state.perguntas = _res["perguntas"]
        st.rerun()

    perguntas = st.session_state.perguntas

    # Tratar erros/timeout
    if perguntas.get("_timeout") or perguntas.get("_erro"):
        if perguntas.get("_timeout"):
            st.error(f"⏱️ O modelo não respondeu em {TIMEOUT if '_timeout' in perguntas else 90}s. "
                     "Tenta mudar de modelo ou avança sem instrução.")
        else:
            st.error(f"❌ {perguntas.get('_erro','Erro desconhecido')[:200]}")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🔄 Tentar de novo"):
                st.session_state.perguntas = None
                st.rerun()
        with c2:
            if st.button("⬅ Voltar"):
                st.session_state.step = 2
                st.rerun()
        with c3:
            if st.button("▶ Avançar sem instrução", type="primary"):
                st.session_state.perguntas = {"perguntas": [], "introducao": ""}
                st.session_state.step = 4
                st.session_state.resultado = None
                st.rerun()
        st.stop()

    if perguntas.get("introducao"):
        st.markdown(f"*{perguntas['introducao']}*")

    n_p = len(perguntas.get("perguntas", []))
    if n_p:
        st.caption(f"✅ {n_p} pergunta(s) geradas especificamente para este caso.")

    with st.form("instrucao_form"):
        for p in perguntas.get("perguntas", []):
            badge = {"critica": "🔴", "relevante": "🟡", "complementar": "🟢"}.get(
                p.get("importancia", ""), "⚪"
            )
            st.markdown(f"**{badge} [{p.get('categoria','?')}]** {p.get('texto','')}")
            if p.get("razao"):
                st.caption(f"_Relevância: {p['razao']}_")
            resp = st.text_area(
                "Resposta:", key=f"resp_{p['id']}", height=65,
                value=st.session_state.respostas.get(p["id"], ""),
                placeholder="Responde ou deixa em branco para ignorar...",
                label_visibility="collapsed",
            )
            st.session_state.respostas[p["id"]] = resp
            st.markdown("---")

        st.markdown("##### 📎 Informações adicionais")
        materiais = st.text_area(
            "Informações adicionais (opcional):",
            value=st.session_state.materiais, height=65,
            placeholder="Ex: Tenho emails, fotografias, testemunhas disponíveis...",
            label_visibility="collapsed",
        )
        st.session_state.materiais = materiais

        c_vol, c_go = st.columns([1, 2])
        with c_vol:
            voltar = st.form_submit_button("⬅ Voltar")
        with c_go:
            go = st.form_submit_button("▶ Iniciar Processo Judicial", type="primary")

    if voltar:
        st.session_state.step = 2
        st.rerun()
    if go:
        st.session_state.step = 4
        st.session_state.resultado = None
        st.session_state.erro = None
        st.rerun()


# ════════════════════════════════════════════════════════════════════════
# PASSO 4 — PROCESSAR
# ════════════════════════════════════════════════════════════════════════
elif step == 4:
    st.markdown("### ⚖️ Processo Judicial em curso...")

    if st.session_state.erro:
        st.error(f"❌ {st.session_state.erro}")
        if is_free():
            st.info("💡 Com modelos gratuitos pode haver rate limits. Aguarda 1-2 min e tenta novamente.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 Tentar novamente"):
                st.session_state.erro = None
                st.rerun()
        with c2:
            if st.button("⬅ Voltar"):
                st.session_state.step = 3
                st.session_state.erro = None
                st.rerun()
        st.stop()

    if st.session_state.resultado is not None:
        st.session_state.step = 5
        st.rerun()

    # Construir dados de instrução
    dados_instrucao = None
    respostas_v = {
        k: {
            "pergunta": next(
                (p["texto"] for p in (st.session_state.perguntas or {}).get("perguntas", [])
                 if p["id"] == k), ""
            ),
            "categoria": next(
                (p["categoria"] for p in (st.session_state.perguntas or {}).get("perguntas", [])
                 if p["id"] == k), ""
            ),
            "resposta": v,
        }
        for k, v in st.session_state.respostas.items() if v.strip()
    }
    if respostas_v:
        dados_instrucao = {
            "respostas": respostas_v,
            "materiais": [{"descricao": st.session_state.materiais}]
            if st.session_state.materiais.strip() else [],
        }

    modelo_info = (
        f"Ollama ({st.session_state.ollama_modelo})"
        if st.session_state.backend == "ollama"
        else st.session_state.modelo_selecionado
    )
    tempo_est = "2-8 min" if is_free() else "30-90s"
    st.info(f"🤖 Modelo: **{modelo_info}** | Tempo estimado: **{tempo_est}**")

    agentes = [
        "🔍 Instrução factual", "⚔️ Acusação / MP", "🛡️ Defesa",
        "⚖️ Juiz Rigoroso", "⚖️ Juiz Garantista", "⚖️ Juiz Equilibrado",
        "📊 Consistência e Incerteza",
    ]
    progress_placeholder = st.empty()
    progress_placeholder.progress(0, text="A iniciar o processo...")

    _res: dict = {"resultado": None, "erro": None}

    # Capturar ANTES do thread — st.session_state não é acessível dentro de threads
    _case_desc_p4  = str(st.session_state.get("case_description", ""))
    _instancia_p4  = str(st.session_state.get("instancia", "TIC"))
    _pdf_docs_p4   = list(st.session_state.get("pdf_docs", []) or [])
    _backend_p4    = str(st.session_state.get("backend", "openrouter"))
    _modelo_p4     = str(st.session_state.get("modelo_selecionado", "openrouter/free"))
    _ollama_mod_p4 = str(st.session_state.get("ollama_modelo", "llama3.3:70b"))
    _ollama_url_p4 = str(st.session_state.get("ollama_url", "http://localhost:11434"))

    def _processar():
        import os as _os
        # Aplicar modelo dentro do thread usando variáveis locais (não session_state)
        _os.environ["BACKEND"] = _backend_p4
        if _backend_p4 == "ollama":
            _os.environ["OLLAMA_MODELO"] = _ollama_mod_p4
            _os.environ["OLLAMA_URL"] = _ollama_url_p4
        else:
            _os.environ["MODELO"] = _modelo_p4
        try:
            from src.utils.config import reset_config
            from src.utils.brain import reset_brain
            reset_config()
            reset_brain()
            from src.pipeline.case_processor import CaseProcessor
            proc = CaseProcessor()
            _res["resultado"] = proc.process(
                case_description=_case_desc_p4,
                instancia_codigo=_instancia_p4,
                dados_instrucao=dados_instrucao,
                gerar_pdf=True,
                pdf_docs_extraidos=_pdf_docs_p4 or None,
            )
        except Exception as ex:
            _res["erro"] = str(ex)

    t = threading.Thread(target=_processar, daemon=True)
    t.start()

    import time
    for i, agente in enumerate(agentes):
        if not t.is_alive():
            break
        frac = (i + 1) / len(agentes)
        progress_placeholder.progress(frac, text=f"A processar: {agente}...")
        t.join(timeout=8)

    t.join(timeout=600)  # timeout máximo total 10 min

    if _res["erro"]:
        st.session_state.erro = _res["erro"]
    elif _res["resultado"]:
        st.session_state.resultado = _res["resultado"]
        st.session_state.step = 5
    else:
        st.session_state.erro = "Processo não concluído — timeout máximo atingido."

    progress_placeholder.empty()
    st.rerun()


# ════════════════════════════════════════════════════════════════════════
# PASSO 5 — RESULTADO
# ════════════════════════════════════════════════════════════════════════
elif step == 5:
    st.markdown("### 📄 Resultado do Processo")

    result = st.session_state.resultado
    if result is None:
        st.warning("Sem resultado disponível.")
        if st.button("⬅ Recomeçar"):
            reset_all()
        st.stop()

    # Métricas de topo
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Tribunal", result.instancia_codigo)
    with m2:
        custo = "Gratuito 🆓" if result.custo_total_usd == 0 else f"${result.custo_total_usd:.4f}"
        st.metric("Custo", custo)
    with m3:
        st.metric("Entidades RGPD", len(result.entities_found))
    with m4:
        grau = result.grau_incerteza
        st.metric("Grau de Incerteza", grau)
    with m5:
        if st.button("🔄 Novo caso", use_container_width=True):
            reset_all()

    # Badge de incerteza
    grau_map = {
        "Baixo": ("incerteza-baixo", "🟢"),
        "Médio": ("incerteza-medio", "🟡"),
        "Alto": ("incerteza-alto", "🔴"),
        "Muito Alto": ("incerteza-alto", "🔴🔴"),
    }
    cls, emoji = grau_map.get(result.grau_incerteza, ("", ""))
    if cls:
        st.markdown(
            f'<div class="{cls}"><strong>{emoji} Grau de incerteza jurídica: {result.grau_incerteza}</strong>'
            " — ver Relatório de Consistência para detalhes.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")

    tabs = st.tabs([
        "📋 Peças Processuais",
        "⚖️ Sentenças (3 Perfis)",
        "📊 Consistência & Incerteza",
        "📄 Ata Completa",
        "🕐 Histórico",
    ])

    # ── Tab 1: Peças ──────────────────────────────────────────────────
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### 🔍 Instrução Factual")
            st.markdown(result.detetive_report or "*Não disponível*")
        with c2:
            st.markdown("#### ⚔️ Acusação / MP")
            st.markdown(result.acusacao or "*Não disponível*")
        with c3:
            st.markdown("#### 🛡️ Defesa")
            st.markdown(result.defesa or "*Não disponível*")

        if result.validacao_citacoes:
            with st.expander("🔎 Validação de Citações Jurídicas"):
                st.markdown(result.validacao_citacoes)

    # ── Tab 2: Sentenças ──────────────────────────────────────────────
    with tabs[1]:
        st.info(
            "Três decisões sobre os mesmos factos, por juízes com perfis diferentes. "
            "A divergência revela a discricionariedade legítima do sistema judicial."
        )
        for titulo, texto, cls in [
            ("🔴 Perfil Rigoroso — prevenção geral", result.sentenca_rigorosa, "sentenca-r"),
            ("🟢 Perfil Garantista — in dubio pro reo", result.sentenca_garantista, "sentenca-g"),
            ("🔵 Perfil Equilibrado — proporcionalidade", result.sentenca_equilibrada, "sentenca-e"),
        ]:
            with st.expander(titulo, expanded=False):
                st.markdown(
                    f'<div class="{cls}">{(texto or "*Não disponível*").replace(chr(10),"<br>")}</div>',
                    unsafe_allow_html=True,
                )

    # ── Tab 3: Consistência ───────────────────────────────────────────
    with tabs[2]:
        st.markdown("#### 📊 Relatório de Consistência e Incerteza")
        if result.relatorio_consistencia:
            st.markdown(result.relatorio_consistencia)
        else:
            st.info("Relatório de consistência não gerado.")

        # Comparação rápida de dispositivos
        st.markdown("---")
        st.markdown("#### Comparação dos Dispositivos")
        import re

        def disp(txt: str) -> str:
            if not txt:
                return "*Não disponível*"
            m = re.search(r"(?:CONDENA|ABSOLVE|JULGA)[^.]*\.", txt, re.IGNORECASE)
            return m.group(0).strip()[:250] if m else txt[:200] + "..."

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**🔴 Rigoroso**")
            st.error(disp(result.sentenca_rigorosa))
        with c2:
            st.markdown("**🟢 Garantista**")
            st.success(disp(result.sentenca_garantista))
        with c3:
            st.markdown("**🔵 Equilibrado**")
            st.info(disp(result.sentenca_equilibrada))

    # ── Tab 4: Ata Completa ───────────────────────────────────────────
    with tabs[3]:
        st.markdown("#### Ata Completa do Processo")
        st.text_area("", value=result.ata_final or "", height=450,
                     disabled=True, label_visibility="collapsed")

        col_dl1, col_dl2, col_dl3 = st.columns(3)
        with col_dl1:
            st.download_button(
                "⬇️ Download TXT",
                data=(result.ata_final or "").encode("utf-8"),
                file_name=f"{result.case_id}.txt",
                mime="text/plain",
            )
        with col_dl2:
            if result.pdf_bytes:
                st.download_button(
                    "⬇️ Download PDF",
                    data=result.pdf_bytes,
                    file_name=f"{result.case_id}.pdf",
                    mime="application/pdf",
                )
            else:
                st.caption("PDF: instala `reportlab` para activar")
        with col_dl3:
            st.caption(f"🔑 `{result.case_id}`")
            st.caption(f"Hash: `{result.doc_hash}`")

    # ── Tab 5: Histórico ──────────────────────────────────────────────
    with tabs[4]:
        st.markdown("#### 🕐 Histórico de Casos")
        try:
            from src.historico import get_historico
            hist = get_historico()
            stats = hist.estatisticas()
            st.caption(f"**{stats['total']}** casos no histórico")

            col_q, col_f = st.columns([3, 1])
            with col_q:
                query_hist = st.text_input("🔍 Pesquisar:", placeholder="Texto, tribunal, decisão...")
            with col_f:
                from src.pipeline.instancias import INSTANCIAS
                filtro_inst = st.selectbox(
                    "Tribunal:", ["Todos"] + list(INSTANCIAS.keys()),
                    label_visibility="collapsed",
                )

            registos = hist.pesquisar(
                query=query_hist,
                instancia=None if filtro_inst == "Todos" else filtro_inst,
                limite=20,
            )

            if registos:
                for r in registos:
                    with st.expander(f"📄 {r.id} — {r.instancia_nome} — {r.grau_incerteza}", expanded=False):
                        st.caption(f"**Data:** {r.timestamp[:19].replace('T',' ')}")
                        st.caption(f"**Modelo:** {r.modelo}")
                        st.markdown(f"**Caso:** {r.resumo}")
                        st.markdown(f"**Decisão:** {r.dispositivo}")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.caption(f"Incerteza: **{r.grau_incerteza}** | Custo: **${r.custo_usd:.4f}**")
                        with col_b:
                            if r.ata_path and Path(r.ata_path).exists():
                                ata = Path(r.ata_path).read_text(encoding="utf-8")
                                st.download_button(
                                    "⬇️ Ata",
                                    data=ata,
                                    file_name=Path(r.ata_path).name,
                                    key=f"dl_{r.id}",
                                )
            else:
                st.info("Sem resultados para esta pesquisa.")

            if st.button("🗑️ Limpar histórico", type="secondary"):
                hist.limpar()
                st.success("Histórico limpo.")
                st.rerun()
        except Exception as e:
            st.warning(f"Histórico não disponível: {e}")
