"""
Tribunal IA Portugal V6 — Interface Streamlit
Wizard 6 passos: Caso → Documentos → Instrução → Contraditório → Processo → Resultado
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# ── Singleton CaseProcessor — instanciado uma vez, partilhado entre passos ──
_processor_singleton = None
_processor_lock = threading.Lock()

def get_processor():
    global _processor_singleton
    with _processor_lock:
        if _processor_singleton is None:
            from src.pipeline.case_processor import CaseProcessor
            _processor_singleton = CaseProcessor()
    return _processor_singleton

def reset_processor():
    global _processor_singleton
    with _processor_lock:
        _processor_singleton = None

st.set_page_config(
    page_title="Tribunal IA Portugal V6",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { --azul:#1a3a5c; --azul2:#2d6a9f; --verde:#1e7e34; --verm:#c0392b; }
.main-title { font-size:2.2rem; font-weight:800; color:var(--azul); }
.sub-title  { font-size:1rem; color:#555; margin-bottom:1.2rem; }
.disclaimer { background:#fff8e1; border-left:4px solid #f39c12;
              border-radius:4px; padding:.7rem 1rem; margin-bottom:1rem; font-size:.88rem; }
.badge-free  { display:inline-block; background:#d4edda; color:#155724;
               border:1px solid #c3e6cb; border-radius:20px; padding:2px 10px; font-size:.8rem; }
.badge-paid  { display:inline-block; background:#cce5ff; color:#004085;
               border:1px solid #b8daff; border-radius:20px; padding:2px 10px; font-size:.8rem; }
.badge-local { display:inline-block; background:#e8d5f5; color:#5a1e8c;
               border:1px solid #d0a8f0; border-radius:20px; padding:2px 10px; font-size:.8rem; }
.badge-lg    { display:inline-block; background:#e3f2fd; color:#0d47a1;
               border:1px solid #90caf9; border-radius:20px; padding:2px 10px; font-size:.8rem; }
.step-done    { background:#1e7e34; color:#fff; border-radius:20px; padding:3px 10px;
                font-weight:700; text-align:center; font-size:.78rem; }
.step-active  { background:#1a3a5c; color:#fff; border-radius:20px; padding:3px 10px;
                font-weight:700; text-align:center; font-size:.78rem; }
.step-waiting { background:#e9ecef; color:#666; border-radius:20px; padding:3px 10px;
                text-align:center; font-size:.78rem; }
.sentenca-r { border-left:5px solid #c0392b; background:#fff5f5; border-radius:6px;
              padding:.8rem 1rem; margin-bottom:.8rem; }
.sentenca-g { border-left:5px solid #1e7e34; background:#f5fff7; border-radius:6px;
              padding:.8rem 1rem; margin-bottom:.8rem; }
.sentenca-e { border-left:5px solid #2d6a9f; background:#f0f5ff; border-radius:6px;
              padding:.8rem 1rem; margin-bottom:.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Modelos ────────────────────────────────────────────────────────────
MODELOS_FREE = {
    "openrouter/free (router gratuito ⭐)":           "openrouter/free",
    "openrouter/auto":                                 "openrouter/auto",
    "LLaMA 3.3 70B (grátis)":                        "meta-llama/llama-3.3-70b-instruct:free",
    "DeepSeek R1 Raciocínio (grátis)":               "deepseek/deepseek-r1:free",
    "Gemini Flash Exp (grátis)":                      "google/gemini-2.0-flash-exp:free",
    "Qwen 2.5 72B (grátis)":                         "qwen/qwen-2.5-72b-instruct:free",
}
MODELOS_PAGOS = {
    "Gemini 2.0 Flash ⭐ ($0.10/1M)":               "google/gemini-2.0-flash-001",
    "Gemini 2.5 Flash ($0.15/1M)":                   "google/gemini-2.5-flash",
    "Claude Haiku 4.5 ($1.00/1M)":                   "anthropic/claude-haiku-4-5",
    "Claude Sonnet 4.6 — máxima qualidade ($3/1M)":  "anthropic/claude-sonnet-4.6",
    "GPT-4.1 Mini ($0.40/1M)":                       "openai/gpt-4.1-mini",
    "DeepSeek Chat V3 ($0.27/1M)":                   "deepseek/deepseek-chat-v3-0324",
}
MODELOS_OLLAMA = ["llama3.3:70b","qwen2.5:72b","deepseek-r1:32b","mistral-nemo:12b","llama3.1:8b"]
NOME_PARA_MODELO = {**MODELOS_FREE, **MODELOS_PAGOS}
MODELO_PARA_NOME = {v: k for k, v in NOME_PARA_MODELO.items()}


def init_state():
    d = {
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
        "modo_contraditorio": False,
        "sessao_contraditorio": None,
        "argumento_defesa": "",
        "feedback_contraditorio": "",
        "_contr_detetive": None,
        "_contr_acusacao": None,
        "_contr_defesa": None,
    }
    for k, v in d.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def reset_all():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init_state()
    from src.utils.brain import reset_brain
    from src.utils.config import reset_config
    reset_brain(); reset_config()
    st.rerun()


def aplicar_modelo():
    import os
    from src.utils.brain import reset_brain
    from src.utils.config import reset_config
    os.environ["BACKEND"] = st.session_state.backend
    if st.session_state.backend == "ollama":
        os.environ["OLLAMA_MODELO"] = st.session_state.ollama_modelo
        os.environ["OLLAMA_URL"] = st.session_state.ollama_url
    else:
        os.environ["MODELO"] = st.session_state.modelo_selecionado
    reset_config(); reset_brain()


def is_free() -> bool:
    if st.session_state.backend == "ollama":
        return True
    m = st.session_state.modelo_selecionado
    return m.endswith(":free") or "free" in m.lower() or m.startswith("openrouter/")


# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚖️ Tribunal IA Portugal V6")
    st.caption("Simulador judicial — República Portuguesa")
    st.divider()

    st.markdown("#### 🤖 Motor de IA")
    backend = st.radio(
        "Backend:", ["☁️ OpenRouter (cloud)", "🖥️ Ollama (local)"],
        index=0 if st.session_state.backend == "openrouter" else 1,
        label_visibility="collapsed",
    )
    novo_backend = "ollama" if "Ollama" in backend else "openrouter"
    if novo_backend != st.session_state.backend:
        st.session_state.backend = novo_backend
        aplicar_modelo()

    if st.session_state.backend == "openrouter":
        tipo = st.radio("Tipo:", ["🆓 Gratuitos", "💳 Pagos"],
                        horizontal=True, label_visibility="collapsed")
        opcoes = list(MODELOS_FREE.keys()) if "Gratuitos" in tipo else list(MODELOS_PAGOS.keys())
        nome_atual = MODELO_PARA_NOME.get(st.session_state.modelo_selecionado, opcoes[0])
        if nome_atual not in opcoes:
            nome_atual = opcoes[0]
        escolhido = st.selectbox("Modelo:", opcoes, index=opcoes.index(nome_atual))
        novo_modelo = NOME_PARA_MODELO[escolhido]
        if novo_modelo != st.session_state.modelo_selecionado:
            st.session_state.modelo_selecionado = novo_modelo
            aplicar_modelo()
        badge = "badge-free" if is_free() else "badge-paid"
        txt = "🆓 GRÁTIS" if is_free() else "💳 PAGO"
        st.markdown(f'<span class="{badge}">{txt}</span>', unsafe_allow_html=True)
    else:
        novo_url = st.text_input("URL Ollama:", value=st.session_state.ollama_url)
        if novo_url != st.session_state.ollama_url:
            st.session_state.ollama_url = novo_url; aplicar_modelo()
        novo_mod = st.selectbox("Modelo Ollama:", MODELOS_OLLAMA + ["outro"],
                                index=MODELOS_OLLAMA.index(st.session_state.ollama_modelo)
                                if st.session_state.ollama_modelo in MODELOS_OLLAMA else len(MODELOS_OLLAMA))
        if novo_mod == "outro":
            novo_mod = st.text_input("Modelo:", value=st.session_state.ollama_modelo)
        if novo_mod != st.session_state.ollama_modelo:
            st.session_state.ollama_modelo = novo_mod; aplicar_modelo()
        st.markdown('<span class="badge-local">🖥️ LOCAL</span>', unsafe_allow_html=True)

    # LangGraph status
    try:
        import langgraph  # noqa
        st.markdown('<span class="badge-lg">🔀 LangGraph activo</span>', unsafe_allow_html=True)
    except ImportError:
        st.caption("⚡ Orquestração imperativa (LangGraph não instalado)")

    st.divider()
    st.markdown("#### ⚙️ Opções V6")
    st.session_state.modo_contraditorio = st.toggle(
        "⚔️ Modo Contraditório",
        value=st.session_state.modo_contraditorio,
        help="Activa o turno completo de Acusação e Defesa antes das sentenças.",
    )
    rag_modo_ui = st.radio(
        "RAG:", ["💨 BM25 (rápido, sem download)", "🔬 Híbrido (embeddings, 118MB+)"],
        horizontal=False, label_visibility="collapsed",
        index=0,
        help="BM25: sem downloads. Híbrido: descarrega modelo de embeddings (118MB mínimo)."
    )
    if "Híbrido" in rag_modo_ui:
        import os
        os.environ["RAG_MODO"] = "hibrido"
        st.caption("⬇️ Primeiro arranque: descarrega modelo (~118MB)")
    else:
        import os
        os.environ["RAG_MODO"] = "bm25"
        st.caption("✅ BM25 activo — sem downloads")

    st.divider()
    try:
        from src.historico import get_historico
        from src.utils.config import get_config
        cfg = get_config()
        if cfg.historico_enabled:
            hist = get_historico()
            stats = hist.estatisticas()
            st.markdown("#### 📋 Histórico")
            st.caption(f"**{stats['total']}** casos processados")
    except Exception:
        pass

    st.divider()
    try:
        from src.cache import get_cache, limpar_cache
        sc = get_cache().estatisticas()
        st.caption(f"Cache: **{sc['entradas']}** entradas")
        if st.button("🗑️ Limpar cache"):
            limpar_cache(0); st.success("Limpo")
    except Exception:
        pass

    st.divider()
    st.markdown(
        '<div style="font-size:.72rem;color:#aaa;text-align:center;">'
        '⚠️ Fins educativos<br>'
        '<a href="https://www.oa.pt" target="_blank">Ordem dos Advogados</a>'
        '</div>', unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏛️ Tribunal IA Portugal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Simulador judicial · Direito Português 🇵🇹 · V6</div>', unsafe_allow_html=True)
st.markdown("""
<div class="disclaimer">
⚠️ <strong>Aviso Legal:</strong> Simulação educativa. Não constitui parecer jurídico.
Para situações reais: <a href="https://www.oa.pt" target="_blank">Ordem dos Advogados de Portugal</a>.
</div>
""", unsafe_allow_html=True)

# Separação de papéis (colapsável)
with st.expander("ℹ️ O que este sistema pode e não pode fazer — Declaração de Separação de Papéis", expanded=False):
    from src.auditoria import DISCLAIMER_SEPARACAO_PAPEIS
    st.code(DISCLAIMER_SEPARACAO_PAPEIS, language=None)

# Config check
os.environ.setdefault("BACKEND", st.session_state.backend)
os.environ.setdefault("MODELO", st.session_state.modelo_selecionado)
try:
    from src.utils.config import get_config
    cfg = get_config()
except Exception as e:
    st.error(f"❌ Configuração: {e}")
    st.code("OPENROUTER_API_KEY=a_tua_chave\nMODELO=openrouter/free", language="bash")
    st.stop()

# Steps
step = st.session_state.step
labels = ["1·Caso", "2·Docs", "3·Instrução", "4·Contraditório", "5·Processo", "6·Resultado"]
cols = st.columns(6)
for i, label in enumerate(labels):
    with cols[i]:
        if i + 1 < step:
            st.markdown(f'<div class="step-done">✅ {label}</div>', unsafe_allow_html=True)
        elif i + 1 == step:
            st.markdown(f'<div class="step-active">▶ {label}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="step-waiting">{label}</div>', unsafe_allow_html=True)
st.divider()


# ══════════════════════════════════════════════════════════════════════════
# PASSO 1 — CASO
# ══════════════════════════════════════════════════════════════════════════
if step == 1:
    st.markdown("### 📝 Descreve o caso")
    case_input = st.text_area(
        "Caso:", value=st.session_state.case_description, height=220,
        placeholder="Descreve o teu caso em linguagem comum. Quanto mais detalhe, melhor.",
        label_visibility="collapsed",
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        st.session_state.auto_detect = st.checkbox(
            "🔎 Detectar tribunal automaticamente", value=st.session_state.auto_detect)
    with c2:
        if not st.session_state.auto_detect:
            from src.pipeline.instancias import INSTANCIAS
            opts = {f"{k} — {v.nome}": k for k, v in INSTANCIAS.items()}
            st.session_state.instancia = opts[st.selectbox("Tribunal:", list(opts.keys()))]

    if st.button("▶ Avançar", type="primary", disabled=not case_input.strip()):
        try:
            from src.auditoria import validar_input
            val = validar_input(case_input, campo="caso")
            if not val.valido:
                st.error(f"❌ {'; '.join(val.avisos)}")
                st.stop()
            if val.avisos:
                for av in val.avisos:
                    st.warning(f"⚠️ {av}")
            case_input = val.texto_sanitizado or case_input
        except ImportError:
            pass
        st.session_state.case_description = case_input
        if st.session_state.auto_detect:
            from src.pipeline.instancias import detectar_instancia_por_keywords
            st.session_state.instancia = detectar_instancia_por_keywords(case_input)
        st.session_state.perguntas = None
        st.session_state.respostas = {}
        st.session_state.pdf_docs = []
        # Reset contraditório ao mudar de caso
        st.session_state._contr_detetive = None
        st.session_state._contr_acusacao = None
        st.session_state._contr_defesa = None
        st.session_state.feedback_contraditorio = ""
        st.session_state.argumento_defesa = ""
        st.session_state.step = 2
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# PASSO 2 — DOCUMENTOS
# ══════════════════════════════════════════════════════════════════════════
elif step == 2:
    from src.pipeline.instancias import INSTANCIAS
    inst = INSTANCIAS[st.session_state.instancia]
    st.markdown(f"### 📎 Documentos — {inst.nome}")
    st.caption("Carrega PDFs (contratos, certidões, relatórios). Opcional.")

    uploaded = st.file_uploader("PDFs:", type=["pdf"], accept_multiple_files=True)
    docs_processados = []
    if uploaded:
        from src.export import extrair_texto_pdf
        from src.utils.brain import get_brain
        from src.agents import PDFExtractorAgent
        from src.utils.logger import get_logger
        with st.spinner("A processar documentos..."):
            for f in uploaded[:5]:
                bts = f.read()
                texto, tipo = extrair_texto_pdf(bts)
                if texto and not texto.startswith("PyMuPDF"):
                    try:
                        resumo = PDFExtractorAgent(get_brain(), get_logger()).executar(texto, tipo)
                        docs_processados.append(resumo)
                        st.success(f"✅ {f.name} ({tipo})")
                    except Exception as ex:
                        docs_processados.append(f"[{f.name}]\n{texto[:800]}")
                        st.warning(f"⚠️ {f.name}: {str(ex)[:80]}")
    st.session_state.pdf_docs = docs_processados

    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅ Voltar"): st.session_state.step = 1; st.rerun()
    with c2:
        if st.button("▶ Avançar para Instrução", type="primary"):
            st.session_state.step = 3; st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# PASSO 3 — INSTRUÇÃO
# ══════════════════════════════════════════════════════════════════════════
elif step == 3:
    from src.pipeline.instancias import INSTANCIAS
    inst = INSTANCIAS[st.session_state.instancia]
    st.markdown(f"### 🔍 Instrução — {inst.nome}")

    col_info, col_skip = st.columns([3, 1])
    with col_info:
        st.info("O sistema gera perguntas **específicas** a este caso. As respostas enriquecem todas as peças.")
    with col_skip:
        if st.button("⏭ Saltar", use_container_width=True):
            st.session_state.perguntas = {"perguntas": [], "introducao": ""}
            st.session_state.step = 4 if st.session_state.modo_contraditorio else 5
            st.session_state.resultado = None; st.rerun()

    if st.session_state.perguntas is None:
        _case = str(st.session_state.get("case_description", ""))
        _inst = str(st.session_state.get("instancia", "TIC"))
        _res: dict = {"perguntas": None, "erro": None}

        def _gerar():
            try:
                _res["perguntas"] = get_processor().gerar_perguntas_instrucao(_case, _inst)
            except Exception as ex:
                _res["erro"] = str(ex)

        t = threading.Thread(target=_gerar, daemon=True)
        t.start()
        with st.spinner("A gerar perguntas específicas ao caso..."):
            t.join(timeout=150)

        if t.is_alive():
            st.session_state.perguntas = {"perguntas":[], "introducao":"", "_timeout": True}
        elif _res["erro"]:
            st.session_state.perguntas = {"perguntas":[], "introducao":"", "_erro": _res["erro"]}
        else:
            st.session_state.perguntas = _res["perguntas"]
        st.rerun()

    perguntas = st.session_state.perguntas
    if perguntas.get("_timeout") or perguntas.get("_erro"):
        msg = ("⏱️ Tempo esgotado (150s). Com modelos gratuitos isto é normal — "
               "tenta de novo ou usa o botão Saltar.") if perguntas.get("_timeout") else f"❌ {perguntas.get('_erro','')[:150]}"
        st.error(msg)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🔄 Tentar de novo"):
                st.session_state.perguntas = None; st.rerun()
        with c2:
            if st.button("⬅ Voltar"): st.session_state.step = 2; st.rerun()
        with c3:
            if st.button("▶ Avançar sem instrução", type="primary"):
                st.session_state.perguntas = {"perguntas":[], "introducao":""}
                st.session_state.step = 4 if st.session_state.modo_contraditorio else 5
                st.session_state.resultado = None; st.rerun()
        st.stop()

    if perguntas.get("introducao"):
        st.markdown(f"*{perguntas['introducao']}*")

    with st.form("instrucao_form"):
        for p in perguntas.get("perguntas", []):
            badge = {"critica":"🔴","relevante":"🟡","complementar":"🟢"}.get(p.get("importancia",""),"⚪")
            st.markdown(f"**{badge} [{p.get('categoria','?')}]** {p.get('texto','')}")
            if p.get("razao"):
                st.caption(f"_Porquê: {p['razao']}_")
            resp = st.text_area("", key=f"resp_{p['id']}", height=60,
                                value=st.session_state.respostas.get(p["id"],""),
                                placeholder="Responde ou deixa em branco...")
            st.session_state.respostas[p["id"]] = resp
            st.markdown("---")

        materiais = st.text_area("📎 Informações adicionais (opcional):",
                                 value=st.session_state.materiais, height=60)
        st.session_state.materiais = materiais

        c_vol, c_go = st.columns([1, 2])
        with c_vol: voltar = st.form_submit_button("⬅ Voltar")
        with c_go:  go = st.form_submit_button("▶ Avançar", type="primary")

    if voltar: st.session_state.step = 2; st.rerun()
    if go:
        st.session_state.step = 4 if st.session_state.modo_contraditorio else 5
        st.session_state.resultado = None; st.session_state.erro = None; st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# PASSO 4 — CONTRADITÓRIO (só se activado)
# ══════════════════════════════════════════════════════════════════════════
elif step == 4:
    if not st.session_state.modo_contraditorio:
        st.session_state.step = 5; st.rerun()

    from src.pipeline.instancias import INSTANCIAS
    inst = INSTANCIAS[st.session_state.instancia]
    st.markdown(f"### ⚔️ Modo Contraditório — {inst.nome}")
    st.info(
        "Neste modo vês o processo contraditório completo: "
        "primeiro a **Acusação** apresenta os seus argumentos, "
        "depois podes acrescentar os teus como **Advogado de Defesa** "
        "e a IA gera a peça formal de defesa antes dos Juízes decidirem."
    )

    # ── FASE 1: Gerar Instrução + Acusação ───────────────────────────
    if not st.session_state.get("_contr_acusacao"):
        _case = str(st.session_state.get("case_description", ""))
        _inst_cod = str(st.session_state.get("instancia", "TIC"))
        _res2: dict = {"detetive": None, "acusacao": None, "erro": None}

        def _gerar_acusacao():
            try:
                from src.pipeline.case_processor import CaseProcessor
                from src.utils import anonymize_text as _anon
                proc = CaseProcessor()
                anon, _ = _anon(_case)
                ctx = proc._rag_ctx(anon, instancia=_inst_cod)
                inst_obj = INSTANCIAS.get(_inst_cod, INSTANCIAS["TIC"])
                _res2["detetive"] = proc._detetive.executar(anon, "", ctx, inst_obj)
                _res2["acusacao"] = proc._acusacao.executar(anon, _res2["detetive"], ctx, inst_obj)
            except Exception as ex:
                _res2["erro"] = str(ex)

        t = threading.Thread(target=_gerar_acusacao, daemon=True)
        t.start()
        with st.spinner("⚔️ A gerar Instrução e Acusação..."):
            t.join(timeout=180)

        if _res2["erro"]:
            st.error(f"Erro: {_res2['erro'][:200]}")
            if st.button("⏭ Avançar sem contraditório"):
                st.session_state.step = 5; st.rerun()
            st.stop()

        st.session_state._contr_detetive = _res2["detetive"]
        st.session_state._contr_acusacao = _res2["acusacao"]
        st.session_state._contr_defesa = None
        st.rerun()

    # ── Layout lado a lado: Acusação | Defesa ────────────────────────
    st.markdown("---")
    col_acus, col_def = st.columns(2)

    with col_acus:
        st.markdown("#### ⚔️ Alegações da Acusação")
        st.markdown(
            f'<div style="border-left:4px solid #c0392b; background:#fff5f5; '
            f'border-radius:6px; padding:.8rem 1rem; min-height:200px;">'
            f'{(st.session_state._contr_acusacao or "").replace(chr(10), "<br>")}'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_def:
        st.markdown("#### 🛡️ Alegações da Defesa")
        if st.session_state.get("_contr_defesa"):
            st.markdown(
                f'<div style="border-left:4px solid #1e7e34; background:#f5fff7; '
                f'border-radius:6px; padding:.8rem 1rem; min-height:200px;">'
                f'{st.session_state._contr_defesa.replace(chr(10), "<br>")}'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="border-left:4px solid #bbb; background:#f8f9fa; '
                'border-radius:6px; padding:.8rem 1rem; min-height:200px; color:#888;">'
                '🕐 A aguardar...<br><br>'
                'Escreve os teus argumentos abaixo e clica em <strong>🛡️ Gerar Defesa</strong>.'
                '</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── FASE 2: Argumento do advogado ────────────────────────────────
    st.markdown("#### ✍️ O teu argumento de defesa")
    st.caption(
        "Apresenta os argumentos que consideras relevantes para a defesa. "
        "A IA incorpora-os na peça formal. Podes regenerar quantas vezes quiseres."
    )

    arg = st.text_area(
        "Argumento:", value=st.session_state.argumento_defesa, height=150,
        placeholder=(
            "Ex: O meu cliente tem um álibi sólido para o período em causa — "
            "estava em Coimbra conforme provam as câmeras de segurança. "
            "Além disso, a prova documental apresentada pela acusação foi obtida ilegalmente..."
        ),
        label_visibility="collapsed",
    )
    st.session_state.argumento_defesa = arg

    c_aval, c_gerar, c_reset = st.columns(3)

    # Botão: avaliar argumento (feedback rápido do juiz presidente)
    with c_aval:
        if st.button("💬 Avaliar argumento", disabled=not arg.strip(),
                     help="O Juiz Presidente avalia a força jurídica do teu argumento."):
            _inst_cod2 = str(st.session_state.instancia)
            _arg = arg
            _acus = str(st.session_state._contr_acusacao or "")
            _det = str(st.session_state._contr_detetive or "")
            _fb: dict = {"feedback": ""}

            def _avaliar():
                try:
                    from src.agents import ContraditórioFeedbackAgent
                    from src.utils.brain import get_brain
                    from src.utils.logger import get_logger
                    inst_obj = INSTANCIAS.get(_inst_cod2, INSTANCIAS["TIC"])
                    ag = ContraditórioFeedbackAgent(get_brain(), get_logger())
                    _fb["feedback"] = ag.executar(inst_obj, _arg, _acus, _det)
                except Exception as ex:
                    _fb["feedback"] = f"Erro: {ex}"

            t2 = threading.Thread(target=_avaliar, daemon=True)
            t2.start()
            with st.spinner("💬 A avaliar o argumento..."):
                t2.join(timeout=90)
            st.session_state.feedback_contraditorio = _fb["feedback"]
            st.rerun()

    # Botão: gerar a defesa completa
    with c_gerar:
        if st.button("🛡️ Gerar Defesa", type="primary", disabled=not arg.strip(),
                     help="Gera as alegações formais da defesa incorporando o teu argumento."):
            _case_d = str(st.session_state.get("case_description", ""))
            _inst_cod_d = str(st.session_state.get("instancia", "TIC"))
            _arg_d = arg
            _det_d = str(st.session_state._contr_detetive or "")
            _acus_d = str(st.session_state._contr_acusacao or "")
            _def_result: dict = {"defesa": None, "erro": None}

            def _gerar_defesa():
                try:
                    from src.pipeline.case_processor import CaseProcessor
                    from src.utils import anonymize_text as _anon
                    from src.agents import DefesaAgent
                    from src.utils.brain import get_brain
                    from src.utils.logger import get_logger
                    proc = CaseProcessor()
                    anon, _ = _anon(_case_d)
                    ctx = proc._rag_ctx(anon, instancia=_inst_cod_d)
                    inst_obj = INSTANCIAS.get(_inst_cod_d, INSTANCIAS["TIC"])
                    ag = DefesaAgent(get_brain(), get_logger())
                    _def_result["defesa"] = ag.executar(
                        anon, _det_d, _acus_d, ctx, inst_obj,
                        intervencao_utilizador=_arg_d,
                    )
                except Exception as ex:
                    _def_result["erro"] = str(ex)

            t3 = threading.Thread(target=_gerar_defesa, daemon=True)
            t3.start()
            with st.spinner("🛡️ A gerar as Alegações da Defesa..."):
                t3.join(timeout=180)

            if _def_result["erro"]:
                st.error(f"❌ Erro ao gerar defesa: {_def_result['erro'][:200]}")
            elif _def_result["defesa"]:
                st.session_state._contr_defesa = _def_result["defesa"]
                st.success("✅ Defesa gerada! Revê acima e avança quando estiveres pronto.")
            st.rerun()

    # Botão: regenerar acusação (recomeçar contraditório)
    with c_reset:
        if st.button("🔄 Regenerar acusação",
                     help="Apaga tudo e gera uma nova acusação."):
            st.session_state._contr_acusacao = None
            st.session_state._contr_detetive = None
            st.session_state._contr_defesa = None
            st.session_state.feedback_contraditorio = ""
            st.rerun()

    # Feedback do juiz ao argumento
    if st.session_state.feedback_contraditorio:
        with st.expander("💬 Avaliação jurídica do argumento pelo Juiz Presidente", expanded=True):
            st.markdown(st.session_state.feedback_contraditorio)

    st.markdown("---")

    # Aviso se a defesa ainda não foi gerada
    if not st.session_state.get("_contr_defesa"):
        st.warning(
            "⚠️ A Defesa ainda não foi gerada. "
            "Clica em **🛡️ Gerar Defesa** para a IA produzir as alegações formais. "
            "Se avançares sem gerar, a defesa será criada automaticamente no Passo 5 "
            "sem o teu contributo pessoal."
        )

    # Navegação
    c_vol, c_go = st.columns(2)
    with c_vol:
        if st.button("⬅ Voltar"):
            st.session_state.step = 3; st.rerun()
    with c_go:
        label_go = (
            "▶ Avançar para Processo (com defesa gerada)"
            if st.session_state.get("_contr_defesa")
            else "▶ Avançar (a defesa será gerada automaticamente)"
        )
        if st.button(label_go, type="primary"):
            st.session_state.step = 5
            st.session_state.resultado = None
            st.session_state.erro = None
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# PASSO 5 — PROCESSAR
# ══════════════════════════════════════════════════════════════════════════
elif step == 5:
    st.markdown("### ⚖️ Processo Judicial em curso...")

    if st.session_state.erro:
        st.error(f"❌ {st.session_state.erro}")
        if is_free():
            st.info("💡 Rate limit? Aguarda 1-2 min e tenta novamente. Ou muda de modelo na sidebar.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 Tentar novamente"):
                st.session_state.erro = None; st.rerun()
        with c2:
            if st.button("⬅ Voltar"):
                st.session_state.step = 3; st.session_state.erro = None; st.rerun()
        st.stop()

    if st.session_state.resultado is not None:
        st.session_state.step = 6; st.rerun()

    # Construir dados de instrução
    dados_instrucao = None
    respostas_v = {
        k: {
            "pergunta": next(
                (p["texto"] for p in (st.session_state.perguntas or {}).get("perguntas",[]) if p["id"]==k), ""),
            "categoria": next(
                (p["categoria"] for p in (st.session_state.perguntas or {}).get("perguntas",[]) if p["id"]==k), ""),
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

    # Capturar ANTES do thread
    _case_p5    = str(st.session_state.get("case_description",""))
    _inst_p5    = str(st.session_state.get("instancia","TIC"))
    _pdfs_p5    = list(st.session_state.get("pdf_docs",[]) or [])
    _backend_p5 = str(st.session_state.get("backend","openrouter"))
    _modelo_p5  = str(st.session_state.get("modelo_selecionado","openrouter/free"))
    _ollama_m5  = str(st.session_state.get("ollama_modelo","llama3.3:70b"))
    _ollama_u5  = str(st.session_state.get("ollama_url","http://localhost:11434"))
    _arg_def    = str(st.session_state.get("argumento_defesa","")).strip() or None
    # Defesa pré-gerada no modo contraditório (evita regenerar)
    _defesa_pre = str(st.session_state.get("_contr_defesa") or "").strip() or None

    _res5: dict = {"resultado": None, "erro": None}

    def _processar():
        import os as _os
        _os.environ["BACKEND"] = _backend_p5
        if _backend_p5 == "ollama":
            _os.environ["OLLAMA_MODELO"] = _ollama_m5
            _os.environ["OLLAMA_URL"] = _ollama_u5
        else:
            _os.environ["MODELO"] = _modelo_p5
        try:
            from src.utils.config import reset_config
            from src.utils.brain import reset_brain
            reset_config(); reset_brain()
            reset_processor()
            proc = get_processor()
            _res5["resultado"] = proc.process(
                case_description=_case_p5,
                instancia_codigo=_inst_p5,
                dados_instrucao=dados_instrucao,
                gerar_pdf=True,
                pdf_docs_extraidos=_pdfs_p5 or None,
                intervencao_utilizador=_arg_def,
                defesa_pre_gerada=_defesa_pre,
            )
        except Exception as ex:
            _res5["erro"] = str(ex)

    t = threading.Thread(target=_processar, daemon=True)
    t.start()

    # Barra de progresso — adapta se a defesa já foi gerada
    if _defesa_pre:
        agentes = ["🔍 Instrução", "⚔️ Acusação",
                   "🛡️ Defesa (já gerada — a reutilizar)",
                   "⚖️ Juiz Rigoroso","⚖️ Juiz Garantista","⚖️ Juiz Equilibrado",
                   "📊 Consistência","🌍 TEDH"]
    else:
        agentes = ["🔍 Instrução", "⚔️ Acusação", "🛡️ Defesa",
                   "⚖️ Juiz Rigoroso","⚖️ Juiz Garantista","⚖️ Juiz Equilibrado",
                   "📊 Consistência","🌍 TEDH"]

    import time as _time
    prog = st.empty()
    for i, ag in enumerate(agentes):
        if not t.is_alive():
            break
        prog.progress((i+1)/len(agentes), text=f"A processar: {ag}...")
        t.join(timeout=8)
    t.join(timeout=600)
    prog.empty()

    if _res5["erro"]:
        st.session_state.erro = _res5["erro"]
    elif _res5["resultado"]:
        st.session_state.resultado = _res5["resultado"]
        st.session_state.step = 6
    else:
        st.session_state.erro = "Timeout máximo atingido."
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# PASSO 6 — RESULTADO
# ══════════════════════════════════════════════════════════════════════════
elif step == 6:
    result = st.session_state.resultado
    if result is None:
        st.warning("Sem resultado.")
        if st.button("⬅ Recomeçar"): reset_all()
        st.stop()

    st.markdown("### 📄 Resultado do Processo")

    # Métricas
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    with m1: st.metric("Tribunal", result.instancia_codigo)
    with m2: st.metric("Incerteza", result.grau_incerteza)
    with m3: custo="🆓" if result.custo_total_usd==0 else f"${result.custo_total_usd:.4f}"; st.metric("Custo", custo)
    with m4: st.metric("RGPD", f"{len(result.entities_found)} entidades")
    with m5: st.metric("Orquestração", "LangGraph" if result.modelo_usado else "Imperativo")
    with m6:
        if st.button("🔄 Novo caso"): reset_all()

    # Grau de incerteza visual
    cor_map = {"Baixo":"🟢","Médio":"🟡","Alto":"🔴","Muito Alto":"🔴🔴"}
    emoji = cor_map.get(result.grau_incerteza, "")
    if emoji:
        st.markdown(f"**{emoji} Grau de incerteza: {result.grau_incerteza}**")

    tabs = st.tabs([
        "📋 Peças", "⚖️ Sentenças", "📊 Consistência",
        "🌍 TEDH", "📄 Ata", "🔗 Auditoria", "🕐 Histórico",
    ])

    with tabs[0]:
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("#### 🔍 Instrução")
            st.markdown(result.detetive_report or "_Não disponível_")
        with c2:
            st.markdown("#### ⚔️ Acusação")
            st.markdown(result.acusacao or "_Não disponível_")
        with c3:
            st.markdown("#### 🛡️ Defesa")
            if st.session_state.get("_contr_defesa"):
                st.success("✅ Incorpora o teu argumento de defesa")
            elif st.session_state.get("argumento_defesa"):
                st.info("💬 Inclui argumento do advogado de defesa")
            st.markdown(result.defesa or "_Não disponível_")
        if result.validacao_citacoes:
            with st.expander("🔎 Validação de citações"):
                st.markdown(result.validacao_citacoes)

    with tabs[1]:
        for titulo, texto, cls in [
            ("🔴 Rigoroso",   result.sentenca_rigorosa,   "sentenca-r"),
            ("🟢 Garantista", result.sentenca_garantista, "sentenca-g"),
            ("🔵 Equilibrado",result.sentenca_equilibrada,"sentenca-e"),
        ]:
            with st.expander(titulo, expanded=False):
                st.markdown(
                    f'<div class="{cls}">{(texto or "_Não disponível_").replace(chr(10),"<br>")}</div>',
                    unsafe_allow_html=True,
                )

    with tabs[2]:
        st.markdown("#### 📊 Consistência e Incerteza")
        if result.relatorio_consistencia:
            st.markdown(result.relatorio_consistencia)
        else:
            st.info("Não gerado.")
        import re
        def disp(t):
            if not t: return "_N/D_"
            m = re.search(r"(?:CONDENA|ABSOLVE|JULGA)[^.]*\.", t, re.IGNORECASE)
            return (m.group(0)[:200] if m else t[:150]) + "..."
        c1,c2,c3 = st.columns(3)
        with c1: st.markdown("**🔴**"); st.error(disp(result.sentenca_rigorosa))
        with c2: st.markdown("**🟢**"); st.success(disp(result.sentenca_garantista))
        with c3: st.markdown("**🔵**"); st.info(disp(result.sentenca_equilibrada))

    with tabs[3]:
        st.markdown("#### 🌍 Análise TEDH / ECHR")
        if result.analise_tedh:
            st.markdown(result.analise_tedh)
        else:
            st.info(
                "Sem dados TEDH disponíveis. "
                "Adiciona ficheiros de jurisprudência em `data/tedh/` para activar esta análise."
            )

    with tabs[4]:
        st.text_area("", value=result.ata_final or "", height=450,
                     disabled=True, label_visibility="collapsed")
        c1,c2,c3 = st.columns(3)
        with c1:
            st.download_button("⬇️ TXT", data=(result.ata_final or "").encode(),
                               file_name=f"{result.case_id}.txt", mime="text/plain")
        with c2:
            if result.pdf_bytes:
                st.download_button("⬇️ PDF", data=result.pdf_bytes,
                                   file_name=f"{result.case_id}.pdf", mime="application/pdf")
            else:
                st.caption("PDF: instala `reportlab`")
        with c3:
            st.caption(f"`{result.case_id}` | Hash: `{result.doc_hash}`")

    with tabs[5]:
        st.markdown("#### 🔗 Cadeia de Auditoria — Git Jurídico")
        try:
            from src.auditoria import get_cadeia_auditoria, analisar_dissenso, DISCLAIMER_SEPARACAO_PAPEIS
            cadeia = get_cadeia_auditoria()
            resumo_c = cadeia.resumo()
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Blocos na cadeia", resumo_c["total_blocos"])
            with c2: st.metric("Integridade", "✅ OK" if resumo_c["cadeia_integra"] else "❌ FALHOU")
            with c3:
                ultimo = resumo_c.get("ultimo_hash","—")
                st.metric("Último hash", f"{ultimo[:16]}..." if ultimo else "—")

            if not resumo_c["cadeia_integra"]:
                for err in resumo_c["erros"]:
                    st.error(f"⚠️ {err}")

            if result.voto_vencido:
                vv = result.voto_vencido
                st.markdown("---")
                st.markdown("#### 🗳️ Voto de Vencido")
                st.warning(
                    f"O juiz **{vv.perfil_divergente.title()}** discordou da maioria "
                    f"({vv.sentido_divergente}).\n\n"
                    f"**Fundamento:** {vv.fundamento_resumo[:300]}"
                )
                if vv.artigos_divergentes:
                    st.caption(f"Artigos em causa: {', '.join(vv.artigos_divergentes)}")
            else:
                st.info("Sem voto de vencido — os 3 perfis convergiram na decisão.")

            st.markdown("---")
            aud_txt = cadeia.exportar_auditoria()
            st.download_button("⬇️ Exportar cadeia de auditoria", data=aud_txt,
                               file_name="auditoria_tribunal_ia.txt", mime="text/plain")

            with st.expander("📋 Declaração de Separação de Papéis"):
                st.code(DISCLAIMER_SEPARACAO_PAPEIS, language=None)
        except Exception as ex:
            st.warning(f"Auditoria: {ex}")

    with tabs[6]:
        st.markdown("#### 🕐 Histórico")
        try:
            from src.historico import get_historico
            from src.pipeline.instancias import INSTANCIAS
            hist = get_historico()
            q = st.text_input("🔍 Pesquisar:", placeholder="Texto, tribunal...")
            fi = st.selectbox("Tribunal:", ["Todos"]+list(INSTANCIAS.keys()),
                              label_visibility="collapsed")
            registos = hist.pesquisar(query=q, instancia=None if fi=="Todos" else fi, limite=20)
            for r in registos:
                with st.expander(f"📄 {r.id} — {r.instancia_codigo} — {r.grau_incerteza}"):
                    st.caption(f"{r.timestamp[:19].replace('T',' ')} | {r.modelo}")
                    st.markdown(r.resumo)
                    st.markdown(f"**Decisão:** {r.dispositivo}")
                    if r.ata_path and Path(r.ata_path).exists():
                        st.download_button("⬇️ Ata", Path(r.ata_path).read_text(encoding="utf-8"),
                                           file_name=Path(r.ata_path).name, key=f"dl_{r.id}")
            if st.button("🗑️ Limpar histórico"):
                hist.limpar(); st.rerun()
        except Exception as ex:
            st.warning(f"Histórico: {ex}")
