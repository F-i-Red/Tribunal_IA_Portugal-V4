"""
CaseProcessor V6 — Orquestração LangGraph + fallback imperativo
════════════════════════════════════════════════════════════════
Fluxo LangGraph:
  anonimizar → rag → instrucao → detetive → acusacao
  → defesa → juizes (3) → consistencia → tedh → ata

Fallback automático para orquestração imperativa se LangGraph
não estiver instalado.
"""
from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from ..utils import get_config, get_logger, anonymize_text
from ..utils.brain import get_brain
from ..agents import (
    DetetiveAgent, AcusacaoAgent, DefesaAgent,
    JuizAgent, InstrucaoAgent, ConsistenciaAgent,
    TEDHAgent, PDFExtractorAgent,
)
from ..rag import MotorRAG, ValidadorCitacoes
from ..export import exportar_pdf
from ..historico import get_historico, criar_registo
from ..auditoria import (
    get_cadeia_auditoria, ProvenanceLog, validar_input,
    analisar_dissenso, DISCLAIMER_SEPARACAO_PAPEIS,
)
from .instancias import INSTANCIAS, InstanciaJudicial, detectar_instancia_por_keywords


# ── Estado do grafo LangGraph ────────────────────────────────────────
class EstadoCaso(TypedDict, total=False):
    # Input
    case_description: str
    instancia_codigo: str
    dados_instrucao: Optional[Dict]
    pdf_docs: Optional[List[str]]
    intervencao_utilizador: Optional[str]
    defesa_pre_gerada: Optional[str]      # defesa gerada no modo contraditório

    # Computado
    anon_text: str
    entities: List[Dict]
    inst: Any  # InstanciaJudicial
    ctx_rag: str
    ctx_rag_tedh: str
    ctx_instrucao: str
    detetive: str
    acusacao: str
    defesa: str
    sentenca_rigorosa: str
    sentenca_garantista: str
    sentenca_equilibrada: str
    relatorio_consistencia: str
    grau_incerteza: str
    analise_tedh: str
    validacao_citacoes: str
    ata_final: str
    doc_hash: str
    case_id: str
    trace_id: str
    custo_usd: float
    modelo_usado: str
    backend_usado: str


@dataclass
class CaseResult:
    case_id: str
    trace_id: str
    original_description: str
    anonymized_description: str
    entities_found: List[Dict]
    instancia_codigo: str = ""
    instancia_nome: str = ""
    modelo_usado: str = ""
    backend_usado: str = ""
    dados_instrucao: Optional[Dict] = None
    detetive_report: Optional[str] = None
    acusacao: Optional[str] = None
    defesa: Optional[str] = None
    sentenca_rigorosa: Optional[str] = None
    sentenca_garantista: Optional[str] = None
    sentenca_equilibrada: Optional[str] = None
    relatorio_consistencia: Optional[str] = None
    grau_incerteza: str = "N/A"
    analise_tedh: Optional[str] = None
    ata_final: Optional[str] = None
    ata_path: Optional[Path] = None
    pdf_bytes: Optional[bytes] = None
    contexto_rag: Optional[str] = None
    validacao_citacoes: Optional[str] = None
    custo_total_usd: float = 0.0
    doc_hash: str = ""
    voto_vencido: Optional[object] = None  # VotoVencido | None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class CaseProcessor:
    def __init__(self) -> None:
        self.config = get_config()
        self.logger = get_logger()
        self.brain = get_brain()

        # RAG V6 — híbrido com reranking
        from ..rag.motor import MotorRAG as MotorRAGV6
        self.rag = MotorRAGV6(
            Path("."),
            modo=self.config.rag_modo,
            embedding_modelo=self.config.rag_embedding_modelo,
            reranker_modelo=self.config.rag_reranker_modelo,
            usar_reranking=self.config.rag_reranking,
            top_k=self.config.rag_top_k,
            top_n=self.config.rag_top_n,
        )
        n = self.rag.indexar()
        if n > 0:
            self.logger.info(f"RAG V6: {n} fragmentos (modo={self.config.rag_modo})")

        self.validador = ValidadorCitacoes(Path("data/leis"))

        # Agentes
        self._detetive    = DetetiveAgent(self.brain, self.logger)
        self._acusacao    = AcusacaoAgent(self.brain, self.logger)
        self._defesa      = DefesaAgent(self.brain, self.logger)
        self._instrucao   = InstrucaoAgent(self.brain, self.logger)
        self._consistencia = ConsistenciaAgent(self.brain, self.logger)
        self._tedh        = TEDHAgent(self.brain, self.logger)

        # LangGraph — disponível?
        self._usar_langgraph = self.config.usar_langgraph
        if self._usar_langgraph:
            self._grafo = self._construir_grafo()
        else:
            self._grafo = None

    # ── RAG helpers ──────────────────────────────────────────────────
    def _rag_ctx(
        self, query: str, instancia: Optional[str] = None,
        lingua_filtro: Optional[str] = None,
    ) -> str:
        if not self.rag.tem_dados():
            return ""
        frags = self.rag.pesquisar(
            query,
            instancia=instancia,
            lingua_filtro=lingua_filtro,
        )
        if frags:
            self.logger.log_rag(query, len(frags), frags[0].relevancia)
        return self.rag.formatar_contexto(frags, incluir_tedh=(lingua_filtro != "en"))

    def _rag_tedh(self, query: str) -> str:
        if not self.rag.tem_dados():
            return ""
        frags = self.rag.pesquisar(query, tipo_filtro="tedh", lingua_filtro="en")
        if not frags:
            frags = self.rag.pesquisar(query, lingua_filtro="en")
        return self.rag.formatar_contexto(frags, incluir_tedh=True)

    # ── Instrução pública ─────────────────────────────────────────────
    def gerar_perguntas_instrucao(
        self, case_description: str, instancia_codigo: str = "TIC"
    ) -> Dict:
        inst = INSTANCIAS.get(instancia_codigo, INSTANCIAS["TIC"])
        ctx_rag = self._rag_ctx(case_description, instancia=instancia_codigo)
        return self._instrucao.executar(case_description, inst, ctx_rag)

    # Alias para compatibilidade com o passo 3 do app.py
    def _rag_ctx_n(self, query: str, instancia: str, n: int) -> str:
        if not self.rag.tem_dados():
            return ""
        frags = self.rag.pesquisar(query, instancia=instancia)
        return self.rag.formatar_contexto(frags)

    # ── Formatar instrução ────────────────────────────────────────────
    def _fmt_instrucao(self, dados: Optional[Dict]) -> str:
        if not dados or not dados.get("respostas"):
            return ""
        linhas = ["\n\n═══ ESCLARECIMENTOS DE INSTRUÇÃO ═══\n"]
        for item in dados["respostas"].values():
            r = item.get("resposta", "")
            if r and r not in ("", "Sem resposta"):
                linhas.append(f"[{item.get('categoria','?')}] {item.get('pergunta','')}")
                linhas.append(f"→ {r}\n")
        for m in dados.get("materiais", []):
            if m.get("descricao"):
                linhas.append(f"📎 {m['descricao']}")
        linhas.append("════════════════════════════════════\n")
        return "\n".join(linhas)

    # ══════════════════════════════════════════════════════════════════
    # ORQUESTRAÇÃO LANGGRAPH
    # ══════════════════════════════════════════════════════════════════
    def _construir_grafo(self):
        """Constrói o grafo LangGraph do pipeline judicial."""
        try:
            from langgraph.graph import StateGraph, END

            grafo = StateGraph(EstadoCaso)

            # Nós do grafo
            grafo.add_node("anonimizar",   self._node_anonimizar)
            grafo.add_node("rag",          self._node_rag)
            grafo.add_node("detetive",     self._node_detetive)
            grafo.add_node("acusacao",     self._node_acusacao)
            grafo.add_node("defesa",       self._node_defesa)
            grafo.add_node("juiz_rigoroso",   self._node_juiz_rigoroso)
            grafo.add_node("juiz_garantista", self._node_juiz_garantista)
            grafo.add_node("juiz_equilibrado",self._node_juiz_equilibrado)
            grafo.add_node("consistencia", self._node_consistencia)
            grafo.add_node("tedh",         self._node_tedh)
            grafo.add_node("finalizar",    self._node_finalizar)

            # Arestas — fluxo principal
            grafo.set_entry_point("anonimizar")
            grafo.add_edge("anonimizar",  "rag")
            grafo.add_edge("rag",         "detetive")
            grafo.add_edge("detetive",    "acusacao")
            grafo.add_edge("acusacao",    "defesa")
            grafo.add_edge("defesa",      "juiz_rigoroso")
            grafo.add_edge("juiz_rigoroso",    "juiz_garantista")
            grafo.add_edge("juiz_garantista",  "juiz_equilibrado")
            grafo.add_edge("juiz_equilibrado", "consistencia")
            grafo.add_edge("consistencia", "tedh")
            grafo.add_edge("tedh",         "finalizar")
            grafo.add_edge("finalizar",    END)

            return grafo.compile()
        except Exception as e:
            self.logger.warning(f"LangGraph não disponível: {e}. Usando orquestração imperativa.")
            self._usar_langgraph = False
            return None

    # ── Nós do grafo ─────────────────────────────────────────────────
    def _node_anonimizar(self, estado: EstadoCaso) -> EstadoCaso:
        self.logger.set_agent("anonimizar")
        anon_text, entities = anonymize_text(estado["case_description"])
        self.logger.log_anonymization(
            len(entities), list({e.label for e in entities})
        )
        # Integrar PDFs
        if estado.get("pdf_docs"):
            anon_text += "\n\n=== DOCUMENTOS SUBMETIDOS ===\n" + \
                         "\n\n".join(estado["pdf_docs"])
        inst = INSTANCIAS.get(estado["instancia_codigo"], INSTANCIAS["TIC"])
        ctx_instrucao = self._fmt_instrucao(estado.get("dados_instrucao"))
        return {
            **estado,
            "anon_text": anon_text,
            "entities": [{"text": e.text, "type": e.label} for e in entities],
            "inst": inst,
            "ctx_instrucao": ctx_instrucao,
        }

    def _node_rag(self, estado: EstadoCaso) -> EstadoCaso:
        self.logger.set_agent("rag")
        query = estado["anon_text"][:500] + " " + estado.get("ctx_instrucao", "")[:200]
        ctx_rag = self._rag_ctx(query, instancia=estado["instancia_codigo"])
        ctx_tedh = self._rag_tedh(query) if self.config.multilingue_enabled else ""
        return {**estado, "ctx_rag": ctx_rag, "ctx_rag_tedh": ctx_tedh}

    def _node_detetive(self, estado: EstadoCaso) -> EstadoCaso:
        self.logger.set_agent("detetive")
        result = self._detetive.executar(
            estado["anon_text"], estado.get("ctx_instrucao", ""),
            estado["ctx_rag"], estado["inst"],
        )
        return {**estado, "detetive": result}

    def _node_acusacao(self, estado: EstadoCaso) -> EstadoCaso:
        self.logger.set_agent("acusacao")
        result = self._acusacao.executar(
            estado["anon_text"], estado["detetive"],
            estado["ctx_rag"], estado["inst"],
        )
        return {**estado, "acusacao": result}

    def _node_defesa(self, estado: EstadoCaso) -> EstadoCaso:
        self.logger.set_agent("defesa")
        # Reutilizar defesa pré-gerada do modo contraditório (se disponível)
        if estado.get("defesa_pre_gerada"):
            self.logger.info("Defesa pré-gerada reutilizada (modo contraditório).")
            return {**estado, "defesa": estado["defesa_pre_gerada"]}
        result = self._defesa.executar(
            estado["anon_text"], estado["detetive"], estado["acusacao"],
            estado["ctx_rag"], estado["inst"],
            intervencao_utilizador=estado.get("intervencao_utilizador"),
        )
        return {**estado, "defesa": result}

    def _node_juiz_rigoroso(self, estado: EstadoCaso) -> EstadoCaso:
        self.logger.set_agent("juiz_rigoroso")
        r = JuizAgent(self.brain, self.logger, "rigoroso").executar(
            estado["anon_text"], estado["detetive"], estado["acusacao"],
            estado["defesa"], estado["inst"], estado["ctx_rag"],
        )
        return {**estado, "sentenca_rigorosa": r}

    def _node_juiz_garantista(self, estado: EstadoCaso) -> EstadoCaso:
        self.logger.set_agent("juiz_garantista")
        r = JuizAgent(self.brain, self.logger, "garantista").executar(
            estado["anon_text"], estado["detetive"], estado["acusacao"],
            estado["defesa"], estado["inst"], estado["ctx_rag"],
        )
        return {**estado, "sentenca_garantista": r}

    def _node_juiz_equilibrado(self, estado: EstadoCaso) -> EstadoCaso:
        self.logger.set_agent("juiz_equilibrado")
        r = JuizAgent(self.brain, self.logger, "equilibrado").executar(
            estado["anon_text"], estado["detetive"], estado["acusacao"],
            estado["defesa"], estado["inst"], estado["ctx_rag"],
        )
        return {**estado, "sentenca_equilibrada": r}

    def _node_consistencia(self, estado: EstadoCaso) -> EstadoCaso:
        self.logger.set_agent("consistencia")
        rel = ""
        grau = "N/A"
        if self.config.consistencia_check:
            try:
                rel = self._consistencia.executar(
                    estado["inst"],
                    estado["sentenca_rigorosa"],
                    estado["sentenca_garantista"],
                    estado["sentenca_equilibrada"],
                )
                m = re.search(
                    r"(?:GRAU DE INCERTEZA GLOBAL|Grau Global)[:\s]*(Baixo|Médio|Alto|Muito Alto)",
                    rel, re.IGNORECASE,
                )
                grau = m.group(1) if m else "N/A"
            except Exception as e:
                self.logger.warning(f"Consistência falhou: {e}")
        return {**estado, "relatorio_consistencia": rel, "grau_incerteza": grau}

    def _node_tedh(self, estado: EstadoCaso) -> EstadoCaso:
        self.logger.set_agent("tedh")
        analise = ""
        if self.config.multilingue_enabled and estado.get("ctx_rag_tedh"):
            try:
                analise = self._tedh.executar(
                    estado["inst"],
                    estado["anon_text"][:600],
                    estado["ctx_rag_tedh"],
                    lingua="pt",
                )
            except Exception as e:
                self.logger.warning(f"TEDH falhou: {e}")
        return {**estado, "analise_tedh": analise}

    def _node_finalizar(self, estado: EstadoCaso) -> EstadoCaso:
        self.logger.set_agent("finalizar")
        _, problemas = self.validador.validar_texto(
            " ".join(filter(None, [
                estado.get("acusacao", ""), estado.get("defesa", ""),
                estado.get("sentenca_rigorosa", ""),
                estado.get("sentenca_garantista", ""),
                estado.get("sentenca_equilibrada", ""),
            ]))
        )
        validacao = self.validador.relatorio_citacoes(problemas)
        return {**estado, "validacao_citacoes": validacao}

    # ══════════════════════════════════════════════════════════════════
    # PROCESSO PRINCIPAL
    # ══════════════════════════════════════════════════════════════════
    def process(
        self,
        case_description: str,
        instancia_codigo: Optional[str] = None,
        dados_instrucao: Optional[Dict] = None,
        gerar_pdf: bool = True,
        pdf_docs_extraidos: Optional[List[str]] = None,
        intervencao_utilizador: Optional[str] = None,
        defesa_pre_gerada: Optional[str] = None,
    ) -> CaseResult:

        # Validar input — threat model básico
        validacao = validar_input(case_description, campo="caso")
        if not validacao.valido:
            raise ValueError(f"Input inválido: {'; '.join(validacao.avisos)}")
        if validacao.texto_sanitizado:
            case_description = validacao.texto_sanitizado

        if not instancia_codigo:
            instancia_codigo = detectar_instancia_por_keywords(case_description)
        inst = INSTANCIAS.get(instancia_codigo, INSTANCIAS["TIC"])

        trace_id = self.logger.start_case(case_description)
        case_id = f"caso_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(
            f"V6 | {case_id} | {inst.nome} | "
            f"{'LangGraph' if self._usar_langgraph else 'imperativo'} | "
            f"{self.config.modelo_activo}"
        )

        # ── Orquestrar ────────────────────────────────────────────────
        if self._usar_langgraph and self._grafo:
            estado_final = self._orquestrar_langgraph(
                case_description, instancia_codigo, inst,
                dados_instrucao, pdf_docs_extraidos, intervencao_utilizador,
                defesa_pre_gerada,
            )
        else:
            estado_final = self._orquestrar_imperativo(
                case_description, instancia_codigo, inst,
                dados_instrucao, pdf_docs_extraidos, intervencao_utilizador,
                defesa_pre_gerada,
            )

        # ── Montar ata ────────────────────────────────────────────────
        cost_stats = self.brain.get_cost_stats()
        ata = self._montar_ata(case_id, trace_id, inst, estado_final, cost_stats)
        doc_hash = hashlib.sha256(ata.encode()).hexdigest()[:16]
        ata_final = (
            self._disclaimer(doc_hash, case_id)
            + ata
            + self._watermark(doc_hash, case_id, trace_id)
        )

        ata_path: Optional[Path] = None
        if self.config.guardar_atas:
            ata_path = self.config.pasta_atas / f"{case_id}.txt"
            ata_path.write_text(ata_final, encoding="utf-8")

        result = CaseResult(
            case_id=case_id,
            trace_id=trace_id,
            original_description=case_description,
            anonymized_description=estado_final.get("anon_text", ""),
            entities_found=estado_final.get("entities", []),
            instancia_codigo=instancia_codigo,
            instancia_nome=inst.nome,
            modelo_usado=self.config.modelo_activo,
            backend_usado=self.config.backend,
            dados_instrucao=dados_instrucao,
            detetive_report=estado_final.get("detetive"),
            acusacao=estado_final.get("acusacao"),
            defesa=estado_final.get("defesa"),
            sentenca_rigorosa=estado_final.get("sentenca_rigorosa"),
            sentenca_garantista=estado_final.get("sentenca_garantista"),
            sentenca_equilibrada=estado_final.get("sentenca_equilibrada"),
            relatorio_consistencia=estado_final.get("relatorio_consistencia"),
            grau_incerteza=estado_final.get("grau_incerteza", "N/A"),
            analise_tedh=estado_final.get("analise_tedh"),
            ata_final=ata_final,
            ata_path=ata_path,
            contexto_rag=estado_final.get("ctx_rag"),
            validacao_citacoes=estado_final.get("validacao_citacoes"),
            custo_total_usd=cost_stats["total_cost_usd"],
            doc_hash=doc_hash,
        )

        # PDF
        if gerar_pdf and self.config.exportar_pdf:
            try:
                pdf_path = self.config.pasta_atas / f"{case_id}.pdf"
                result.pdf_bytes = exportar_pdf(result, destino=pdf_path)
            except Exception as e:
                self.logger.warning(f"PDF falhou: {e}")

        # Cadeia de auditoria (Git jurídico)
        try:
            cadeia = get_cadeia_auditoria()
            cadeia.adicionar(
                case_id=result.case_id,
                instancia=result.instancia_codigo,
                modelo=result.modelo_usado,
                grau_incerteza=result.grau_incerteza,
                hash_ata=result.doc_hash,
            )
        except Exception as e:
            self.logger.warning(f"Cadeia auditoria: {e}")

        # Voto de vencido
        try:
            result.voto_vencido = analisar_dissenso(
                result.sentenca_rigorosa or "",
                result.sentenca_garantista or "",
                result.sentenca_equilibrada or "",
            )
        except Exception:
            result.voto_vencido = None

        # Histórico
        if self.config.historico_enabled:
            try:
                get_historico().adicionar(
                    criar_registo(result, result.grau_incerteza)
                )
            except Exception as e:
                self.logger.warning(f"Histórico falhou: {e}")

        return result

    # ── LangGraph ─────────────────────────────────────────────────────
    def _orquestrar_langgraph(
        self, case_description, instancia_codigo, inst,
        dados_instrucao, pdf_docs, intervencao_utilizador,
        defesa_pre_gerada=None,
    ) -> Dict:
        estado_inicial: EstadoCaso = {
            "case_description": case_description,
            "instancia_codigo": instancia_codigo,
            "dados_instrucao": dados_instrucao,
            "pdf_docs": pdf_docs,
            "intervencao_utilizador": intervencao_utilizador,
            "defesa_pre_gerada": defesa_pre_gerada,
            "inst": inst,
        }
        try:
            return dict(self._grafo.invoke(estado_inicial))
        except Exception as e:
            self.logger.warning(f"LangGraph falhou ({e}), fallback imperativo")
            return self._orquestrar_imperativo(
                case_description, instancia_codigo, inst,
                dados_instrucao, pdf_docs, intervencao_utilizador,
                defesa_pre_gerada,
            )

    # ── Imperativo (fallback robusto) ─────────────────────────────────
    def _orquestrar_imperativo(
        self, case_description, instancia_codigo, inst,
        dados_instrucao, pdf_docs, intervencao_utilizador,
        defesa_pre_gerada=None,
    ) -> Dict:
        # Anonimizar
        anon_text, entities = anonymize_text(case_description)
        self.logger.log_anonymization(
            len(entities), list({e.label for e in entities})
        )
        if pdf_docs:
            anon_text += "\n\n=== DOCUMENTOS SUBMETIDOS ===\n" + "\n\n".join(pdf_docs)

        ctx_instrucao = self._fmt_instrucao(dados_instrucao)
        query = anon_text[:500] + " " + ctx_instrucao[:200]

        # RAG
        ctx_rag = self._rag_ctx(query, instancia=instancia_codigo)
        ctx_rag_tedh = self._rag_tedh(query) if self.config.multilingue_enabled else ""

        # Agentes sequenciais
        detetive = self._detetive.executar(anon_text, ctx_instrucao, ctx_rag, inst)
        acusacao = self._acusacao.executar(anon_text, detetive, ctx_rag, inst)

        # Defesa — usa pré-gerada se disponível (modo contraditório)
        if defesa_pre_gerada:
            defesa = defesa_pre_gerada
            self.logger.info("Defesa pré-gerada reutilizada do modo contraditório.")
        else:
            defesa = self._defesa.executar(
                anon_text, detetive, acusacao, ctx_rag, inst,
                intervencao_utilizador=intervencao_utilizador,
            )

        # Sentenças — paralelo só se pago e configurado
        usar_paralelo = (
            self.config.paralelismo
            and not self.config.is_free_model
            and not self.config.usar_ollama
        )

        if usar_paralelo:
            s_rig, s_gar, s_equ = self._sentencas_paralelo(
                anon_text, detetive, acusacao, defesa, inst, ctx_rag
            )
        else:
            s_rig = JuizAgent(self.brain, self.logger, "rigoroso").executar(
                anon_text, detetive, acusacao, defesa, inst, ctx_rag)
            s_gar = JuizAgent(self.brain, self.logger, "garantista").executar(
                anon_text, detetive, acusacao, defesa, inst, ctx_rag)
            s_equ = JuizAgent(self.brain, self.logger, "equilibrado").executar(
                anon_text, detetive, acusacao, defesa, inst, ctx_rag)

        # Consistência
        rel_cons, grau = "", "N/A"
        if self.config.consistencia_check:
            try:
                rel_cons = self._consistencia.executar(inst, s_rig, s_gar, s_equ)
                m = re.search(
                    r"(?:GRAU DE INCERTEZA GLOBAL|Grau Global)[:\s]*(Baixo|Médio|Alto|Muito Alto)",
                    rel_cons, re.IGNORECASE,
                )
                grau = m.group(1) if m else "N/A"
            except Exception as e:
                self.logger.warning(f"Consistência: {e}")

        # TEDH
        analise_tedh = ""
        if self.config.multilingue_enabled and ctx_rag_tedh:
            try:
                analise_tedh = self._tedh.executar(
                    inst, anon_text[:600], ctx_rag_tedh, lingua="pt"
                )
            except Exception as e:
                self.logger.warning(f"TEDH: {e}")

        # Validação
        _, problemas = self.validador.validar_texto(
            " ".join(filter(None, [acusacao, defesa, s_rig, s_gar, s_equ]))
        )
        validacao = self.validador.relatorio_citacoes(problemas)

        return {
            "anon_text": anon_text,
            "entities": [{"text": e.text, "type": e.label} for e in entities],
            "inst": inst,
            "ctx_rag": ctx_rag,
            "ctx_rag_tedh": ctx_rag_tedh,
            "ctx_instrucao": ctx_instrucao,
            "detetive": detetive,
            "acusacao": acusacao,
            "defesa": defesa,
            "sentenca_rigorosa": s_rig,
            "sentenca_garantista": s_gar,
            "sentenca_equilibrada": s_equ,
            "relatorio_consistencia": rel_cons,
            "grau_incerteza": grau,
            "analise_tedh": analise_tedh,
            "validacao_citacoes": validacao,
        }

    def _sentencas_paralelo(
        self, case_text, detetive, acusacao, defesa, inst, ctx_rag
    ):
        resultados: Dict[str, str] = {}

        def _run(perfil: str) -> None:
            try:
                resultados[perfil] = JuizAgent(
                    self.brain, self.logger, perfil
                ).executar(case_text, detetive, acusacao, defesa, inst, ctx_rag)
            except Exception as e:
                resultados[perfil] = f"[SENTENÇA {perfil.upper()}: erro — {e}]"

        threads = [
            threading.Thread(target=_run, args=(p,), daemon=True)
            for p in ("rigoroso", "garantista", "equilibrado")
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=300)
        return (
            resultados.get("rigoroso", "[timeout]"),
            resultados.get("garantista", "[timeout]"),
            resultados.get("equilibrado", "[timeout]"),
        )

    # ── Ata ───────────────────────────────────────────────────────────
    def _montar_ata(
        self, case_id: str, trace_id: str,
        inst: InstanciaJudicial, e: Dict, cost_stats: Dict,
    ) -> str:
        now = datetime.now(timezone.utc)
        meses = ["janeiro","fevereiro","março","abril","maio","junho",
                 "julho","agosto","setembro","outubro","novembro","dezembro"]
        data_pt = f"{now.day} de {meses[now.month-1]} de {now.year}, {now.strftime('%H:%M')} UTC"
        sep = "═" * 72
        orq = "LangGraph" if self._usar_langgraph else "Imperativo"
        custo = "Gratuito" if cost_stats["total_cost_usd"] == 0 \
            else f"${cost_stats['total_cost_usd']:.4f}"

        sec_tedh = ""
        if e.get("analise_tedh"):
            sec_tedh = (
                f"\n{sep}\nSECÇÃO IX — ANÁLISE COMPARATIVA TEDH / ECHR\n{sep}\n\n"
                f"{e['analise_tedh']}\n"
            )

        sec_cons = ""
        if e.get("relatorio_consistencia"):
            sec_cons = (
                f"\n{sep}\nSECÇÃO X — CONSISTÊNCIA E INCERTEZA\n{sep}\n\n"
                f"{e['relatorio_consistencia']}\n"
            )

        sec_rag = ""
        if e.get("ctx_rag"):
            sec_rag = (
                f"\n{sep}\nSECÇÃO XII — CONTEXTO JURÍDICO RAG V6\n{sep}\n\n"
                f"{e['ctx_rag'][:1500]}\n"
            )

        return f"""{sep}
ATA DE SIMULAÇÃO JUDICIAL — TRIBUNAL IA PORTUGAL V6
{sep}

PROCESSO Nº  : {case_id}
TRACE ID     : {trace_id}
TRIBUNAL     : {inst.nome}
MATÉRIA      : {inst.materia}
DIPLOMA      : {inst.diploma_principal}
DATA         : {data_pt}
MODELO       : {cost_stats['modelo']} [{cost_stats['backend']}]
ORQUESTRAÇÃO : {orq}
RAG MODO     : {self.config.rag_modo}
CUSTO        : {custo}
ESTADO       : SIMULAÇÃO EDUCATIVA — SEM VALOR JURÍDICO

{sep}
SECÇÃO I — CASO (ANONIMIZADO — RGPD)
{sep}

{e.get('anon_text','')}

{sep}
SECÇÃO II — RELATÓRIO DE INSTRUÇÃO FACTUAL
{sep}

{e.get('detetive','')}

{sep}
SECÇÃO III — ALEGAÇÕES DA ACUSAÇÃO
{sep}

{e.get('acusacao','')}

{sep}
SECÇÃO IV — ALEGAÇÕES DA DEFESA
{sep}

{e.get('defesa','')}

{sep}
SECÇÃO V — {inst.termo_decisao.upper()}: PERFIL RIGOROSO
{sep}

{e.get('sentenca_rigorosa','')}

{sep}
SECÇÃO VI — {inst.termo_decisao.upper()}: PERFIL GARANTISTA
{sep}

{e.get('sentenca_garantista','')}

{sep}
SECÇÃO VII — {inst.termo_decisao.upper()}: PERFIL EQUILIBRADO
{sep}

{e.get('sentenca_equilibrada','')}

{sep}
SECÇÃO VIII — VALIDAÇÃO DE CITAÇÕES JURÍDICAS
{sep}

{e.get('validacao_citacoes','')}
{sec_tedh}{sec_cons}{sec_rag}
{sep}
SECÇÃO XI — NOTA EDUCATIVA
{sep}

Três decisões, mesmos factos, perfis distintos:
 • Rigoroso   → prevenção geral, rigor punitivo
 • Garantista → in dubio pro reo, garantias fundamentais
 • Equilibrado→ proporcionalidade, equidade

Orquestração: {orq} | RAG: {self.config.rag_modo}
Para situações reais: Ordem dos Advogados — www.oa.pt

"""

    def _disclaimer(self, h: str, cid: str) -> str:
        l = "═" * 70
        return (
            f"\n╔{l}╗\n"
            "║  ⚠️  AVISO LEGAL — SIMULAÇÃO EDUCATIVA                          ║\n"
            "║  Não constitui parecer jurídico. Para casos reais: www.oa.pt    ║\n"
            f"║  Hash: {h:<60}║\n"
            f"║  ID:   {cid:<60}║\n"
            f"╚{l}╝\n\n"
        )

    def _watermark(self, h: str, cid: str, tid: str) -> str:
        return (
            f"\n{'─'*70}\n"
            f"TRIBUNAL IA PORTUGAL V6 | Hash: {h} | ID: {cid} | Trace: {tid}\n"
            "SIMULAÇÃO EDUCATIVA SEM VALOR JURÍDICO\n"
            f"{'─'*70}\n"
        )
