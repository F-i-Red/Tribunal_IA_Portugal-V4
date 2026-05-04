"""
CaseProcessor V5 — Orquestrador principal
─────────────────────────────────────────
Fluxo:
1. Anonimização RGPD
2. RAG (contexto jurídico com metadata filtering)
3. Instrução (perguntas específicas ao caso)
4. Detetive → Acusação → Defesa (sequencial — dependências)
5. 3 Juízes (paralelo em pagos, sequencial em free/Ollama)
6. Consistência + Incerteza
7. Ata TXT + PDF
8. Histórico
"""
from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..utils import get_config, get_logger, anonymize_text
from ..utils.brain import get_brain
from ..agents import (
    DetetiveAgent, AcusacaoAgent, DefesaAgent,
    JuizAgent, InstrucaoAgent, ConsistenciaAgent,
)
from ..rag import MotorRAG, ValidadorCitacoes
from ..export import exportar_pdf
from ..historico import get_historico, criar_registo
from .instancias import INSTANCIAS, InstanciaJudicial, detectar_instancia_por_keywords


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
    ata_final: Optional[str] = None
    ata_path: Optional[Path] = None
    pdf_bytes: Optional[bytes] = None
    contexto_rag: Optional[str] = None
    validacao_citacoes: Optional[str] = None
    custo_total_usd: float = 0.0
    doc_hash: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class CaseProcessor:
    def __init__(self) -> None:
        self.config = get_config()
        self.logger = get_logger()
        self.brain = get_brain()

        # RAG
        self.rag = MotorRAG(
            Path("."),
            modo=self.config.rag_modo,
            embedding_modelo=self.config.rag_embedding_modelo,
        )
        n = self.rag.indexar()
        if n > 0:
            self.logger.info(f"RAG indexado: {n} fragmentos (modo={self.config.rag_modo})")

        self.validador = ValidadorCitacoes(Path("data/leis"))

        # Agentes
        self._detetive = DetetiveAgent(self.brain, self.logger)
        self._acusacao = AcusacaoAgent(self.brain, self.logger)
        self._defesa   = DefesaAgent(self.brain, self.logger)
        self._instrucao = InstrucaoAgent(self.brain, self.logger)
        self._consistencia = ConsistenciaAgent(self.brain, self.logger)

    # ── RAG contextualizado por instância ────────────────────────────
    def _rag_ctx(
        self, query: str, instancia: Optional[str] = None, n: int = 6
    ) -> str:
        if not self.rag.tem_dados():
            return ""
        frags = self.rag.pesquisar(query, n_resultados=n, instancia=instancia)
        if frags:
            self.logger.log_rag(query, len(frags), frags[0].relevancia)
        return self.rag.formatar_contexto(frags, max_chars=3000)

    # ── Instrução pública ─────────────────────────────────────────────
    def gerar_perguntas_instrucao(
        self, case_description: str, instancia_codigo: str = "TIC"
    ) -> Dict:
        inst = INSTANCIAS.get(instancia_codigo, INSTANCIAS["TIC"])
        ctx_rag = self._rag_ctx(case_description, instancia=instancia_codigo, n=3)
        # Propaga excepção — sem fallback genérico
        return self._instrucao.executar(case_description, inst, ctx_rag)

    # ── Formatar esclarecimentos ──────────────────────────────────────
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
            desc = m.get("descricao", "")
            if desc:
                linhas.append(f"📎 Material: {desc}")
        linhas.append("════════════════════════════════════\n")
        return "\n".join(linhas)

    # ── Sentenças em paralelo (só modelos pagos) ──────────────────────
    def _sentencas_paralelo(
        self,
        case_text: str,
        detetive: str,
        acusacao: str,
        defesa: str,
        inst: InstanciaJudicial,
        ctx_rag: str,
    ) -> tuple[str, str, str]:
        resultados: Dict[str, str] = {}
        erros: Dict[str, str] = {}

        def _run(perfil: str) -> None:
            try:
                resultados[perfil] = JuizAgent(self.brain, self.logger, perfil).executar(
                    case_text, detetive, acusacao, defesa, inst, ctx_rag
                )
            except Exception as e:
                erros[perfil] = str(e)
                resultados[perfil] = f"[SENTENÇA {perfil.upper()}: erro — {str(e)[:200]}]"

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

    # ── Processo principal ────────────────────────────────────────────
    def process(
        self,
        case_description: str,
        instancia_codigo: Optional[str] = None,
        dados_instrucao: Optional[Dict] = None,
        gerar_pdf: bool = True,
        pdf_docs_extraidos: Optional[List[str]] = None,
    ) -> CaseResult:

        if not instancia_codigo:
            instancia_codigo = detectar_instancia_por_keywords(case_description)
        inst = INSTANCIAS.get(instancia_codigo, INSTANCIAS["TIC"])

        trace_id = self.logger.start_case(case_description)
        case_id = f"caso_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"Iniciando {case_id} | {inst.nome} | {self.config.modelo_activo}")

        # 1. Anonimizar
        anon_text, entities = anonymize_text(case_description)
        self.logger.log_anonymization(len(entities), list({e.label for e in entities}))

        # Integrar documentos PDF extraídos no texto do caso
        if pdf_docs_extraidos:
            anon_text += "\n\n=== DOCUMENTOS SUBMETIDOS ===\n" + "\n\n".join(pdf_docs_extraidos)

        # 2. RAG com metadata filtering por instância
        ctx_instrucao = self._fmt_instrucao(dados_instrucao)
        ctx_rag = self._rag_ctx(
            anon_text + " " + ctx_instrucao,
            instancia=instancia_codigo,
            n=6,
        )

        # 3. Agentes sequenciais
        detetive = self._detetive.executar(anon_text, ctx_instrucao, ctx_rag, inst)
        acusacao = self._acusacao.executar(anon_text, detetive, ctx_rag, inst)
        defesa   = self._defesa.executar(anon_text, detetive, acusacao, ctx_rag, inst)

        # 4. Sentenças (paralelo só se pago e paralelismo activado)
        usar_paralelo = (
            self.config.paralelismo
            and not self.config.is_free_model
            and not self.config.usar_ollama
        )

        if usar_paralelo:
            s_rigorosa, s_garantista, s_equilibrada = self._sentencas_paralelo(
                anon_text, detetive, acusacao, defesa, inst, ctx_rag
            )
        else:
            s_rigorosa   = JuizAgent(self.brain, self.logger, "rigoroso").executar(
                anon_text, detetive, acusacao, defesa, inst, ctx_rag)
            s_garantista = JuizAgent(self.brain, self.logger, "garantista").executar(
                anon_text, detetive, acusacao, defesa, inst, ctx_rag)
            s_equilibrada = JuizAgent(self.brain, self.logger, "equilibrado").executar(
                anon_text, detetive, acusacao, defesa, inst, ctx_rag)

        # 5. Consistência + Incerteza
        relatorio_consistencia = None
        grau_incerteza = "N/A"
        if self.config.consistencia_check:
            try:
                relatorio_consistencia = self._consistencia.executar(
                    inst, s_rigorosa, s_garantista, s_equilibrada
                )
                # Extrair grau de incerteza
                m = re.search(
                    r"(?:GRAU DE INCERTEZA GLOBAL|Grau Global)[:\s]*(Baixo|Médio|Alto|Muito Alto)",
                    relatorio_consistencia, re.IGNORECASE,
                )
                if m:
                    grau_incerteza = m.group(1)
                self.logger.log_consistencia(0.0, 0)
            except Exception as e:
                self.logger.warning(f"Consistência falhou: {e}")

        # 6. Validação de citações
        _, problemas = self.validador.validar_texto(
            " ".join(filter(None, [acusacao, defesa, s_rigorosa, s_garantista, s_equilibrada]))
        )
        validacao = self.validador.relatorio_citacoes(problemas)

        # 7. Montar ata TXT
        cost_stats = self.brain.get_cost_stats()
        ata = self._montar_ata(
            case_id, trace_id, anon_text, inst,
            detetive, acusacao, defesa,
            s_rigorosa, s_garantista, s_equilibrada,
            relatorio_consistencia, dados_instrucao, ctx_rag, validacao,
            cost_stats,
        )
        doc_hash = hashlib.sha256(ata.encode()).hexdigest()[:16]
        ata_final = self._disclaimer(doc_hash, case_id) + ata + self._watermark(doc_hash, case_id, trace_id)

        ata_path: Optional[Path] = None
        if self.config.guardar_atas:
            ata_path = self.config.pasta_atas / f"{case_id}.txt"
            ata_path.write_text(ata_final, encoding="utf-8")

        # Criar resultado parcial para PDF e histórico
        result = CaseResult(
            case_id=case_id,
            trace_id=trace_id,
            original_description=case_description,
            anonymized_description=anon_text,
            entities_found=[{"text": e.text, "type": e.label} for e in entities],
            instancia_codigo=instancia_codigo,
            instancia_nome=inst.nome,
            modelo_usado=self.config.modelo_activo,
            backend_usado=self.config.backend,
            dados_instrucao=dados_instrucao,
            detetive_report=detetive,
            acusacao=acusacao,
            defesa=defesa,
            sentenca_rigorosa=s_rigorosa,
            sentenca_garantista=s_garantista,
            sentenca_equilibrada=s_equilibrada,
            relatorio_consistencia=relatorio_consistencia,
            grau_incerteza=grau_incerteza,
            ata_final=ata_final,
            ata_path=ata_path,
            contexto_rag=ctx_rag,
            validacao_citacoes=validacao,
            custo_total_usd=cost_stats["total_cost_usd"],
            doc_hash=doc_hash,
        )

        # 8. Exportar PDF
        if gerar_pdf and self.config.exportar_pdf:
            try:
                pdf_path = self.config.pasta_atas / f"{case_id}.pdf"
                result.pdf_bytes = exportar_pdf(result, destino=pdf_path)
            except Exception as e:
                self.logger.warning(f"PDF falhou: {e}")

        # 9. Histórico
        if self.config.historico_enabled:
            try:
                hist = get_historico()
                hist.adicionar(criar_registo(result, grau_incerteza))
            except Exception as e:
                self.logger.warning(f"Histórico falhou: {e}")

        return result

    # ── Ata TXT ───────────────────────────────────────────────────────
    def _montar_ata(
        self,
        case_id, trace_id, case_text, inst: InstanciaJudicial,
        detetive, acusacao, defesa,
        rigorosa, garantista, equilibrada,
        consistencia, dados_instrucao, ctx_rag, validacao,
        cost_stats,
    ) -> str:
        now = datetime.now(timezone.utc)
        meses = ["janeiro","fevereiro","março","abril","maio","junho",
                 "julho","agosto","setembro","outubro","novembro","dezembro"]
        data_pt = f"{now.day} de {meses[now.month-1]} de {now.year}, {now.strftime('%H:%M')} UTC"
        sep = "═" * 72

        sec_instrucao = ""
        if dados_instrucao and dados_instrucao.get("respostas"):
            resp_validas = [
                (k, v) for k, v in dados_instrucao["respostas"].items()
                if v.get("resposta", "") not in ("", "Sem resposta")
            ]
            if resp_validas:
                sec_instrucao = f"\n{sep}\nSECÇÃO III — ESCLARECIMENTOS DE INSTRUÇÃO\n{sep}\n\n"
                for _, item in resp_validas:
                    sec_instrucao += f" [{item.get('categoria','?')}] {item.get('pergunta','')}\n"
                    sec_instrucao += f" ➜ {item.get('resposta','')}\n\n"

        sec_consistencia = ""
        if consistencia:
            sec_consistencia = f"\n{sep}\nSECÇÃO IX — CONSISTÊNCIA E INCERTEZA\n{sep}\n\n{consistencia}\n"

        sec_rag = ""
        if ctx_rag:
            sec_rag = f"\n{sep}\nSECÇÃO XI — FUNDAMENTOS JURÍDICOS (RAG)\n{sep}\n\n{ctx_rag[:1500]}\n"

        custo_str = "Gratuito (modelo free / local)" if cost_stats["total_cost_usd"] == 0 else \
                    f"${cost_stats['total_cost_usd']:.4f} USD"

        return f"""{sep}
ATA DE SIMULAÇÃO JUDICIAL — TRIBUNAL IA PORTUGAL V5
{sep}

PROCESSO Nº  : {case_id}
TRACE ID     : {trace_id}
TRIBUNAL     : {inst.nome}
MATÉRIA      : {inst.materia}
DIPLOMA      : {inst.diploma_principal}
DATA         : {data_pt}
MODELO       : {cost_stats['modelo']} [{cost_stats['backend']}]
CUSTO        : {custo_str}
ESTADO       : SIMULAÇÃO EDUCATIVA — SEM VALOR JURÍDICO

{sep}
SECÇÃO I — DESCRIÇÃO DO CASO (ANONIMIZADO — RGPD)
{sep}

{case_text}

{sep}
SECÇÃO II — RELATÓRIO DE INSTRUÇÃO FACTUAL
{sep}

{detetive}
{sec_instrucao}
{sep}
SECÇÃO IV — ALEGAÇÕES DA ACUSAÇÃO / MP
{sep}

{acusacao}

{sep}
SECÇÃO V — ALEGAÇÕES DA DEFESA
{sep}

{defesa}

{sep}
SECÇÃO VI — {inst.termo_decisao.upper()}: PERFIL RIGOROSO
{sep}

{rigorosa}

{sep}
SECÇÃO VII — {inst.termo_decisao.upper()}: PERFIL GARANTISTA
{sep}

{garantista}

{sep}
SECÇÃO VIII — {inst.termo_decisao.upper()}: PERFIL EQUILIBRADO
{sep}

{equilibrada}
{sec_consistencia}
{sep}
SECÇÃO X — VALIDAÇÃO DE CITAÇÕES JURÍDICAS
{sep}

{validacao}
{sec_rag}
{sep}
SECÇÃO XII — NOTA EDUCATIVA
{sep}

Esta simulação ilustra como o mesmo caso produz decisões diferentes
consoante o perfil decisório do julgador e a força probatória disponível.

 • Prevenção geral / rigor punitivo  → Perfil Rigoroso
 • Garantias processuais / in dubio  → Perfil Garantista
 • Proporcionalidade / equidade      → Perfil Equilibrado

O Relatório de Consistência e Incerteza indica a robustez jurídica
global da análise e os pontos factuais mais frágeis.

Diploma aplicável: {inst.diploma_principal}
Para situações reais: Ordem dos Advogados — www.oa.pt

"""

    def _disclaimer(self, hash_doc: str, case_id: str) -> str:
        l = "═" * 70
        return (
            f"\n╔{l}╗\n"
            "║  ⚠️  AVISO LEGAL — DOCUMENTO DE SIMULAÇÃO EDUCATIVA            ║\n"
            "║                                                                  ║\n"
            "║  Gerado por IA. NÃO constitui parecer jurídico nem decisão      ║\n"
            "║  judicial. Para situações reais: Advogado (www.oa.pt)           ║\n"
            "║                                                                  ║\n"
            f"║  Hash: {hash_doc:<60}║\n"
            f"║  ID:   {case_id:<60}║\n"
            f"╚{l}╝\n\n"
        )

    def _watermark(self, hash_doc: str, case_id: str, trace_id: str) -> str:
        return (
            f"\n{'─'*70}\n"
            "WATERMARK — TRIBUNAL IA PORTUGAL V5\n"
            f"Hash: {hash_doc} | ID: {case_id} | Trace: {trace_id}\n"
            "DOCUMENTO DE SIMULAÇÃO EDUCATIVA SEM VALOR JURÍDICO\n"
            f"{'─'*70}\n"
        )
