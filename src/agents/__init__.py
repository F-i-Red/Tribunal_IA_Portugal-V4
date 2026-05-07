"""
Agentes jurídicos V6.
Inclui: Detetive, Acusação, Defesa, Juiz×3,
        Instrução, Consistência, PDF, TEDH, Contraditório.
"""
from __future__ import annotations

import json
import re
from typing import Dict, Optional

from ..utils.brain import TribunalBrain
from ..utils.logger import TribunalLogger
from ..prompts import Prompts
from ..pipeline.instancias import InstanciaJudicial


class BaseAgent:
    nome: str = "base"

    def __init__(self, brain: TribunalBrain, logger: TribunalLogger) -> None:
        self.brain = brain
        self.logger = logger

    def _call(
        self,
        user_content: str,
        system_prompt: str,
        temperature: float = 0.15,
        max_tokens: int = 1600,
    ) -> str:
        self.logger.set_agent(self.nome)
        try:
            resp = self.brain.call(
                messages=[{"role": "user", "content": user_content}],
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.content.strip()
            return content if content else f"[{self.nome.upper()}: resposta vazia]"
        except Exception as e:
            self.logger.error(f"Agente {self.nome}: {e}")
            raise


class DetetiveAgent(BaseAgent):
    nome = "detetive"

    def executar(
        self, case_text: str, ctx_instrucao: str, ctx_rag: str, inst: InstanciaJudicial
    ) -> str:
        user = f"CASO:\n{case_text}{ctx_instrucao}"
        return self._call(user, Prompts.detetive(inst, ctx_rag),
                         temperature=0.1, max_tokens=1600)


class AcusacaoAgent(BaseAgent):
    nome = "acusacao"

    def executar(
        self, case_text: str, detetive: str, ctx_rag: str, inst: InstanciaJudicial
    ) -> str:
        user = f"CASO:\n{case_text}\n\nRELATÓRIO DE INSTRUÇÃO:\n{detetive[:1000]}"
        return self._call(user, Prompts.acusacao(inst, ctx_rag),
                         temperature=0.15, max_tokens=1400)


class DefesaAgent(BaseAgent):
    nome = "defesa"

    def executar(
        self, case_text: str, detetive: str, acusacao: str,
        ctx_rag: str, inst: InstanciaJudicial,
        intervencao_utilizador: Optional[str] = None,
    ) -> str:
        user = (
            f"CASO:\n{case_text}\n\n"
            f"INSTRUÇÃO:\n{detetive[:700]}\n\n"
            f"ACUSAÇÃO:\n{acusacao[:800]}"
        )
        if intervencao_utilizador:
            system = Prompts.defesa_contraditorio(inst, ctx_rag, intervencao_utilizador)
        else:
            system = Prompts.defesa(inst, ctx_rag)
        return self._call(user, system, temperature=0.15, max_tokens=1400)


class JuizAgent(BaseAgent):
    def __init__(
        self, brain: TribunalBrain, logger: TribunalLogger, perfil: str
    ) -> None:
        super().__init__(brain, logger)
        self.perfil = perfil
        self.nome = f"juiz_{perfil}"

    def executar(
        self, case_text: str, detetive: str, acusacao: str,
        defesa: str, inst: InstanciaJudicial, ctx_rag: str,
    ) -> str:
        user = (
            f"CASO:\n{case_text[:900]}\n\n"
            f"INSTRUÇÃO:\n{detetive[:600]}\n\n"
            f"ACUSAÇÃO:\n{acusacao[:500]}\n\n"
            f"DEFESA:\n{defesa[:500]}"
        )
        return self._call(
            user, Prompts.juiz(inst, self.perfil, ctx_rag),
            temperature=0.05, max_tokens=1800,
        )


class ConsistenciaAgent(BaseAgent):
    nome = "consistencia"

    def executar(
        self, inst: InstanciaJudicial,
        s_rigorosa: str, s_garantista: str, s_equilibrada: str,
    ) -> str:
        system = Prompts.consistencia(inst, s_rigorosa, s_garantista, s_equilibrada)
        return self._call(
            "Produz o relatório de consistência e incerteza.",
            system, temperature=0.1, max_tokens=1200,
        )


class TEDHAgent(BaseAgent):
    """Compara o caso com jurisprudência do TEDH/ECHR."""
    nome = "tedh"

    def executar(
        self, inst: InstanciaJudicial, caso_pt: str,
        ctx_tedh: str, lingua: str = "pt",
    ) -> str:
        system = Prompts.analise_tedh(inst, caso_pt, ctx_tedh, lingua)
        user_msg = (
            "Analyse this case in light of ECtHR jurisprudence."
            if lingua == "en"
            else "Analisa este caso à luz da jurisprudência do TEDH."
        )
        return self._call(user_msg, system, temperature=0.1, max_tokens=1200)


class ContraditórioFeedbackAgent(BaseAgent):
    """Avalia o argumento do utilizador no modo contraditório."""
    nome = "contraditorio_feedback"

    def executar(
        self, inst: InstanciaJudicial, argumento: str,
        acusacao: str, detetive: str,
    ) -> str:
        system = Prompts.contraditorio_feedback(inst, argumento, acusacao, detetive)
        return self._call(
            f"Argumento do advogado de defesa:\n{argumento}",
            system, temperature=0.1, max_tokens=900,
        )


class InstrucaoAgent(BaseAgent):
    nome = "instrucao"

    def executar(
        self, case_text: str, inst: InstanciaJudicial, ctx_rag: str,
    ) -> Dict:
        system = Prompts.instrucao(inst, ctx_rag)
        raw = self._call(
            f"Caso para instrução:\n\n{case_text}",
            system, temperature=0.1, max_tokens=1200,
        )
        return self._parse_json(raw)

    def _parse_json(self, raw: str) -> Dict:
        t = raw.strip()
        t = re.sub(r"```(?:json)?\s*", "", t)
        t = re.sub(r"```", "", t).strip()
        for extrator in (lambda s: s, lambda s: s[s.find("{"):s.rfind("}") + 1]):
            try:
                parsed = json.loads(extrator(t))
                if "perguntas" in parsed and len(parsed["perguntas"]) > 0:
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        raise ValueError(
            f"Modelo não devolveu JSON válido. Resposta (200 chars): {raw[:200]}"
        )


class PDFExtractorAgent(BaseAgent):
    nome = "pdf_extractor"

    def executar(self, conteudo_pdf: str, tipo_doc: str = "documento jurídico") -> str:
        return self._call(
            f"Extrai a informação:\n\n{conteudo_pdf[:4000]}",
            Prompts.pdf_extraction(conteudo_pdf, tipo_doc),
            temperature=0.05, max_tokens=1000,
        )
