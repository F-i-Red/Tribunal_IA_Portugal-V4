"""
Agentes jurídicos V5.
Inclui: Detetive, Acusação, Defesa, Juiz (3 perfis),
        Instrução, Consistência/Incerteza, PDF Extractor.
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
            if not content:
                return f"[{self.nome.upper()}: resposta vazia do modelo]"
            return content
        except Exception as e:
            self.logger.error(f"Agente {self.nome}: {e}")
            raise


class DetetiveAgent(BaseAgent):
    nome = "detetive"

    def executar(
        self, case_text: str, ctx_instrucao: str, ctx_rag: str, inst: InstanciaJudicial
    ) -> str:
        user = f"CASO:\n{case_text}{ctx_instrucao}"
        return self._call(user, Prompts.detetive(inst, ctx_rag), temperature=0.1, max_tokens=1600)


class AcusacaoAgent(BaseAgent):
    nome = "acusacao"

    def executar(
        self, case_text: str, detetive: str, ctx_rag: str, inst: InstanciaJudicial
    ) -> str:
        user = f"CASO:\n{case_text}\n\nRELATÓRIO DE INSTRUÇÃO:\n{detetive[:1000]}"
        return self._call(user, Prompts.acusacao(inst, ctx_rag), temperature=0.15, max_tokens=1400)


class DefesaAgent(BaseAgent):
    nome = "defesa"

    def executar(
        self, case_text: str, detetive: str, acusacao: str,
        ctx_rag: str, inst: InstanciaJudicial
    ) -> str:
        user = (
            f"CASO:\n{case_text}\n\n"
            f"INSTRUÇÃO (resumo):\n{detetive[:700]}\n\n"
            f"ACUSAÇÃO:\n{acusacao[:800]}"
        )
        return self._call(user, Prompts.defesa(inst, ctx_rag), temperature=0.15, max_tokens=1400)


class JuizAgent(BaseAgent):
    def __init__(
        self, brain: TribunalBrain, logger: TribunalLogger, perfil: str
    ) -> None:
        super().__init__(brain, logger)
        self.perfil = perfil
        self.nome = f"juiz_{perfil}"

    def executar(
        self, case_text: str, detetive: str, acusacao: str,
        defesa: str, inst: InstanciaJudicial, ctx_rag: str
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
    """Analisa consistência e incerteza entre as 3 sentenças."""
    nome = "consistencia"

    def executar(
        self,
        inst: InstanciaJudicial,
        s_rigorosa: str,
        s_garantista: str,
        s_equilibrada: str,
    ) -> str:
        system = Prompts.consistencia(inst, s_rigorosa, s_garantista, s_equilibrada)
        return self._call(
            "Produz o relatório de consistência e incerteza solicitado.",
            system,
            temperature=0.1,
            max_tokens=1200,
        )


class InstrucaoAgent(BaseAgent):
    nome = "instrucao"

    def executar(
        self, case_text: str, inst: InstanciaJudicial, ctx_rag: str
    ) -> Dict:
        system = Prompts.instrucao(inst, ctx_rag)
        raw = self._call(
            f"Caso para instrução:\n\n{case_text}",
            system, temperature=0.1, max_tokens=1200,
        )
        return self._parse_json(raw, inst)

    def _parse_json(self, raw: str, inst: InstanciaJudicial) -> Dict:
        t = raw.strip()
        # Remover markdown fences
        t = re.sub(r"```(?:json)?\s*", "", t)
        t = re.sub(r"```", "", t).strip()
        # Tentar parse directo e depois por extracção
        for extrator in (
            lambda s: s,
            lambda s: s[s.find("{"):s.rfind("}") + 1],
        ):
            try:
                parsed = json.loads(extrator(t))
                if "perguntas" in parsed and len(parsed["perguntas"]) > 0:
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass

        # Sem fallback genérico — propagar erro para a UI decidir
        raise ValueError(
            f"O modelo não devolveu JSON válido com perguntas para este caso. "
            f"Resposta (200 chars): {raw[:200]}"
        )


class PDFExtractorAgent(BaseAgent):
    """Extrai e estrutura informação de documentos PDF para o processo."""
    nome = "pdf_extractor"

    def executar(self, conteudo_pdf: str, tipo_doc: str = "documento jurídico") -> str:
        system = Prompts.pdf_extraction(conteudo_pdf, tipo_doc)
        return self._call(
            f"Extrai a informação relevante do seguinte documento:\n\n{conteudo_pdf[:4000]}",
            system,
            temperature=0.05,
            max_tokens=1000,
        )
