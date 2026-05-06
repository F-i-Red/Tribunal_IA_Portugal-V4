"""
Modo Contraditório V6
══════════════════════
Permite ao utilizador intervir como advogado de defesa
após ver a acusação e antes das sentenças finais.

Fluxo:
  1. Sistema gera instrução + detetive + acusação normalmente
  2. Utilizador lê a acusação e escreve os seus argumentos
  3. DefesaAgent incorpora os argumentos do utilizador
  4. ContraditórioFeedbackAgent avalia o argumento juridicamente
  5. Pipeline continua com a defesa enriquecida
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


@dataclass
class SessaoContraditorio:
    """Estado de uma sessão de contraditório."""
    case_id: str
    instancia_codigo: str
    detetive: str
    acusacao: str
    intervencoes: List["IntervencaoDefesa"] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def adicionar_intervencao(
        self,
        argumento: str,
        feedback_juridico: str = "",
    ) -> "IntervencaoDefesa":
        iv = IntervencaoDefesa(
            numero=len(self.intervencoes) + 1,
            argumento=argumento,
            feedback_juridico=feedback_juridico,
        )
        self.intervencoes.append(iv)
        return iv

    def texto_completo_intervencoes(self) -> str:
        if not self.intervencoes:
            return ""
        partes = []
        for iv in self.intervencoes:
            partes.append(
                f"[Argumento {iv.numero}]\n{iv.argumento}"
            )
        return "\n\n".join(partes)


@dataclass
class IntervencaoDefesa:
    numero: int
    argumento: str
    feedback_juridico: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class GestorContraditorio:
    """
    Gere o fluxo do modo contraditório.
    Integra com o CaseProcessor para enriquecer a defesa
    com as intervenções do utilizador.
    """

    def __init__(self) -> None:
        self._sessoes: dict[str, SessaoContraditorio] = {}

    def iniciar_sessao(
        self,
        case_id: str,
        instancia_codigo: str,
        detetive: str,
        acusacao: str,
    ) -> SessaoContraditorio:
        sessao = SessaoContraditorio(
            case_id=case_id,
            instancia_codigo=instancia_codigo,
            detetive=detetive,
            acusacao=acusacao,
        )
        self._sessoes[case_id] = sessao
        return sessao

    def obter_sessao(self, case_id: str) -> Optional[SessaoContraditorio]:
        return self._sessoes.get(case_id)

    def submeter_argumento(
        self,
        case_id: str,
        argumento: str,
        avaliar: bool = True,
    ) -> IntervencaoDefesa:
        """
        Submete um argumento de defesa e opcionalmente avalia-o.
        Retorna a intervenção com feedback jurídico se avaliar=True.
        """
        sessao = self._sessoes.get(case_id)
        if not sessao:
            raise ValueError(f"Sessão {case_id} não encontrada.")

        feedback = ""
        if avaliar:
            try:
                from ..agents import ContraditórioFeedbackAgent
                from ..utils.brain import get_brain
                from ..utils.logger import get_logger
                from ..pipeline.instancias import INSTANCIAS

                inst = INSTANCIAS.get(sessao.instancia_codigo, INSTANCIAS["TIC"])
                agente = ContraditórioFeedbackAgent(get_brain(), get_logger())
                feedback = agente.executar(
                    inst, argumento, sessao.acusacao, sessao.detetive
                )
            except Exception as e:
                feedback = f"[Avaliação não disponível: {e}]"

        return sessao.adicionar_intervencao(argumento, feedback)

    def resumo_intervencoes(self, case_id: str) -> str:
        sessao = self._sessoes.get(case_id)
        if not sessao or not sessao.intervencoes:
            return "Sem intervenções registadas."
        linhas = [f"📋 {len(sessao.intervencoes)} intervenção(ões) do advogado de defesa:\n"]
        for iv in sessao.intervencoes:
            linhas.append(f"── Argumento {iv.numero} ──")
            linhas.append(iv.argumento[:300])
            if iv.feedback_juridico:
                linhas.append(f"\n💬 Avaliação jurídica:\n{iv.feedback_juridico[:400]}")
            linhas.append("")
        return "\n".join(linhas)


# Instância global
_gestor: Optional[GestorContraditorio] = None


def get_gestor_contraditorio() -> GestorContraditorio:
    global _gestor
    if _gestor is None:
        _gestor = GestorContraditorio()
    return _gestor
