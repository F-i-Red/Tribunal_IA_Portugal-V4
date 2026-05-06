from .case_processor import CaseProcessor, CaseResult
from .instancias import (
    INSTANCIAS, InstanciaJudicial,
    listar_instancias_menu, detectar_instancia_por_keywords,
)
__all__ = [
    "CaseProcessor", "CaseResult",
    "INSTANCIAS", "InstanciaJudicial",
    "listar_instancias_menu", "detectar_instancia_por_keywords",
]
