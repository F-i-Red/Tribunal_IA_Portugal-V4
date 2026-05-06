"""
Histórico de Casos V5
─────────────────────
• Persistência em JSON por caso
• Índice de pesquisa por keywords + instância + data
• Thread-safe
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RegistoHistorico:
    id: str
    timestamp: str
    instancia_codigo: str
    instancia_nome: str
    resumo: str                    # primeiros 200 chars do caso anonimizado
    dispositivo: str               # extrato da decisão equilibrada
    grau_incerteza: str            # do relatório de consistência
    custo_usd: float
    modelo: str
    n_entidades_anonimizadas: int
    ata_path: Optional[str] = None


class HistoricoCasos:
    def __init__(self, pasta: Path) -> None:
        self.pasta = pasta
        self.pasta.mkdir(parents=True, exist_ok=True)
        self._indice_path = pasta / "indice.json"
        self._lock = threading.Lock()
        self._indice: List[RegistoHistorico] = []
        self._carregar()

    def _carregar(self) -> None:
        if self._indice_path.exists():
            try:
                dados = json.loads(self._indice_path.read_text(encoding="utf-8"))
                self._indice = [RegistoHistorico(**r) for r in dados]
            except Exception:
                self._indice = []

    def _guardar(self) -> None:
        try:
            dados = [asdict(r) for r in self._indice]
            self._indice_path.write_text(
                json.dumps(dados, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def adicionar(self, registo: RegistoHistorico) -> None:
        with self._lock:
            self._indice.insert(0, registo)  # mais recente primeiro
            if len(self._indice) > 500:       # limitar a 500 entradas
                self._indice = self._indice[:500]
            self._guardar()

    def pesquisar(
        self,
        query: str = "",
        instancia: Optional[str] = None,
        limite: int = 20,
    ) -> List[RegistoHistorico]:
        with self._lock:
            resultados = list(self._indice)

        if instancia:
            resultados = [r for r in resultados if r.instancia_codigo == instancia]

        if query.strip():
            q = query.lower()
            resultados = [
                r for r in resultados
                if q in r.resumo.lower()
                or q in r.dispositivo.lower()
                or q in r.instancia_nome.lower()
            ]

        return resultados[:limite]

    def total(self) -> int:
        with self._lock:
            return len(self._indice)

    def estatisticas(self) -> Dict:
        with self._lock:
            if not self._indice:
                return {"total": 0}
            por_instancia: Dict[str, int] = {}
            custo_total = 0.0
            for r in self._indice:
                por_instancia[r.instancia_codigo] = por_instancia.get(r.instancia_codigo, 0) + 1
                custo_total += r.custo_usd
            return {
                "total": len(self._indice),
                "por_instancia": por_instancia,
                "custo_total_usd": round(custo_total, 4),
                "primeiro": self._indice[-1].timestamp if self._indice else None,
                "ultimo": self._indice[0].timestamp if self._indice else None,
            }

    def limpar(self) -> None:
        with self._lock:
            self._indice = []
            self._guardar()


def criar_registo(result: "CaseResult", grau_incerteza: str = "N/A") -> RegistoHistorico:  # type: ignore[name-defined]
    import re
    # Extrair dispositivo da sentença equilibrada
    dispositivo = "Ver ata completa"
    if result.sentenca_equilibrada:
        m = re.search(
            r"(?:CONDENA|ABSOLVE|JULGA)[^.]*\.",
            result.sentenca_equilibrada, re.IGNORECASE
        )
        if m:
            dispositivo = m.group(0).strip()[:200]

    return RegistoHistorico(
        id=result.case_id,
        timestamp=result.timestamp,
        instancia_codigo=result.instancia_codigo,
        instancia_nome=result.instancia_nome,
        resumo=result.anonymized_description[:200],
        dispositivo=dispositivo,
        grau_incerteza=grau_incerteza,
        custo_usd=result.custo_total_usd,
        modelo=result.modelo_usado,
        n_entidades_anonimizadas=len(result.entities_found),
        ata_path=str(result.ata_path) if result.ata_path else None,
    )


_historico: Optional[HistoricoCasos] = None
_hist_lock = threading.Lock()


def get_historico() -> HistoricoCasos:
    global _historico
    with _hist_lock:
        if _historico is None:
            from ..utils.config import get_config
            cfg = get_config()
            _historico = HistoricoCasos(cfg.pasta_historico)
    return _historico
