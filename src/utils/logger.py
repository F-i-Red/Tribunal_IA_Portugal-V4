"""
Logger estruturado V5 — usa structlog quando disponível, stdlib como fallback.
Output JSON em ficheiro, output legível na consola.
"""
from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import structlog

    def _configurar_structlog(level: str) -> None:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, level.upper(), logging.INFO)
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(
                file=open("logs/tribunal.log", "a", encoding="utf-8")
            ),
        )

    STRUCTLOG_OK = True
except ImportError:
    STRUCTLOG_OK = False


class TribunalLogger:
    def __init__(self, level: str = "INFO") -> None:
        self._level = level
        self._trace_id: str = "no-trace"
        self._agent: str = "system"
        self._cost_log: List[Dict[str, Any]] = []

        Path("logs").mkdir(exist_ok=True)

        # Logger stdlib para ficheiro (sempre)
        self._file_logger = logging.getLogger("tribunal_ia")
        self._file_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        if not self._file_logger.handlers:
            fh = logging.FileHandler("logs/tribunal.log", encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(message)s"))
            self._file_logger.addHandler(fh)
            self._file_logger.propagate = False

        # structlog se disponível
        if STRUCTLOG_OK:
            _configurar_structlog(level)
            self._sl = structlog.get_logger()
        else:
            self._sl = None

    def _emit(self, level: str, msg: str, **kw: Any) -> None:
        record: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "lvl": level,
            "agent": self._agent,
            "trace": self._trace_id,
            "msg": msg,
            **kw,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        getattr(self._file_logger, level.lower(), self._file_logger.info)(line)

    def info(self, msg: str, **kw: Any) -> None:
        self._emit("INFO", msg, **kw)

    def warning(self, msg: str, **kw: Any) -> None:
        self._emit("WARNING", msg, **kw)

    def error(self, msg: str, **kw: Any) -> None:
        self._emit("ERROR", msg, **kw)

    def start_case(self, description: str) -> str:
        self._trace_id = uuid.uuid4().hex[:10]
        self._cost_log = []
        self._emit("INFO", "caso_iniciado", desc_len=len(description))
        return self._trace_id

    def set_agent(self, agent: str) -> None:
        self._agent = agent

    def log_anonymization(self, n: int, tipos: List[str]) -> None:
        self._emit("INFO", "anonimizacao", entidades=n, tipos=tipos)

    def log_api_call(self, model: str, tin: int, tout: int, ms: float) -> None:
        entry = {"model": model, "tin": tin, "tout": tout, "ms": round(ms, 1)}
        self._cost_log.append(entry)
        self._emit("INFO", "api_call", **entry)

    def log_rag(self, query: str, n_resultados: int, top_score: float) -> None:
        self._emit("INFO", "rag_pesquisa",
                   query_len=len(query), n=n_resultados, top=round(top_score, 3))

    def log_consistencia(self, score: float, divergencias: int) -> None:
        self._emit("INFO", "consistencia", score=round(score, 3), divergencias=divergencias)

    def get_case_cost(self) -> Dict[str, Any]:
        return {
            "calls": len(self._cost_log),
            "models": list({e["model"] for e in self._cost_log}),
        }


_logger: Optional[TribunalLogger] = None


def get_logger() -> TribunalLogger:
    global _logger
    if _logger is None:
        try:
            from .config import get_config
            cfg = get_config()
            _logger = TribunalLogger(cfg.log_level)
        except Exception:
            _logger = TribunalLogger("INFO")
    return _logger
