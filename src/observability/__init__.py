"""
Observability V7 — OpenTelemetry + Prometheus
══════════════════════════════════════════════
Métricas expostas em /metrics (Prometheus scrape).
Traces enviados para Jaeger/Tempo via OTLP.
Tudo opcional — fallback gracioso se libs não instaladas.
"""
from __future__ import annotations

import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Optional

# Prometheus
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Info,
        start_http_server, REGISTRY,
    )
    PROM_OK = True
except ImportError:
    PROM_OK = False

# OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    OTEL_OK = True
except ImportError:
    OTEL_OK = False


@dataclass
class MetricasCaso:
    """Snapshot de métricas de um caso processado."""
    case_id: str
    instancia: str
    modelo: str
    backend: str
    rag_modo: str
    orquestracao: str
    duracao_total_s: float
    custo_usd: float
    n_entidades_anonimizadas: int
    grau_incerteza: str
    tem_tedh: bool
    tem_contraditorio: bool
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TribunalMetrics:
    """
    Singleton de métricas do Tribunal IA.
    Regista tudo: casos, latências, erros, RAG, custo.
    """
    _instance: Optional["TribunalMetrics"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "TribunalMetrics":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._inicializado = False
            return cls._instance

    def __init__(self) -> None:
        if self._inicializado:
            return
        self._inicializado = True
        self._metricas: list[MetricasCaso] = []
        self._erros: list[Dict[str, str]] = []

        if not PROM_OK:
            self._prom_activo = False
            return

        self._prom_activo = True

        # Contadores
        self.casos_total = Counter(
            "tribunal_casos_total",
            "Total de casos processados",
            ["instancia", "modelo", "backend"],
        )
        self.erros_total = Counter(
            "tribunal_erros_total",
            "Total de erros",
            ["tipo", "agente"],
        )
        self.cache_hits = Counter(
            "tribunal_cache_hits_total",
            "Cache hits no LLM",
        )
        self.rag_pesquisas = Counter(
            "tribunal_rag_pesquisas_total",
            "Pesquisas RAG efectuadas",
            ["modo"],
        )

        # Histogramas
        self.duracao_caso = Histogram(
            "tribunal_duracao_caso_segundos",
            "Duração total de um caso",
            ["instancia", "orquestracao"],
            buckets=[30, 60, 120, 180, 300, 600],
        )
        self.duracao_agente = Histogram(
            "tribunal_duracao_agente_segundos",
            "Duração de cada agente",
            ["agente"],
            buckets=[5, 10, 20, 40, 80, 160],
        )
        self.rag_top_score = Histogram(
            "tribunal_rag_top_score",
            "Top score dos resultados RAG",
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        # Gauges
        self.custo_acumulado_usd = Gauge(
            "tribunal_custo_acumulado_usd",
            "Custo total acumulado em USD",
        )
        self.casos_em_curso = Gauge(
            "tribunal_casos_em_curso",
            "Casos actualmente a ser processados",
        )
        self.rag_fragmentos = Gauge(
            "tribunal_rag_fragmentos_total",
            "Total de fragmentos indexados no RAG",
            ["modo"],
        )

        # Info
        self.info = Info(
            "tribunal_ia",
            "Informação do sistema Tribunal IA",
        )

    def registar_info(self, versao: str, modelo: str, rag_modo: str) -> None:
        if self._prom_activo:
            self.info.info({
                "versao": versao,
                "modelo": modelo,
                "rag_modo": rag_modo,
            })

    def registar_caso(self, m: MetricasCaso) -> None:
        self._metricas.append(m)
        if not self._prom_activo:
            return
        self.casos_total.labels(
            instancia=m.instancia, modelo=m.modelo, backend=m.backend
        ).inc()
        self.duracao_caso.labels(
            instancia=m.instancia, orquestracao=m.orquestracao
        ).observe(m.duracao_total_s)
        self.custo_acumulado_usd.inc(m.custo_usd)

    def registar_erro(self, tipo: str, agente: str, detalhe: str) -> None:
        self._erros.append({"tipo": tipo, "agente": agente, "detalhe": detalhe[:200]})
        if self._prom_activo:
            self.erros_total.labels(tipo=tipo, agente=agente).inc()

    def registar_cache_hit(self) -> None:
        if self._prom_activo:
            self.cache_hits.inc()

    def registar_rag(self, modo: str, top_score: float) -> None:
        if self._prom_activo:
            self.rag_pesquisas.labels(modo=modo).inc()
            self.rag_top_score.observe(top_score)

    def actualizar_rag_fragmentos(self, n: int, modo: str) -> None:
        if self._prom_activo:
            self.rag_fragmentos.labels(modo=modo).set(n)

    @contextmanager
    def medir_agente(self, nome: str) -> Generator[None, None, None]:
        start = time.time()
        if self._prom_activo:
            self.casos_em_curso.inc()
        try:
            yield
        finally:
            elapsed = time.time() - start
            if self._prom_activo:
                self.duracao_agente.labels(agente=nome).observe(elapsed)
                self.casos_em_curso.dec()

    def resumo(self) -> Dict[str, Any]:
        total = len(self._metricas)
        if not total:
            return {"total_casos": 0}
        custo = sum(m.custo_usd for m in self._metricas)
        dur_media = sum(m.duracao_total_s for m in self._metricas) / total
        por_instancia: Dict[str, int] = {}
        por_incerteza: Dict[str, int] = {}
        for m in self._metricas:
            por_instancia[m.instancia] = por_instancia.get(m.instancia, 0) + 1
            por_incerteza[m.grau_incerteza] = por_incerteza.get(m.grau_incerteza, 0) + 1
        return {
            "total_casos": total,
            "custo_total_usd": round(custo, 6),
            "duracao_media_s": round(dur_media, 1),
            "total_erros": len(self._erros),
            "por_instancia": por_instancia,
            "por_incerteza": por_incerteza,
            "prometheus_activo": self._prom_activo,
        }


# ── OpenTelemetry setup ───────────────────────────────────────────────
_tracer: Optional[Any] = None


def setup_tracing(service_name: str = "tribunal-ia-v7", endpoint: str = "") -> None:
    """Configura OpenTelemetry tracing. Silencioso se OTEL não instalado."""
    global _tracer
    if not OTEL_OK:
        return
    try:
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        if endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
                provider.add_span_processor(BatchSpanProcessor(exporter))
            except Exception:
                pass  # OTLP exporter opcional
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(service_name)
    except Exception:
        pass


def get_tracer() -> Optional[Any]:
    return _tracer


@contextmanager
def span(nome: str, **atributos: str) -> Generator[Any, None, None]:
    """Context manager para um span OpenTelemetry. Gracioso se OTEL não disponível."""
    if _tracer is None or not OTEL_OK:
        yield None
        return
    with _tracer.start_as_current_span(nome) as s:
        for k, v in atributos.items():
            s.set_attribute(k, v)
        yield s


def iniciar_servidor_metricas(port: int = 9090) -> None:
    """Inicia servidor HTTP Prometheus em segundo plano."""
    if not PROM_OK:
        return
    try:
        start_http_server(port)
        print(f"[Observability] Prometheus métricas em ::{port}/metrics")
    except Exception as e:
        print(f"[Observability] Prometheus falhou: {e}")


# Singleton global
metrics = TribunalMetrics()
