"""Testes do módulo de observability V7."""
import pytest
import time
from src.observability import TribunalMetrics, MetricasCaso, metrics, span


def _caso(i: int = 1) -> MetricasCaso:
    return MetricasCaso(
        case_id=f"caso_obs_{i:04d}",
        instancia="TIC", modelo="openrouter/free",
        backend="openrouter", rag_modo="bm25",
        orquestracao="imperativo", duracao_total_s=45.3,
        custo_usd=0.0, n_entidades_anonimizadas=3,
        grau_incerteza="Médio", tem_tedh=False, tem_contraditorio=False,
    )


class TestMetricas:
    def test_singleton(self):
        m1 = TribunalMetrics()
        m2 = TribunalMetrics()
        assert m1 is m2

    def test_registar_caso(self):
        m = TribunalMetrics()
        antes = m.resumo().get("total_casos", 0)
        m.registar_caso(_caso())
        depois = m.resumo()["total_casos"]
        assert depois == antes + 1

    def test_registar_erro(self):
        m = TribunalMetrics()
        antes = m.resumo().get("total_erros", 0)
        m.registar_erro("timeout", "juiz_rigoroso", "Tempo esgotado na chamada LLM")
        assert m.resumo()["total_erros"] == antes + 1

    def test_resumo_campos(self):
        m = TribunalMetrics()
        m.registar_caso(_caso(99))
        r = m.resumo()
        assert "total_casos" in r
        assert "custo_total_usd" in r
        assert "duracao_media_s" in r
        assert "por_instancia" in r
        assert "por_incerteza" in r

    def test_resumo_sem_casos(self):
        m = TribunalMetrics.__new__(TribunalMetrics)
        m._inicializado = False
        m.__init__()
        m._metricas = []
        m._erros = []
        r = m.resumo()
        assert r["total_casos"] == 0

    def test_medir_agente_context_manager(self):
        m = TribunalMetrics()
        with m.medir_agente("detetive"):
            time.sleep(0.01)
        # Não deve lançar excepção

    def test_custo_acumulado(self):
        m = TribunalMetrics()
        m._metricas = []
        caso_pago = MetricasCaso(
            case_id="caso_pago", instancia="TIC", modelo="gemini",
            backend="openrouter", rag_modo="hibrido",
            orquestracao="langgraph", duracao_total_s=30.0,
            custo_usd=0.0542, n_entidades_anonimizadas=2,
            grau_incerteza="Baixo", tem_tedh=True, tem_contraditorio=False,
        )
        m.registar_caso(caso_pago)
        r = m.resumo()
        assert r["custo_total_usd"] >= 0.0542

    def test_por_instancia_contagem(self):
        m = TribunalMetrics()
        m._metricas = []
        for i in range(3):
            m.registar_caso(_caso(i))
        c2 = MetricasCaso(
            case_id="trab_01", instancia="TRAB", modelo="x",
            backend="openrouter", rag_modo="bm25",
            orquestracao="imperativo", duracao_total_s=20.0,
            custo_usd=0.0, n_entidades_anonimizadas=1,
            grau_incerteza="Alto", tem_tedh=False, tem_contraditorio=True,
        )
        m.registar_caso(c2)
        r = m.resumo()
        assert r["por_instancia"].get("TIC", 0) >= 3
        assert r["por_instancia"].get("TRAB", 0) >= 1


class TestSpan:
    def test_span_sem_otel(self):
        """span() deve funcionar graciosamente sem OpenTelemetry."""
        with span("teste_agente", agente="detetive", case_id="c001") as s:
            pass  # Não deve lançar excepção

    def test_span_retorna_none_sem_tracer(self):
        from src.observability import _tracer
        if _tracer is None:
            with span("teste") as s:
                assert s is None


class TestSetupTracing:
    def test_setup_sem_otel_nao_falha(self):
        from src.observability import setup_tracing
        setup_tracing("tribunal-test", "http://nao-existe:4317")
        # Não deve lançar excepção

    def test_setup_com_endpoint_vazio(self):
        from src.observability import setup_tracing
        setup_tracing("tribunal-test", "")


class TestRegistarInfo:
    def test_registar_info_nao_falha(self):
        m = TribunalMetrics()
        m.registar_info("7.0.0", "openrouter/free", "hibrido")
