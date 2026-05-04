"""Testes do histórico de casos V5."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.historico import HistoricoCasos, RegistoHistorico
import time


@pytest.fixture
def hist(tmp_path):
    return HistoricoCasos(tmp_path / "historico")


def _registo(i: int = 1) -> RegistoHistorico:
    return RegistoHistorico(
        id=f"caso_test_{i:04d}",
        timestamp=f"2025-01-{i:02d}T10:00:00+00:00",
        instancia_codigo="TIC",
        instancia_nome="Tribunal de Instrução Criminal",
        resumo=f"Caso de furto qualificado número {i}",
        dispositivo=f"O Tribunal CONDENA o arguido {i} a 2 anos de prisão suspensa.",
        grau_incerteza="Médio",
        custo_usd=0.0,
        modelo="meta-llama/llama-3.3-70b-instruct:free",
        n_entidades_anonimizadas=3,
    )


def test_adicionar_e_recuperar(hist):
    r = _registo(1)
    hist.adicionar(r)
    assert hist.total() == 1
    resultados = hist.pesquisar()
    assert len(resultados) == 1
    assert resultados[0].id == r.id


def test_ordem_mais_recente_primeiro(hist):
    for i in range(1, 4):
        hist.adicionar(_registo(i))
    assert hist.pesquisar()[0].id == "caso_test_0003"


def test_pesquisa_por_query(hist):
    hist.adicionar(_registo(1))
    hist.adicionar(RegistoHistorico(
        id="caso_laboral_01", timestamp="2025-02-01T10:00:00+00:00",
        instancia_codigo="TRAB", instancia_nome="Tribunal do Trabalho",
        resumo="Despedimento ilícito sem justa causa",
        dispositivo="CONDENA a pagar indemnização",
        grau_incerteza="Baixo", custo_usd=0.05,
        modelo="gemini", n_entidades_anonimizadas=2,
    ))
    res = hist.pesquisar(query="despedimento")
    assert len(res) == 1
    assert res[0].instancia_codigo == "TRAB"


def test_pesquisa_por_instancia(hist):
    hist.adicionar(_registo(1))
    hist.adicionar(RegistoHistorico(
        id="caso_trab", timestamp="2025-02-01T10:00:00+00:00",
        instancia_codigo="TRAB", instancia_nome="Tribunal do Trabalho",
        resumo="Trabalho", dispositivo="CONDENA", grau_incerteza="Baixo",
        custo_usd=0.0, modelo="x", n_entidades_anonimizadas=1,
    ))
    res = hist.pesquisar(instancia="TRAB")
    assert all(r.instancia_codigo == "TRAB" for r in res)
    assert len(res) == 1


def test_persistencia(tmp_path):
    pasta = tmp_path / "hist"
    h1 = HistoricoCasos(pasta)
    h1.adicionar(_registo(1))
    h1.adicionar(_registo(2))

    h2 = HistoricoCasos(pasta)  # nova instância, mesma pasta
    assert h2.total() == 2


def test_limpar(hist):
    for i in range(3):
        hist.adicionar(_registo(i))
    assert hist.total() == 3
    hist.limpar()
    assert hist.total() == 0


def test_limite_500(hist):
    for i in range(510):
        hist.adicionar(_registo(i % 500 + 1))
    assert hist.total() <= 500


def test_estatisticas(hist):
    hist.adicionar(_registo(1))
    hist.adicionar(RegistoHistorico(
        id="x", timestamp="2025-01-01T00:00:00+00:00",
        instancia_codigo="TRAB", instancia_nome="Trabalho",
        resumo="", dispositivo="", grau_incerteza="Alto",
        custo_usd=0.10, modelo="x", n_entidades_anonimizadas=0,
    ))
    stats = hist.estatisticas()
    assert stats["total"] == 2
    assert "TIC" in stats["por_instancia"]
    assert "TRAB" in stats["por_instancia"]
    assert stats["custo_total_usd"] == pytest.approx(0.10)
