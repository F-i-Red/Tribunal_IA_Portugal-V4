"""Testes do módulo de modo contraditório V6."""
import pytest
from src.contraditorio import (
    GestorContraditorio, SessaoContraditorio, IntervencaoDefesa, get_gestor_contraditorio
)


@pytest.fixture
def gestor():
    return GestorContraditorio()


@pytest.fixture
def sessao(gestor):
    return gestor.iniciar_sessao(
        case_id="caso_test_001",
        instancia_codigo="TIC",
        detetive="Relatório de instrução factual...",
        acusacao="O arguido é acusado de furto qualificado no valor de 5.000€...",
    )


def test_iniciar_sessao(gestor):
    s = gestor.iniciar_sessao("c001", "TIC", "detetive", "acusacao")
    assert s.case_id == "c001"
    assert s.instancia_codigo == "TIC"
    assert len(s.intervencoes) == 0


def test_adicionar_intervencao(sessao):
    iv = sessao.adicionar_intervencao(
        argumento="O meu cliente tem álibi confirmado por três testemunhas.",
        feedback_juridico="Argumento forte — álibi testemunhal é prova directa.",
    )
    assert iv.numero == 1
    assert iv.argumento == "O meu cliente tem álibi confirmado por três testemunhas."
    assert iv.feedback_juridico != ""


def test_multiplas_intervencoes(sessao):
    sessao.adicionar_intervencao("Argumento 1")
    sessao.adicionar_intervencao("Argumento 2")
    sessao.adicionar_intervencao("Argumento 3")
    assert len(sessao.intervencoes) == 3
    assert sessao.intervencoes[0].numero == 1
    assert sessao.intervencoes[2].numero == 3


def test_texto_completo_intervencoes(sessao):
    sessao.adicionar_intervencao("Argumento A")
    sessao.adicionar_intervencao("Argumento B")
    texto = sessao.texto_completo_intervencoes()
    assert "Argumento A" in texto
    assert "Argumento B" in texto
    assert "[Argumento 1]" in texto
    assert "[Argumento 2]" in texto


def test_texto_sem_intervencoes(sessao):
    texto = sessao.texto_completo_intervencoes()
    assert texto == ""


def test_gestor_obter_sessao(gestor):
    gestor.iniciar_sessao("c002", "TRAB", "det", "acus")
    s = gestor.obter_sessao("c002")
    assert s is not None
    assert s.case_id == "c002"


def test_gestor_sessao_inexistente(gestor):
    s = gestor.obter_sessao("nao_existe")
    assert s is None


def test_gestor_submeter_sem_avaliar(gestor):
    gestor.iniciar_sessao("c003", "TIC", "det", "acus")
    iv = gestor.submeter_argumento("c003", "Argumento sem avaliação", avaliar=False)
    assert iv.argumento == "Argumento sem avaliação"
    assert iv.feedback_juridico == ""


def test_gestor_sessao_nao_encontrada(gestor):
    with pytest.raises(ValueError, match="não encontrada"):
        gestor.submeter_argumento("sessao_invalida", "argumento")


def test_resumo_intervencoes(gestor):
    gestor.iniciar_sessao("c004", "TC_CIVEL", "det", "acus")
    gestor.submeter_argumento("c004", "Argumento sobre prescrição", avaliar=False)
    resumo = gestor.resumo_intervencoes("c004")
    assert "Argumento sobre prescrição" in resumo
    assert "1" in resumo


def test_resumo_sem_intervencoes(gestor):
    gestor.iniciar_sessao("c005", "TIC", "det", "acus")
    resumo = gestor.resumo_intervencoes("c005")
    assert "Sem intervenções" in resumo


def test_singleton_gestor():
    g1 = get_gestor_contraditorio()
    g2 = get_gestor_contraditorio()
    assert g1 is g2
