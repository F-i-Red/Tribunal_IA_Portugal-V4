"""Testes das instâncias judiciais — V4."""
import pytest
from src.pipeline.instancias import INSTANCIAS, detectar_instancia_por_keywords


def test_todas_instancias_tem_campos():
    for cod, inst in INSTANCIAS.items():
        assert inst.codigo == cod
        assert inst.nome
        assert inst.diploma_principal
        assert inst.keywords


def test_deteccao_penal():
    caso = "O arguido foi acusado de furto. A polícia fez a detenção."
    cod = detectar_instancia_por_keywords(caso)
    assert cod == "TIC"


def test_deteccao_laboral():
    caso = "Fui despedido sem justa causa. Não recebi os salários em atraso."
    cod = detectar_instancia_por_keywords(caso)
    assert cod == "TRAB"


def test_deteccao_familia():
    caso = "Quero o divórcio. Temos um filho menor e discordamos da guarda."
    cod = detectar_instancia_por_keywords(caso)
    assert cod == "TFM"


def test_deteccao_civel():
    caso = "O réu não cumpriu o contrato e causou danos. Quero indemnização."
    cod = detectar_instancia_por_keywords(caso)
    assert cod == "TC_CIVEL"


def test_deteccao_administrativo():
    caso = "A câmara municipal recusou a licença de obra sem fundamentação."
    cod = detectar_instancia_por_keywords(caso)
    assert cod == "TAF"


def test_fallback_tudo_desconhecido():
    cod = detectar_instancia_por_keywords("palavras aleatórias xyz abc 123")
    assert cod == "TIC"  # default


def test_recursos_nao_detectados_automaticamente():
    # STJ, TR, TC não devem aparecer na detecção automática
    caso = "recurso apelação recorrente recorrido relação acórdão"
    cod = detectar_instancia_por_keywords(caso)
    assert cod not in ("TR", "STJ", "TC")
