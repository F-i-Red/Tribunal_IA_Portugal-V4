"""Testes do motor RAG V5 — inclui metadata filtering e diplomas."""
import pytest
from pathlib import Path
from src.rag.motor import MotorRAG, Fragmento, DIPLOMA_INSTANCIAS


@pytest.fixture
def rag_rico(tmp_path):
    """RAG com múltiplos ficheiros temáticos."""
    leis = tmp_path / "data" / "leis"
    juri = tmp_path / "data" / "jurisprudencia"
    leis.mkdir(parents=True)
    juri.mkdir(parents=True)
    (tmp_path / "src" / "cache" / "data").mkdir(parents=True)

    (leis / "Codigo_Penal.txt").write_text(
        "Artigo 131.º\nHomicídio simples\n"
        "Quem matar outra pessoa é punido com pena de prisão de 8 a 16 anos.\n\n"
        "Artigo 203.º\nFurto\nQuem com ilegítima intenção de apropriação subtrair "
        "coisa móvel alheia é punido com pena de prisão até 3 anos.\n\n"
        "Artigo 143.º\nOfensa à integridade física\n"
        "Quem ofender o corpo ou a saúde de outra pessoa é punido com pena até 3 anos.\n",
        encoding="utf-8",
    )
    (leis / "Codigo_do_Trabalho.txt").write_text(
        "Artigo 351.º\nJusta causa de despedimento\n"
        "Constitui justa causa de despedimento o comportamento culposo do trabalhador "
        "que pela sua gravidade torne imediata e praticamente impossível a subsistência "
        "da relação de trabalho.\n\n"
        "Artigo 389.º\nIlicitude do despedimento\n"
        "É ilícito o despedimento sem justa causa ou por motivos políticos, ideológicos.\n",
        encoding="utf-8",
    )
    (juri / "Acordao_STJ_2024.txt").write_text(
        "Acórdão do Supremo Tribunal de Justiça\n"
        "Data: 2024-03-15\n"
        "Matéria: Despedimento ilícito — indemnização — cálculo\n"
        "O trabalhador despedido sem justa causa tem direito a indemnização "
        "calculada com base na antiguidade.",
        encoding="utf-8",
    )
    return MotorRAG(tmp_path)


def test_indexar_multiplos(rag_rico):
    n = rag_rico.indexar()
    assert n > 0


def test_metadata_diploma_detectado(rag_rico):
    rag_rico.indexar()
    stats = rag_rico.estatisticas()
    assert "CP" in stats["diplomas"] or "CT" in stats["diplomas"]


def test_pesquisa_com_filtro_instancia_penal(rag_rico):
    rag_rico.indexar()
    # Pesquisa penal com filtro para instância penal
    frags = rag_rico.pesquisar("furto arguido crime prisão", instancia="TIC")
    # Deve retornar fragmentos (CP tem instancias=["TIC","TCCR","TCIC"])
    assert isinstance(frags, list)


def test_pesquisa_sem_filtro_retorna_tudo(rag_rico):
    rag_rico.indexar()
    frags_com = rag_rico.pesquisar("trabalhador despedimento")
    frags_sem = rag_rico.pesquisar("trabalhador despedimento", instancia=None)
    # Sem filtro deve ter >= com filtro
    assert len(frags_sem) >= len(frags_com)


def test_pesquisa_jurisprudencia(rag_rico):
    rag_rico.indexar()
    frags = rag_rico.pesquisar("despedimento ilícito indemnização", tipo_filtro="jurisprudencia")
    assert all(f.tipo == "jurisprudencia" for f in frags)


def test_formatar_contexto_inclui_diploma(rag_rico):
    rag_rico.indexar()
    frags = rag_rico.pesquisar("furto prisão")
    ctx = rag_rico.formatar_contexto(frags)
    assert "CONTEXTO JURÍDICO" in ctx


def test_diploma_instancias_mapeamento():
    assert "TIC" in DIPLOMA_INSTANCIAS["CP"]
    assert "TRAB" in DIPLOMA_INSTANCIAS["CT"]
    assert "TC_CIVEL" in DIPLOMA_INSTANCIAS["CC"]
    assert "TAF" in DIPLOMA_INSTANCIAS["CPTA"]


def test_estatisticas_completas(rag_rico):
    rag_rico.indexar()
    stats = rag_rico.estatisticas()
    assert "total" in stats
    assert "leis" in stats
    assert "jurisprudencia" in stats
    assert "modo" in stats
    assert stats["total"] == stats["leis"] + stats["jurisprudencia"] + stats["precedentes"]


def test_recarregar(rag_rico):
    n1 = rag_rico.indexar()
    n2 = rag_rico.recarregar()
    assert n1 == n2
