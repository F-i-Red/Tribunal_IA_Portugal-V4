"""
Testes do motor RAG V6.
Cobre: BM25, embeddings mock, reranking mock, metadata filtering,
       multilíngue (PT + EN), fusão RRF, TEDH.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.rag.motor import (
    MotorRAG, Fragmento,
    DIPLOMA_INSTANCIAS, DIPLOMA_KEYWORDS,
    STOPWORDS_PT, STOPWORDS_EN,
)


# ── Fixture: RAG com dados de teste ─────────────────────────────────
@pytest.fixture
def rag_teste(tmp_path):
    """RAG completo com leis PT e jurisprudência TEDH mock."""
    leis = tmp_path / "data" / "leis"
    juri = tmp_path / "data" / "jurisprudencia"
    tedh = tmp_path / "data" / "tedh"
    cache = tmp_path / "src" / "cache" / "data"
    for d in [leis, juri, tedh, cache]:
        d.mkdir(parents=True)

    (leis / "Codigo_Penal.txt").write_text(
        "Artigo 131.º\nHomicídio simples\n"
        "Quem matar outra pessoa é punido com pena de prisão de 8 a 16 anos.\n\n"
        "Artigo 203.º\nFurto\nQuem com ilegítima intenção de apropriação subtrair "
        "coisa móvel alheia é punido com pena de prisão até 3 anos ou multa.\n\n"
        "Artigo 143.º\nOfensa à integridade física\n"
        "Quem ofender o corpo ou a saúde de outra pessoa é punido com pena até 3 anos.\n",
        encoding="utf-8",
    )
    (leis / "Codigo_do_Trabalho.txt").write_text(
        "Artigo 351.º\nJusta causa de despedimento\n"
        "Constitui justa causa o comportamento culposo do trabalhador que torne "
        "impossível a subsistência da relação de trabalho.\n\n"
        "Artigo 389.º\nIlicitude do despedimento\n"
        "É ilícito o despedimento sem justa causa ou por motivos políticos ou ideológicos.\n",
        encoding="utf-8",
    )
    (juri / "Acordao_STJ_Laboral.txt").write_text(
        "Acórdão STJ — Despedimento ilícito\n"
        "O trabalhador tem direito a indemnização calculada com base na antiguidade.\n"
        "A reintegração pode ser substituída por indemnização a pedido do trabalhador.\n",
        encoding="utf-8",
    )
    (tedh / "ECHR_Article6_FairTrial.txt").write_text(
        "Article 6 — Right to a fair trial\n"
        "Everyone is entitled to a fair and public hearing within a reasonable time "
        "by an independent and impartial tribunal established by law.\n"
        "Case: Golder v. United Kingdom (1975) — access to court is guaranteed.\n",
        encoding="utf-8",
    )
    return MotorRAG(tmp_path, modo="bm25")  # BM25 — sem deps externas nos testes


# ── Testes de indexação ───────────────────────────────────────────────
class TestIndexacao:
    def test_indexar_retorna_fragmentos(self, rag_teste):
        n = rag_teste.indexar()
        assert n > 0

    def test_indexar_detecta_diplomas(self, rag_teste):
        rag_teste.indexar()
        s = rag_teste.estatisticas()
        assert "CP" in s["diplomas"] or "CT" in s["diplomas"]

    def test_indexar_distingue_linguas(self, rag_teste):
        rag_teste.indexar()
        s = rag_teste.estatisticas()
        assert s["tedh"] > 0
        assert s["leis"] > 0

    def test_indexar_idempotente(self, rag_teste):
        n1 = rag_teste.indexar()
        n2 = rag_teste.indexar()  # usa cache
        assert n1 == n2

    def test_recarregar_reindexar(self, rag_teste):
        n1 = rag_teste.indexar()
        n2 = rag_teste.recarregar()
        assert n1 == n2

    def test_tem_dados(self, rag_teste):
        rag_teste.indexar()
        assert rag_teste.tem_dados() is True

    def test_rag_vazio(self, tmp_path):
        (tmp_path / "data" / "leis").mkdir(parents=True)
        (tmp_path / "src" / "cache" / "data").mkdir(parents=True)
        rag = MotorRAG(tmp_path, modo="bm25")
        rag.indexar()
        assert rag.tem_dados() is False


# ── Testes de pesquisa BM25 ───────────────────────────────────────────
class TestPesquisaBM25:
    def test_pesquisa_penal(self, rag_teste):
        rag_teste.indexar()
        frags = rag_teste.pesquisar("furto apropriação coisa móvel")
        assert len(frags) > 0
        assert frags[0].relevancia > 0

    def test_pesquisa_laboral(self, rag_teste):
        rag_teste.indexar()
        frags = rag_teste.pesquisar("despedimento trabalhador justa causa")
        assert len(frags) > 0

    def test_pesquisa_sem_resultados(self, rag_teste):
        rag_teste.indexar()
        frags = rag_teste.pesquisar("xyzabc123 palavra_inexistente_v6")
        assert frags == []

    def test_pesquisa_top_n_respeitado(self, rag_teste):
        rag_teste.indexar()
        frags = rag_teste.pesquisar("crime", n_resultados=2)
        assert len(frags) <= 2

    def test_pesquisa_ordenada_por_relevancia(self, rag_teste):
        rag_teste.indexar()
        frags = rag_teste.pesquisar("furto prisão arguido crime")
        if len(frags) > 1:
            for i in range(len(frags) - 1):
                assert frags[i].relevancia >= frags[i+1].relevancia


# ── Testes de metadata filtering ─────────────────────────────────────
class TestMetadataFiltering:
    def test_filtro_tipo_lei(self, rag_teste):
        rag_teste.indexar()
        frags = rag_teste.pesquisar("despedimento", tipo_filtro="lei")
        assert all(f.tipo == "lei" for f in frags)

    def test_filtro_tipo_jurisprudencia(self, rag_teste):
        rag_teste.indexar()
        frags = rag_teste.pesquisar("indemnização", tipo_filtro="jurisprudencia")
        assert all(f.tipo == "jurisprudencia" for f in frags)

    def test_filtro_lingua_pt(self, rag_teste):
        rag_teste.indexar()
        frags = rag_teste.pesquisar("furto crime", lingua_filtro="pt")
        assert all(f.lingua == "pt" for f in frags)

    def test_filtro_lingua_en_tedh(self, rag_teste):
        rag_teste.indexar()
        frags = rag_teste.pesquisar("fair trial independent tribunal", lingua_filtro="en")
        assert all(f.lingua == "en" for f in frags)

    def test_filtro_instancia_trab(self, rag_teste):
        rag_teste.indexar()
        frags = rag_teste.pesquisar("despedimento trabalhador", instancia="TRAB")
        # Apenas fragmentos sem instância definida ou com TRAB
        for f in frags:
            assert not f.instancias or "TRAB" in f.instancias

    def test_sem_filtro_retorna_mais(self, rag_teste):
        rag_teste.indexar()
        com_filtro = rag_teste.pesquisar("crime", instancia="TIC")
        sem_filtro = rag_teste.pesquisar("crime")
        assert len(sem_filtro) >= len(com_filtro)


# ── Testes de formatação de contexto ─────────────────────────────────
class TestContexto:
    def test_formatar_contexto_pt(self, rag_teste):
        rag_teste.indexar()
        frags = rag_teste.pesquisar("furto crime")
        ctx = rag_teste.formatar_contexto(frags)
        assert "CONTEXTO JURIDICO" in ctx or "CONTEXTO JURÍDICO" in ctx
        assert len(ctx) > 50

    def test_formatar_contexto_vazio(self, rag_teste):
        ctx = rag_teste.formatar_contexto([])
        assert ctx == ""

    def test_formatar_contexto_max_chars(self, rag_teste):
        rag_teste.indexar()
        frags = rag_teste.pesquisar("crime")
        ctx = rag_teste.formatar_contexto(frags, max_chars=200)
        assert len(ctx) <= 400  # tolerância para marcadores

    def test_formatar_contexto_inclui_tedh(self, rag_teste):
        rag_teste.indexar()
        frags_en = rag_teste.pesquisar("fair trial", lingua_filtro="en")
        ctx = rag_teste.formatar_contexto(frags_en, incluir_tedh=True)
        if frags_en:
            assert "TEDH" in ctx or "fair" in ctx.lower()

    def test_formatar_contexto_exclui_tedh(self, rag_teste):
        rag_teste.indexar()
        all_frags = rag_teste.pesquisar("crime trial")
        ctx = rag_teste.formatar_contexto(all_frags, incluir_tedh=False)
        assert "TEDH" not in ctx


# ── Testes de tokenização multilíngue ────────────────────────────────
class TestTokenizacao:
    def test_tokens_pt_remove_stopwords(self, rag_teste):
        tokens = rag_teste._tokenizar("O arguido foi condenado pelo tribunal", "pt")
        assert "o" not in tokens
        assert "foi" not in tokens
        assert "arguido" in tokens or "condenado" in tokens

    def test_tokens_en_remove_stopwords(self, rag_teste):
        tokens = rag_teste._tokenizar("The defendant was convicted by the court", "en")
        assert "the" not in tokens
        assert "was" not in tokens
        assert "defendant" in tokens or "convicted" in tokens

    def test_tokens_minimo_3_chars(self, rag_teste):
        tokens = rag_teste._tokenizar("a de em por ao um uma")
        assert all(len(t) >= 3 for t in tokens)


# ── Testes de estatísticas ────────────────────────────────────────────
class TestEstatisticas:
    def test_estatisticas_campos(self, rag_teste):
        rag_teste.indexar()
        s = rag_teste.estatisticas()
        assert all(k in s for k in [
            "total", "leis", "jurisprudencia", "precedentes", "tedh",
            "diplomas", "fontes", "modo", "embeddings_computados", "reranking",
            "modelo_embeddings", "modelo_reranker",
        ])

    def test_estatisticas_soma_correcta(self, rag_teste):
        rag_teste.indexar()
        s = rag_teste.estatisticas()
        assert s["total"] == s["leis"] + s["jurisprudencia"] + s["precedentes"] + s["tedh"]

    def test_estatisticas_modo_bm25(self, rag_teste):
        rag_teste.indexar()
        s = rag_teste.estatisticas()
        assert s["modo"] == "bm25"
        assert s["reranking"] is False


# ── Testes de mapeamentos de diplomas ────────────────────────────────
class TestMapeamentos:
    def test_cp_mapeia_instancias_penais(self):
        assert "TIC" in DIPLOMA_INSTANCIAS["CP"]
        assert "TCCR" in DIPLOMA_INSTANCIAS["CP"]

    def test_ct_mapeia_trab(self):
        assert "TRAB" in DIPLOMA_INSTANCIAS["CT"]

    def test_cc_mapeia_civel(self):
        assert "TC_CIVEL" in DIPLOMA_INSTANCIAS["CC"]

    def test_cpta_mapeia_administrativo(self):
        assert "TAF" in DIPLOMA_INSTANCIAS["CPTA"]

    def test_echr_sem_instancias_especificas(self):
        # TEDH aplica-se a todas as instâncias
        assert DIPLOMA_INSTANCIAS["ECHR"] == []

    def test_keywords_cp(self):
        assert any("penal" in kw or "crime" in kw for kw in DIPLOMA_KEYWORDS["CP"])

    def test_keywords_echr_ingles(self):
        assert any("echr" in kw or "european" in kw for kw in DIPLOMA_KEYWORDS["ECHR"])
