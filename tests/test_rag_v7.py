"""Testes RAG V7 — prefixos por modelo, Cohere mock, metadata filtering."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.rag.motor import MotorRAG, Fragmento, DIPLOMA_INSTANCIAS, DIPLOMA_KEYWORDS


@pytest.fixture
def rag_base(tmp_path):
    """RAG BM25 com dados de teste."""
    for pasta in ["data/leis","data/jurisprudencia","data/tedh","src/cache/data"]:
        (tmp_path / pasta).mkdir(parents=True)
    (tmp_path / "data/leis/Codigo_Penal.txt").write_text(
        "Artigo 131.º\nHomicídio\nQuem matar outra pessoa é punido com 8 a 16 anos de prisão.\n\n"
        "Artigo 203.º\nFurto\nQuem subtrair coisa alheia é punido até 3 anos.\n",
        encoding="utf-8",
    )
    (tmp_path / "data/leis/Codigo_do_Trabalho.txt").write_text(
        "Artigo 351.º\nJusta causa\nConstitui justa causa o comportamento culposo do trabalhador.\n\n"
        "Artigo 389.º\nIlicitude\nÉ ilícito o despedimento sem justa causa.\n",
        encoding="utf-8",
    )
    (tmp_path / "data/tedh/ECHR_Article6.txt").write_text(
        "Article 6 — Right to a fair trial\n"
        "Everyone is entitled to a fair hearing by an independent tribunal.\n",
        encoding="utf-8",
    )
    return MotorRAG(tmp_path, modo="bm25")


# ── Prefixos por modelo ───────────────────────────────────────────────
class TestPrefixos:
    def test_e5_large_tem_prefixos(self, tmp_path):
        rag = MotorRAG(tmp_path, modo="hibrido",
                       embedding_modelo="intfloat/multilingual-e5-large-instruct")
        assert rag._prefixo_query == "query: "
        assert rag._prefixo_passage == "passage: "

    def test_e5_base_tem_prefixos(self, tmp_path):
        rag = MotorRAG(tmp_path, modo="hibrido",
                       embedding_modelo="intfloat/multilingual-e5-base")
        assert rag._prefixo_query == "query: "

    def test_neuralmind_sem_prefixos(self, tmp_path):
        rag = MotorRAG(tmp_path, modo="hibrido",
                       embedding_modelo="neuralmind/bert-base-portuguese-cased")
        assert rag._prefixo_query == ""
        assert rag._prefixo_passage == ""

    def test_portulan_sem_prefixos(self, tmp_path):
        rag = MotorRAG(tmp_path, modo="hibrido",
                       embedding_modelo="PORTULAN/serafim-pt-small-100m-lingua-pt")
        assert rag._prefixo_query == ""
        assert rag._prefixo_passage == ""

    def test_minilm_sem_prefixos(self, tmp_path):
        rag = MotorRAG(tmp_path, modo="hibrido",
                       embedding_modelo="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        assert rag._prefixo_query == ""


# ── Fallback BM25 sem sentence-transformers ───────────────────────────
class TestFallback:
    def test_hibrido_sem_st_usa_bm25(self, tmp_path):
        (tmp_path / "src/cache/data").mkdir(parents=True)
        (tmp_path / "data/leis").mkdir(parents=True)
        with patch("src.rag.motor.ST_OK", False):
            rag = MotorRAG(tmp_path, modo="hibrido")
            assert rag.modo == "bm25"

    def test_reranking_desactivado_sem_st(self, tmp_path):
        (tmp_path / "src/cache/data").mkdir(parents=True)
        with patch("src.rag.motor.ST_OK", False):
            rag = MotorRAG(tmp_path, modo="hibrido", usar_reranking=True)
            assert rag.usar_reranking is False


# ── BM25 básico ───────────────────────────────────────────────────────
class TestBM25:
    def test_indexar(self, rag_base): assert rag_base.indexar() > 0
    def test_pesquisa_penal(self, rag_base):
        rag_base.indexar()
        frags = rag_base.pesquisar("furto prisão arguido")
        assert len(frags) > 0 and frags[0].relevancia > 0
    def test_pesquisa_laboral(self, rag_base):
        rag_base.indexar()
        frags = rag_base.pesquisar("despedimento trabalhador justa causa")
        assert len(frags) > 0
    def test_pesquisa_vazia(self, rag_base):
        rag_base.indexar()
        assert rag_base.pesquisar("xyz_inexistente_v7") == []
    def test_top_n_respeitado(self, rag_base):
        rag_base.indexar()
        assert len(rag_base.pesquisar("crime", n_resultados=2)) <= 2


# ── Metadata filtering ────────────────────────────────────────────────
class TestFiltering:
    def test_filtro_lingua_pt(self, rag_base):
        rag_base.indexar()
        frags = rag_base.pesquisar("crime prisão", lingua_filtro="pt")
        assert all(f.lingua == "pt" for f in frags)

    def test_filtro_lingua_en_tedh(self, rag_base):
        rag_base.indexar()
        frags = rag_base.pesquisar("fair trial tribunal", lingua_filtro="en")
        assert all(f.lingua == "en" for f in frags)

    def test_filtro_tipo_lei(self, rag_base):
        rag_base.indexar()
        frags = rag_base.pesquisar("trabalhador", tipo_filtro="lei")
        assert all(f.tipo == "lei" for f in frags)

    def test_filtro_instancia(self, rag_base):
        rag_base.indexar()
        frags = rag_base.pesquisar("despedimento", instancia="TRAB")
        for f in frags:
            assert not f.instancias or "TRAB" in f.instancias

    def test_sem_filtro_mais_resultados(self, rag_base):
        rag_base.indexar()
        sem = rag_base.pesquisar("crime")
        com = rag_base.pesquisar("crime", lingua_filtro="pt")
        assert len(sem) >= len(com)


# ── RRF ───────────────────────────────────────────────────────────────
class TestRRF:
    def test_rrf_formula(self):
        score = MotorRAG._rrf(1, 1, k=60)
        assert abs(score - 2/61) < 1e-9

    def test_rrf_rank_alto_menor_score(self):
        s1 = MotorRAG._rrf(1, 1)
        s2 = MotorRAG._rrf(10, 10)
        assert s1 > s2

    def test_rrf_simetrico(self):
        assert MotorRAG._rrf(3, 5) == MotorRAG._rrf(5, 3)


# ── Cohere rerank mock ────────────────────────────────────────────────
class TestCohereRerank:
    def test_cohere_rerank_mock(self, tmp_path):
        (tmp_path / "src/cache/data").mkdir(parents=True)
        (tmp_path / "data/leis").mkdir(parents=True)

        mock_cohere = MagicMock()
        result_mock = MagicMock()
        result_mock.results = [
            MagicMock(index=0, relevance_score=0.95),
            MagicMock(index=1, relevance_score=0.72),
        ]
        mock_cohere.rerank.return_value = result_mock

        candidatos = [
            (0.8, Fragmento("fonte1","lei","titulo1","conteudo furto crime",0.8,lingua="pt")),
            (0.6, Fragmento("fonte2","lei","titulo2","conteudo arguido pena",0.6,lingua="pt")),
        ]

        # Forçar COHERE_OK=True para testar o ramo Cohere
        with patch("src.rag.motor.COHERE_OK", True):
            rag = MotorRAG(tmp_path, modo="bm25", reranker_backend="cohere")
            rag._cohere_client = mock_cohere
            resultado = rag._rerank_cohere("furto qualificado", candidatos)

        assert len(resultado) > 0
        assert resultado[0].rerank_score > 0

    def test_cohere_falha_fallback_local(self, tmp_path):
        (tmp_path / "src/cache/data").mkdir(parents=True)

        mock_cohere = MagicMock()
        mock_cohere.rerank.side_effect = Exception("API error")

        rag = MotorRAG(tmp_path, modo="bm25", reranker_backend="cohere")
        rag._cohere_client = mock_cohere
        # _rerank_model None → retorna sem reranking
        rag._rerank_model = None

        candidatos = [
            (0.8, Fragmento("f1","lei","t1","conteudo",0.8,lingua="pt")),
        ]
        resultado = rag._rerank_cohere("query", candidatos)
        assert isinstance(resultado, list)


# ── Estatísticas V7 ───────────────────────────────────────────────────
class TestEstatisticasV7:
    def test_campos_v7(self, rag_base):
        rag_base.indexar()
        s = rag_base.estatisticas()
        assert "reranker_backend" in s
        assert "prefixo_query" in s
        assert s["total"] == s["leis"] + s["jurisprudencia"] + s["precedentes"] + s["tedh"]

    def test_tedh_contado(self, rag_base):
        rag_base.indexar()
        s = rag_base.estatisticas()
        assert s["tedh"] > 0
