"""
Testes E2E da API V7.
Usa httpx.AsyncClient + respx para mock das chamadas LLM.
Não requer chave API real — tudo mockado.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Any, Dict

try:
    import httpx
    from fastapi.testclient import TestClient
    HTTPX_OK = True
except ImportError:
    HTTPX_OK = False

pytestmark = pytest.mark.skipif(not HTTPX_OK, reason="httpx/fastapi não instalados")


# ── Mock da resposta LLM ──────────────────────────────────────────────
MOCK_LLM_RESPONSE = {
    "choices": [{
        "message": {
            "content": (
                "## FACTOS ALEGADOS\n1. Facto A\n2. Facto B\n\n"
                "## DISPOSITIVO\nO Tribunal DECIDE: CONDENA o arguido a 2 anos suspensos."
            )
        }
    }],
    "model": "openrouter/free",
    "usage": {"prompt_tokens": 100, "completion_tokens": 150},
}

MOCK_INSTRUCAO_RESPONSE = {
    "choices": [{
        "message": {
            "content": json.dumps({
                "introducao": "O juiz solicita esclarecimentos sobre este caso concreto.",
                "perguntas": [
                    {"id": "q1", "texto": "Quando ocorreram os factos?",
                     "categoria": "TEMPORAL", "importancia": "critica",
                     "aceita_documentos": False, "razao": "Determinar prescrição"},
                    {"id": "q2", "texto": "Existem testemunhas presenciais?",
                     "categoria": "TESTEMUNHAS", "importancia": "relevante",
                     "aceita_documentos": False, "razao": "Suporte probatório"},
                ]
            })
        }
    }],
    "model": "openrouter/free",
    "usage": {"prompt_tokens": 80, "completion_tokens": 120},
}


@pytest.fixture(scope="module")
def mock_env(tmp_path_factory: Any):
    """Configura ambiente de teste com variáveis mock."""
    import os
    tmp = tmp_path_factory.mktemp("tribunal_test")
    os.environ["OPENROUTER_API_KEY"] = "sk-or-test-e2e-key"
    os.environ["MODELO"] = "openrouter/free"
    os.environ["BACKEND"] = "openrouter"
    os.environ["API_SECRET_KEY"] = "chave_teste_e2e_32_chars_minimo_abc"
    os.environ["CACHE_ENABLED"] = "false"
    os.environ["GUARDAR_ATAS"] = "false"
    os.environ["HISTORICO_ENABLED"] = "false"
    os.environ["EXPORTAR_PDF"] = "false"
    os.environ["CONSISTENCIA_CHECK"] = "false"
    os.environ["MULTILINGUE_ENABLED"] = "false"
    os.environ["RAG_MODO"] = "bm25"
    os.environ["ORQUESTRACAO"] = "imperativo"
    for d in ["data/leis","data/jurisprudencia","data/precedentes","data/tedh",
              "output_atas","logs","src/cache/data","src/historico/data"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    yield tmp


@pytest.fixture(scope="module")
def client(mock_env: Any):
    """TestClient FastAPI com LLM completamente mockado."""
    from src.utils.config import reset_config
    from src.utils.brain import reset_brain
    reset_config()
    reset_brain()

    try:
        from src.api import app, FASTAPI_OK
        if not FASTAPI_OK or app is None:
            pytest.skip("FastAPI não instalada")
    except Exception as e:
        pytest.skip(f"API não disponível: {e}")

    def _mock_llm(*args: Any, **kwargs: Any) -> MagicMock:
        r = MagicMock()
        r.status_code = 200
        # Detectar se é instrução
        body = kwargs.get("json", {})
        msgs = body.get("messages", [])
        conteudo = " ".join(m.get("content","") for m in msgs)
        if "instrução" in conteudo.lower() or "perguntas" in conteudo.lower() or "instrucao" in conteudo.lower():
            r.json.return_value = MOCK_INSTRUCAO_RESPONSE
        else:
            r.json.return_value = MOCK_LLM_RESPONSE
        r.raise_for_status = MagicMock()
        return r

    with patch("httpx.Client") as mock_httpx:
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.post = _mock_llm
        mock_httpx.return_value = mock_ctx

        with TestClient(app) as c:
            yield c


@pytest.fixture(scope="module")
def token(client: Any) -> str:
    """JWT token para testes autenticados."""
    resp = client.post("/auth/token", json={"username": "test", "password": "test"})
    assert resp.status_code == 200
    return resp.json()["access_token"]


@pytest.fixture
def auth_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ── Testes de sistema ─────────────────────────────────────────────────
class TestSistema:
    def test_raiz(self, client: Any) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "versao" in data
        assert data["versao"] == "7.0.0"

    def test_saude(self, client: Any) -> None:
        resp = client.get("/saude")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["versao"] == "7.0.0"
        assert "modelo" in data
        assert "rag_modo" in data

    def test_instancias(self, client: Any) -> None:
        resp = client.get("/instancias")
        assert resp.status_code == 200
        data = resp.json()
        assert "TIC" in data
        assert "TRAB" in data
        assert "TC_CIVEL" in data
        assert len(data) >= 10

    def test_metrics_endpoint(self, client: Any) -> None:
        resp = client.get("/metrics")
        assert resp.status_code == 200


# ── Testes de autenticação ────────────────────────────────────────────
class TestAuth:
    def test_token_emitido(self, client: Any) -> None:
        resp = client.post("/auth/token", json={"username": "user", "password": "pass"})
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    def test_endpoint_protegido_sem_token(self, client: Any) -> None:
        resp = client.get("/historico")
        assert resp.status_code == 401

    def test_endpoint_protegido_token_invalido(self, client: Any) -> None:
        resp = client.get("/historico", headers={"Authorization": "Bearer token_invalido"})
        assert resp.status_code == 401

    def test_endpoint_com_token_valido(self, client: Any, auth_headers: Dict[str,str]) -> None:
        resp = client.get("/historico", headers=auth_headers)
        assert resp.status_code == 200


# ── Testes de instrução ───────────────────────────────────────────────
class TestInstrucao:
    def test_instrucao_ok(self, client: Any, auth_headers: Dict[str,str]) -> None:
        resp = client.post("/instrucao",
                           json={"descricao": "Fui despedido sem justa causa após 8 anos.", "instancia": "TRAB"},
                           headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["sucesso"] is True
        assert "perguntas" in data

    def test_instrucao_descricao_curta(self, client: Any, auth_headers: Dict[str,str]) -> None:
        resp = client.post("/instrucao", json={"descricao": "curta"}, headers=auth_headers)
        assert resp.status_code == 422

    def test_instrucao_sem_auth(self, client: Any) -> None:
        resp = client.post("/instrucao", json={"descricao": "Caso de teste adequado."})
        assert resp.status_code == 401


# ── Testes de processamento ───────────────────────────────────────────
class TestProcessar:
    def test_processar_caso_basico(self, client: Any, auth_headers: Dict[str,str]) -> None:
        resp = client.post("/processar",
                           json={"descricao": "Fui despedido sem justa causa. Tenho contrato e recibos.", "instancia": "TRAB"},
                           headers=auth_headers, timeout=120)
        assert resp.status_code == 200
        data = resp.json()
        assert data["sucesso"] is True
        assert "case_id" in data
        assert "tribunal" in data
        assert "pecas" in data
        assert "detetive" in data["pecas"]

    def test_processar_sem_auth(self, client: Any) -> None:
        resp = client.post("/processar",
                           json={"descricao": "Caso de teste com descrição suficientemente longa."})
        assert resp.status_code == 401

    def test_processar_descricao_curta(self, client: Any, auth_headers: Dict[str,str]) -> None:
        resp = client.post("/processar", json={"descricao": "curta"}, headers=auth_headers)
        assert resp.status_code == 422


# ── Testes RAG ────────────────────────────────────────────────────────
class TestRAG:
    def test_rag_stats(self, client: Any, auth_headers: Dict[str,str]) -> None:
        resp = client.get("/rag/stats", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "modo" in data

    def test_rag_pesquisar(self, client: Any, auth_headers: Dict[str,str]) -> None:
        resp = client.get("/rag/pesquisar?q=furto+crime&n=3", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "query" in data
        assert "resultados" in data


# ── Testes histórico ──────────────────────────────────────────────────
class TestHistorico:
    def test_historico_vazio(self, client: Any, auth_headers: Dict[str,str]) -> None:
        resp = client.get("/historico", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "registos" in data

    def test_historico_com_filtro(self, client: Any, auth_headers: Dict[str,str]) -> None:
        resp = client.get("/historico?instancia=TRAB&limite=5", headers=auth_headers)
        assert resp.status_code == 200


# ── Testes métricas ───────────────────────────────────────────────────
class TestMetricas:
    def test_resumo_metricas(self, client: Any, auth_headers: Dict[str,str]) -> None:
        resp = client.get("/metricas/resumo", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "total_casos" in data
