"""Testes de configuração V6 — RAG híbrido, reranking, LangGraph, API."""
import pytest
from src.utils.config import Settings, ConfigError, get_config, reset_config


def _env(monkeypatch, **kw):
    defaults = {
        "OPENROUTER_API_KEY": "sk-or-test-valid-v6",
        "MODELO": "openrouter/free",
        "BACKEND": "openrouter",
    }
    defaults.update(kw)
    for k, v in defaults.items():
        monkeypatch.setenv(k, v)
    reset_config()


def test_config_valida_openrouter(monkeypatch):
    _env(monkeypatch)
    cfg = get_config()
    assert cfg.backend == "openrouter"
    assert cfg.is_free_model is True
    assert cfg.usar_ollama is False
    reset_config()


def test_config_ollama_sem_chave(monkeypatch):
    monkeypatch.setenv("BACKEND", "ollama")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sem-chave")
    monkeypatch.setenv("OLLAMA_MODELO", "qwen2.5:72b")
    reset_config()
    cfg = get_config()
    assert cfg.usar_ollama is True
    assert cfg.is_free_model is True
    assert cfg.modelo_activo == "qwen2.5:72b"
    assert cfg.custo_por_token == (0.0, 0.0)
    reset_config()


def test_rag_modo_hibrido(monkeypatch):
    _env(monkeypatch, RAG_MODO="hibrido")
    cfg = get_config()
    assert cfg.rag_modo == "hibrido"
    reset_config()


def test_rag_reranking_activo(monkeypatch):
    _env(monkeypatch, RAG_RERANKING="true")
    cfg = get_config()
    assert cfg.rag_reranking is True
    reset_config()


def test_rag_top_k_top_n(monkeypatch):
    _env(monkeypatch, RAG_TOP_K="20", RAG_TOP_N="8")
    cfg = get_config()
    assert cfg.rag_top_k == 20
    assert cfg.rag_top_n == 8
    reset_config()


def test_rag_embedding_modelo(monkeypatch):
    _env(monkeypatch, RAG_EMBEDDING_MODELO="intfloat/multilingual-e5-base")
    cfg = get_config()
    assert cfg.rag_embedding_modelo == "intfloat/multilingual-e5-base"
    reset_config()


def test_orquestracao_langgraph(monkeypatch):
    _env(monkeypatch, ORQUESTRACAO="langgraph")
    cfg = get_config()
    assert cfg.orquestracao == "langgraph"
    # usar_langgraph depende de langgraph estar instalado
    assert isinstance(cfg.usar_langgraph, bool)
    reset_config()


def test_orquestracao_imperativa(monkeypatch):
    _env(monkeypatch, ORQUESTRACAO="imperativo")
    cfg = get_config()
    assert cfg.orquestracao == "imperativo"
    assert cfg.usar_langgraph is False
    reset_config()


def test_funcionalidades_v6(monkeypatch):
    _env(monkeypatch,
         CONTRADITORIO_ENABLED="true",
         MULTILINGUE_ENABLED="true",
         CONSISTENCIA_CHECK="true",
         EXPORTAR_PDF="true")
    cfg = get_config()
    assert cfg.contraditorio_enabled is True
    assert cfg.multilingue_enabled is True
    assert cfg.consistencia_check is True
    assert cfg.exportar_pdf is True
    reset_config()


def test_api_config(monkeypatch):
    _env(monkeypatch, API_PORT="9000", API_HOST="127.0.0.1")
    cfg = get_config()
    assert cfg.api_port == 9000
    assert cfg.api_host == "127.0.0.1"
    reset_config()


def test_openrouter_free_reconhecido(monkeypatch):
    _env(monkeypatch, MODELO="openrouter/free")
    cfg = get_config()
    assert cfg.is_free_model is True
    assert cfg.custo_por_token == (0.0, 0.0)
    reset_config()


def test_modelo_pago_tem_custo(monkeypatch):
    _env(monkeypatch, MODELO="anthropic/claude-sonnet-4.6")
    cfg = get_config()
    assert cfg.is_free_model is False
    assert cfg.custo_por_token[0] > 0
    reset_config()


def test_chave_invalida_levanta_erro(monkeypatch):
    monkeypatch.setenv("BACKEND", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sem-chave")
    reset_config()
    with pytest.raises((ConfigError, Exception)):
        get_config()
    reset_config()
