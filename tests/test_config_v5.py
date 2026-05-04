"""Testes de configuração V5 — inclui Ollama e RAG modes."""
import pytest
import os
from src.utils.config import Settings, ConfigError, get_config, reset_config


def _make_env(monkeypatch, **kw):
    defaults = {
        "OPENROUTER_API_KEY": "sk-or-test-valid-key-abcdef",
        "MODELO": "openrouter/auto",
        "BACKEND": "openrouter",
    }
    defaults.update(kw)
    for k, v in defaults.items():
        monkeypatch.setenv(k, v)
    reset_config()


def test_config_openrouter_valida(monkeypatch, tmp_path):
    _make_env(monkeypatch)
    cfg = get_config()
    assert cfg.backend == "openrouter"
    assert cfg.is_free_model is True
    assert cfg.usar_ollama is False
    reset_config()


def test_config_ollama(monkeypatch):
    monkeypatch.setenv("BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_MODELO", "llama3.3:70b")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sem-chave")  # sem chave — ollama não precisa
    reset_config()
    cfg = get_config()
    assert cfg.usar_ollama is True
    assert cfg.is_free_model is True  # ollama é sempre "free"
    assert cfg.custo_por_token == (0.0, 0.0)
    assert cfg.modelo_activo == "llama3.3:70b"
    reset_config()


def test_config_modelo_pago(monkeypatch):
    _make_env(monkeypatch, MODELO="anthropic/claude-sonnet-4.6")
    cfg = get_config()
    assert cfg.is_free_model is False
    assert cfg.custo_por_token[0] > 0
    reset_config()


def test_config_chave_invalida_openrouter(monkeypatch):
    monkeypatch.setenv("BACKEND", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sem-chave")
    reset_config()
    with pytest.raises((ConfigError, Exception)):
        get_config()
    reset_config()


def test_config_rag_modo(monkeypatch):
    _make_env(monkeypatch, RAG_MODO="hibrido")
    cfg = get_config()
    assert cfg.rag_modo == "hibrido"
    reset_config()


def test_config_features(monkeypatch):
    _make_env(monkeypatch,
              HISTORICO_ENABLED="true",
              EXPORTAR_PDF="true",
              CONSISTENCIA_CHECK="true")
    cfg = get_config()
    assert cfg.historico_enabled is True
    assert cfg.exportar_pdf is True
    assert cfg.consistencia_check is True
    reset_config()


def test_modelo_activo_openrouter(monkeypatch):
    _make_env(monkeypatch, MODELO="google/gemini-2.0-flash-001")
    cfg = get_config()
    assert cfg.modelo_activo == "google/gemini-2.0-flash-001"
    reset_config()


def test_modelo_activo_ollama(monkeypatch):
    monkeypatch.setenv("BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_MODELO", "qwen2.5:72b")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sem-chave")
    reset_config()
    cfg = get_config()
    assert cfg.modelo_activo == "qwen2.5:72b"
    reset_config()
