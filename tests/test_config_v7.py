"""Testes de configuração V7 — PORTULAN, Cohere, JWT, Observability."""
import pytest
from src.utils.config import (
    Settings, ConfigError, get_config, reset_config,
    FREE_MODELS, PAID_MODELS, EMBEDDING_MODELS,
)


def _env(monkeypatch, **kw):
    defaults = {
        "OPENROUTER_API_KEY": "sk-or-test-v7-valid",
        "MODELO": "openrouter/free",
        "BACKEND": "openrouter",
    }
    defaults.update(kw)
    for k, v in defaults.items():
        monkeypatch.setenv(k, v)
    reset_config()


# ── Modelos de embedding ──────────────────────────────────────────────
class TestEmbeddingModels:
    def test_e5_tem_prefixos(self):
        perfil = EMBEDDING_MODELS["intfloat/multilingual-e5-large-instruct"]
        assert perfil["prefixo_query"] == "query: "
        assert perfil["prefixo_passage"] == "passage: "

    def test_neuralmind_sem_prefixos(self):
        perfil = EMBEDDING_MODELS["neuralmind/bert-base-portuguese-cased"]
        assert perfil["prefixo_query"] == ""
        assert perfil["lingua"] == "pt"

    def test_portulan_sem_prefixos(self):
        perfil = EMBEDDING_MODELS["PORTULAN/serafim-pt-small-100m-lingua-pt"]
        assert perfil["prefixo_query"] == ""
        assert perfil["lingua"] == "pt"

    def test_minilm_multilingual(self):
        perfil = EMBEDDING_MODELS["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]
        assert perfil["lingua"] == "multilingual"
        assert perfil["qualidade"] == "media"


# ── Config base ───────────────────────────────────────────────────────
class TestConfigBase:
    def test_valida_openrouter(self, monkeypatch):
        _env(monkeypatch)
        cfg = get_config()
        assert cfg.backend == "openrouter"
        assert cfg.is_free_model is True
        reset_config()

    def test_ollama_sem_chave(self, monkeypatch):
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

    def test_openrouter_free_frozenset(self):
        assert "openrouter/free" in FREE_MODELS
        assert "openrouter/auto" in FREE_MODELS
        assert isinstance(FREE_MODELS, frozenset)

    def test_chave_invalida_erro(self, monkeypatch):
        monkeypatch.setenv("BACKEND", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sem-chave")
        reset_config()
        with pytest.raises((ConfigError, Exception)):
            get_config()
        reset_config()


# ── RAG V7 ────────────────────────────────────────────────────────────
class TestRAGConfig:
    def test_rag_hibrido(self, monkeypatch):
        _env(monkeypatch, RAG_MODO="hibrido")
        cfg = get_config()
        assert cfg.rag_modo == "hibrido"
        reset_config()

    def test_rag_embedding_portulan(self, monkeypatch):
        _env(monkeypatch, RAG_EMBEDDING_MODELO="PORTULAN/serafim-pt-small-100m-lingua-pt")
        cfg = get_config()
        assert "PORTULAN" in cfg.rag_embedding_modelo
        perfil = cfg.embedding_perfil
        assert perfil["lingua"] == "pt"
        assert perfil["prefixo_query"] == ""
        reset_config()

    def test_rag_embedding_neuralmind(self, monkeypatch):
        _env(monkeypatch, RAG_EMBEDDING_MODELO="neuralmind/bert-base-portuguese-cased")
        cfg = get_config()
        assert "neuralmind" in cfg.rag_embedding_modelo
        reset_config()

    def test_rag_reranker_cohere(self, monkeypatch):
        _env(monkeypatch, RAG_RERANKER_BACKEND="cohere", COHERE_API_KEY="co-test-key")
        cfg = get_config()
        assert cfg.rag_reranker_backend == "cohere"
        assert cfg.usar_cohere_rerank is True
        reset_config()

    def test_rag_reranker_local(self, monkeypatch):
        _env(monkeypatch, RAG_RERANKER_BACKEND="local")
        cfg = get_config()
        assert cfg.rag_reranker_backend == "local"
        assert cfg.usar_cohere_rerank is False
        reset_config()

    def test_rag_top_k_n(self, monkeypatch):
        _env(monkeypatch, RAG_TOP_K="20", RAG_TOP_N="8")
        cfg = get_config()
        assert cfg.rag_top_k == 20
        assert cfg.rag_top_n == 8
        reset_config()


# ── API produção ──────────────────────────────────────────────────────
class TestAPIConfig:
    def test_secret_key_insegura(self, monkeypatch):
        _env(monkeypatch)
        cfg = get_config()
        # Default contém "muda_isto" → insegura
        assert cfg.api_secret_key_segura is False
        reset_config()

    def test_secret_key_segura(self, monkeypatch):
        _env(monkeypatch, API_SECRET_KEY="chave_super_segura_32_chars_producao_abc")
        cfg = get_config()
        assert cfg.api_secret_key_segura is True
        reset_config()

    def test_rate_limit(self, monkeypatch):
        _env(monkeypatch, API_RATE_LIMIT="50", API_RATE_LIMIT_BURST="15")
        cfg = get_config()
        assert cfg.api_rate_limit == 50
        assert cfg.api_rate_limit_burst == 15
        reset_config()

    def test_api_port(self, monkeypatch):
        _env(monkeypatch, API_PORT="9000")
        cfg = get_config()
        assert cfg.api_port == 9000
        reset_config()

    def test_jwt_expire(self, monkeypatch):
        _env(monkeypatch, API_ACCESS_TOKEN_EXPIRE_MINUTES="120")
        cfg = get_config()
        assert cfg.api_access_token_expire_minutes == 120
        reset_config()


# ── Observability ─────────────────────────────────────────────────────
class TestObservabilityConfig:
    def test_observability_enabled(self, monkeypatch):
        _env(monkeypatch, OBSERVABILITY_ENABLED="true")
        cfg = get_config()
        assert cfg.observability_enabled is True
        reset_config()

    def test_metrics_port(self, monkeypatch):
        _env(monkeypatch, METRICS_PORT="9091")
        cfg = get_config()
        assert cfg.metrics_port == 9091
        reset_config()

    def test_otel_service_name(self, monkeypatch):
        _env(monkeypatch, OTEL_SERVICE_NAME="tribunal-prod-v7")
        cfg = get_config()
        assert cfg.otel_service_name == "tribunal-prod-v7"
        reset_config()

    def test_otel_endpoint(self, monkeypatch):
        _env(monkeypatch, OTEL_EXPORTER_OTLP_ENDPOINT="http://jaeger:4317")
        cfg = get_config()
        assert "jaeger" in cfg.otel_exporter_otlp_endpoint
        reset_config()


# ── LangGraph ─────────────────────────────────────────────────────────
class TestOrquestracao:
    def test_imperativo_desactiva_langgraph(self, monkeypatch):
        _env(monkeypatch, ORQUESTRACAO="imperativo")
        cfg = get_config()
        assert cfg.orquestracao == "imperativo"
        assert cfg.usar_langgraph is False
        reset_config()

    def test_langgraph_depende_de_instalacao(self, monkeypatch):
        _env(monkeypatch, ORQUESTRACAO="langgraph")
        cfg = get_config()
        # usar_langgraph depende de langgraph estar instalado
        assert isinstance(cfg.usar_langgraph, bool)
        reset_config()
