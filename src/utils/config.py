"""
Configuração V7 — tipagem 100% forte, PORTULAN, Cohere, JWT, Observability.
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class ConfigError(Exception):
    pass

FREE_MODELS: frozenset[str] = frozenset({
    "openrouter/free", "openrouter/auto",
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-r1:free",
    "google/gemini-2.0-flash-exp:free",
    "mistralai/mistral-7b-instruct:free",
    "qwen/qwen-2.5-72b-instruct:free",
})

PAID_MODELS: dict[str, tuple[float, float]] = {
    "google/gemini-2.0-flash-001":       (0.10, 0.40),
    "google/gemini-2.5-flash":           (0.15, 0.60),
    "google/gemini-2.5-pro":             (1.25, 5.00),
    "anthropic/claude-haiku-4-5":        (1.00, 5.00),
    "anthropic/claude-sonnet-4.6":       (3.00, 15.00),
    "anthropic/claude-opus-4.6":         (15.00, 75.00),
    "openai/gpt-4.1-mini":               (0.40, 1.60),
    "openai/gpt-4.1":                    (2.00, 8.00),
    "deepseek/deepseek-chat-v3-0324":    (0.27, 1.10),
    "meta-llama/llama-3.3-70b-instruct": (0.12, 0.30),
}

EMBEDDING_MODELS: dict[str, dict[str, str]] = {
    "intfloat/multilingual-e5-large-instruct": {
        "prefixo_query": "query: ", "prefixo_passage": "passage: ",
        "lingua": "multilingual", "tamanho": "560MB", "qualidade": "muito_alta",
    },
    "intfloat/multilingual-e5-base": {
        "prefixo_query": "query: ", "prefixo_passage": "passage: ",
        "lingua": "multilingual", "tamanho": "280MB", "qualidade": "alta",
    },
    "neuralmind/bert-base-portuguese-cased": {
        "prefixo_query": "", "prefixo_passage": "",
        "lingua": "pt", "tamanho": "440MB", "qualidade": "alta_pt",
    },
    "PORTULAN/serafim-pt-small-100m-lingua-pt": {
        "prefixo_query": "", "prefixo_passage": "",
        "lingua": "pt", "tamanho": "400MB", "qualidade": "alta_pt",
    },
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
        "prefixo_query": "", "prefixo_passage": "",
        "lingua": "multilingual", "tamanho": "118MB", "qualidade": "media",
    },
}

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        case_sensitive=False, extra="ignore",
    )
    backend: Literal["openrouter", "ollama"] = "openrouter"
    openrouter_api_key: str = "sem-chave"
    modelo: str = "openrouter/free"
    ollama_url: str = "http://localhost:11434"
    ollama_modelo: str = "llama3.3:70b"
    rag_modo: Literal["bm25", "hibrido", "api"] = "bm25"
    rag_embedding_modelo: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    rag_reranking: bool = True
    rag_reranker_backend: Literal["local", "cohere"] = "local"
    rag_reranker_modelo: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cohere_api_key: str = ""
    rag_top_k: int = 15
    rag_top_n: int = 6
    orquestracao: Literal["langgraph", "imperativo"] = "langgraph"
    guardar_atas: bool = True
    anonimizar_entidades: bool = True
    cache_enabled: bool = True
    paralelismo: bool = False
    modo_economico: bool = True
    historico_enabled: bool = True
    exportar_pdf: bool = True
    consistencia_check: bool = True
    contraditorio_enabled: bool = True
    multilingue_enabled: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_secret_key: str = "muda_isto_para_chave_segura_em_producao_min32chars"
    api_access_token_expire_minutes: int = 60
    api_rate_limit: int = 30
    api_rate_limit_burst: int = 10
    observability_enabled: bool = True
    metrics_port: int = 9090
    otel_service_name: str = "tribunal-ia-v7"
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    max_retries: int = 5
    request_timeout: int = 180
    log_level: str = "INFO"
    pasta_leis: Path = Path("data/leis")
    pasta_jurisprudencia: Path = Path("data/jurisprudencia")
    pasta_precedentes: Path = Path("data/precedentes")
    pasta_tedh: Path = Path("data/tedh")
    pasta_atas: Path = Path("output_atas")
    pasta_cache: Path = Path("src/cache/data")
    pasta_historico: Path = Path("src/historico/data")

    @property
    def usar_ollama(self) -> bool:
        return self.backend == "ollama"

    @property
    def modelo_activo(self) -> str:
        return self.ollama_modelo if self.usar_ollama else self.modelo

    @property
    def is_free_model(self) -> bool:
        if self.usar_ollama:
            return True
        m = self.modelo.lower()
        return (self.modelo in FREE_MODELS or m.endswith(":free")
                or "free" in m or m.startswith("openrouter/"))

    @property
    def custo_por_token(self) -> tuple[float, float]:
        if self.is_free_model:
            return (0.0, 0.0)
        return PAID_MODELS.get(self.modelo, (1.0, 3.0))

    @property
    def usar_langgraph(self) -> bool:
        if self.orquestracao != "langgraph":
            return False
        try:
            import langgraph  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def usar_cohere_rerank(self) -> bool:
        return (self.rag_reranking and self.rag_reranker_backend == "cohere"
                and bool(self.cohere_api_key))

    @property
    def embedding_perfil(self) -> dict[str, str]:
        return EMBEDDING_MODELS.get(
            self.rag_embedding_modelo,
            {"prefixo_query": "", "prefixo_passage": "", "lingua": "multilingual",
             "tamanho": "?", "qualidade": "desconhecida"},
        )

    @property
    def api_secret_key_segura(self) -> bool:
        return len(self.api_secret_key) >= 32 and "muda_isto" not in self.api_secret_key

    @field_validator("api_secret_key")
    @classmethod
    def validar_secret_key(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def validate_and_setup(self) -> "Settings":
        if self.backend == "openrouter":
            k = self.openrouter_api_key
            if not k or k in ("sem-chave", "COLA_AQUI_A_TUA_CHAVE") or "cola" in k.lower():
                raise ValueError(
                    "OPENROUTER_API_KEY não configurada. Edita .env — "
                    "https://openrouter.ai/keys ou usa BACKEND=ollama."
                )
        for p in [
            self.pasta_leis, self.pasta_jurisprudencia, self.pasta_precedentes,
            self.pasta_tedh, self.pasta_atas, self.pasta_cache,
            self.pasta_historico, Path("logs"),
        ]:
            p.mkdir(parents=True, exist_ok=True)
        return self

_settings: Optional[Settings] = None

def get_config() -> Settings:
    global _settings
    if _settings is None:
        try:
            _settings = Settings()  # type: ignore[call-arg]
        except Exception as e:
            raise ConfigError(str(e)) from e
    return _settings

def reset_config() -> None:
    global _settings
    _settings = None
