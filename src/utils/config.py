"""
Configuração centralizada V5 — Pydantic Settings v2.
Suporta: OpenRouter (cloud) + Ollama (local/soberania de dados).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(Exception):
    pass


FREE_MODELS = {
    "openrouter/auto",
    "openrouter/free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-r1:free",
    "google/gemini-2.0-flash-exp:free",
    "mistralai/mistral-7b-instruct:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
}

PAID_MODELS: dict[str, tuple[float, float]] = {
    "google/gemini-2.0-flash-001":       (0.10, 0.40),
    "google/gemini-2.5-flash":           (0.15, 0.60),
    "google/gemini-2.5-pro":             (1.25, 5.00),
    "anthropic/claude-haiku-4-5":        (1.00, 5.00),
    "anthropic/claude-sonnet-4.6":       (3.00, 15.00),
    "anthropic/claude-opus-4.6":         (15.00, 75.00),
    "openai/gpt-4.1-nano":               (0.10, 0.40),
    "openai/gpt-4.1-mini":               (0.40, 1.60),
    "openai/gpt-4.1":                    (2.00, 8.00),
    "deepseek/deepseek-chat-v3-0324":    (0.27, 1.10),
    "deepseek/deepseek-r1":              (0.55, 2.19),
    "meta-llama/llama-3.3-70b-instruct": (0.12, 0.30),
    "mistralai/mistral-small-24b":       (0.10, 0.30),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Backend
    backend: Literal["openrouter", "ollama"] = "openrouter"

    # OpenRouter
    openrouter_api_key: str = "sem-chave"
    modelo: str = "openrouter/free"

    # Ollama
    ollama_url: str = "http://localhost:11434"
    ollama_modelo: str = "llama3.3:70b"

    # RAG
    rag_modo: Literal["bm25", "hibrido", "api"] = "bm25"
    rag_embedding_modelo: str = "intfloat/multilingual-e5-large"

    # Rede
    max_retries: int = 5
    request_timeout: int = 180

    # Funcionalidades
    guardar_atas: bool = True
    anonimizar_entidades: bool = True
    cache_enabled: bool = True
    paralelismo: bool = False
    modo_economico: bool = True
    historico_enabled: bool = True
    exportar_pdf: bool = True
    consistencia_check: bool = True

    # Pastas
    pasta_leis: Path = Path("data/leis")
    pasta_jurisprudencia: Path = Path("data/jurisprudencia")
    pasta_precedentes: Path = Path("data/precedentes")
    pasta_atas: Path = Path("output_atas")
    pasta_cache: Path = Path("src/cache/data")
    pasta_historico: Path = Path("src/historico/data")
    log_level: str = "INFO"

    # Computed
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
        return (
            self.modelo.endswith(":free")
            or self.modelo in FREE_MODELS
            or "free" in m
            or m.startswith("openrouter/")
        )

    @property
    def custo_por_token(self) -> tuple[float, float]:
        if self.is_free_model:
            return (0.0, 0.0)
        return PAID_MODELS.get(self.modelo, (1.0, 3.0))

    @field_validator("openrouter_api_key")
    @classmethod
    def validate_key(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def validate_and_create(self) -> "Settings":
        # Validar chave só se usar OpenRouter
        if self.backend == "openrouter":
            k = self.openrouter_api_key
            if not k or k in ("sem-chave", "COLA_AQUI_A_TUA_CHAVE") or "cola" in k.lower():
                raise ValueError(
                    "OPENROUTER_API_KEY não configurada. "
                    "Edita .env com a tua chave de https://openrouter.ai/keys "
                    "ou muda BACKEND=ollama para usar modelo local."
                )
        # Criar pastas
        for p in [
            self.pasta_leis, self.pasta_jurisprudencia, self.pasta_precedentes,
            self.pasta_atas, self.pasta_cache, self.pasta_historico,
            Path("logs"),
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
