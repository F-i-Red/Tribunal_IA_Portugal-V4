"""
Configuração V6 — Pydantic Settings v2.
Novas opções: RAG híbrido, reranking, LangGraph, FastAPI, multi-idioma.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(Exception):
    pass


FREE_MODELS = {
    "openrouter/free",
    "openrouter/auto",
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
    "openai/gpt-4.1-mini":               (0.40, 1.60),
    "openai/gpt-4.1":                    (2.00, 8.00),
    "deepseek/deepseek-chat-v3-0324":    (0.27, 1.10),
    "meta-llama/llama-3.3-70b-instruct": (0.12, 0.30),
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

    # ── RAG V6 ───────────────────────────────────────────────
    rag_modo: Literal["bm25", "hibrido", "api"] = "hibrido"
    rag_embedding_modelo: str = "intfloat/multilingual-e5-large-instruct"
    rag_reranking: bool = True
    rag_reranker_modelo: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rag_top_k: int = 15   # candidatos antes do reranking
    rag_top_n: int = 6    # resultado final após reranking

    # ── Orquestração ─────────────────────────────────────────
    orquestracao: Literal["langgraph", "imperativo"] = "langgraph"

    # ── Funcionalidades ───────────────────────────────────────
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

    # ── API REST ──────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_secret_key: str = "muda_isto_em_producao"

    # ── Rede ──────────────────────────────────────────────────
    max_retries: int = 5
    request_timeout: int = 180
    log_level: str = "INFO"

    # ── Pastas ────────────────────────────────────────────────
    pasta_leis: Path = Path("data/leis")
    pasta_jurisprudencia: Path = Path("data/jurisprudencia")
    pasta_precedentes: Path = Path("data/precedentes")
    pasta_tedh: Path = Path("data/tedh")
    pasta_atas: Path = Path("output_atas")
    pasta_cache: Path = Path("src/cache/data")
    pasta_historico: Path = Path("src/historico/data")

    # ── Computed ──────────────────────────────────────────────
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
            self.modelo in FREE_MODELS
            or m.endswith(":free")
            or "free" in m
            or m.startswith("openrouter/")
        )

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

    @model_validator(mode="after")
    def validate_and_setup(self) -> "Settings":
        if self.backend == "openrouter":
            k = self.openrouter_api_key
            if not k or k in ("sem-chave", "COLA_AQUI_A_TUA_CHAVE") or "cola" in k.lower():
                raise ValueError(
                    "OPENROUTER_API_KEY não configurada. "
                    "Edita .env com a tua chave de https://openrouter.ai/keys "
                    "ou usa BACKEND=ollama para modelo local."
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
