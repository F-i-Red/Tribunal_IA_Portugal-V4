"""
Cache semântico por hash de prompt. V4 — thread-safe, persistente.
"""
from __future__ import annotations

import hashlib
import pickle
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from ..utils.config import get_config


@dataclass
class CacheEntry:
    prompt_hash: str
    response: str
    model: str
    timestamp: float
    tokens_input: int
    tokens_output: int
    cost_usd: float


class SemanticCache:
    def __init__(self, pasta_cache: Optional[Path] = None) -> None:
        cfg = get_config()
        self.pasta = pasta_cache or cfg.pasta_cache
        self.pasta.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._carregar()

    def _carregar(self) -> None:
        path = self.pasta / "cache_index.pkl"
        if path.exists():
            try:
                with open(path, "rb") as f:
                    self._index = pickle.load(f)
            except Exception:
                self._index = {}

    def _guardar(self) -> None:
        path = self.pasta / "cache_index.pkl"
        try:
            with open(path, "wb") as f:
                pickle.dump(self._index, f)
        except Exception:
            pass

    def _hash(self, prompt: str, system: Optional[str], model: str) -> str:
        texto = f"{model}:{system or ''}:{prompt}"
        return hashlib.sha256(texto.encode()).hexdigest()[:20]

    def get(self, prompt: str, system: Optional[str] = None, model: str = "") -> Optional[CacheEntry]:
        if not get_config().cache_enabled:
            return None
        h = self._hash(prompt, system, model)
        with self._lock:
            return self._index.get(h)

    def put(
        self,
        prompt: str,
        response: str,
        system: Optional[str] = None,
        model: str = "",
        tokens_input: int = 0,
        tokens_output: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        if not get_config().cache_enabled:
            return
        h = self._hash(prompt, system, model)
        entry = CacheEntry(
            prompt_hash=h,
            response=response,
            model=model,
            timestamp=time.time(),
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost_usd,
        )
        with self._lock:
            self._index[h] = entry
            self._guardar()

    def estatisticas(self) -> Dict:
        with self._lock:
            total = len(self._index)
            cost = sum(e.cost_usd for e in self._index.values())
        return {
            "entradas": total,
            "custo_total_usd": round(cost, 6),
            "poupanca_estimada_usd": round(cost * 0.8, 6),
        }

    def limpar(self, max_idade_dias: int = 30) -> int:
        agora = time.time()
        limite = max_idade_dias * 86400
        with self._lock:
            chaves = [h for h, e in self._index.items() if agora - e.timestamp > limite]
            for h in chaves:
                del self._index[h]
            if chaves:
                self._guardar()
        return len(chaves)


_cache_instance: Optional[SemanticCache] = None
_cache_lock = threading.Lock()


def get_cache() -> SemanticCache:
    global _cache_instance
    with _cache_lock:
        if _cache_instance is None:
            _cache_instance = SemanticCache()
    return _cache_instance


def limpar_cache(dias: int = 30) -> int:
    return get_cache().limpar(dias)
