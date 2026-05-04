"""
Motor RAG V5 — Híbrido BM25 + Embeddings
─────────────────────────────────────────
Modos:
  bm25    → apenas BM25 (sem dependências extra, default)
  hibrido → BM25 + sentence-transformers local
  api     → BM25 + embeddings via OpenRouter API

Metadata filtering por:
  - instância judicial (TIC, TRAB, TC_CIVEL, etc.)
  - tipo de diploma (lei, jurisprudencia, precedente)
  - diploma específico (CP, CPP, CC, CT, etc.)
"""
from __future__ import annotations

import math
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    NUMPY_OK = True
except ImportError:
    NUMPY_OK = False

try:
    from sentence_transformers import SentenceTransformer
    ST_OK = True
except ImportError:
    ST_OK = False


@dataclass
class Fragmento:
    fonte: str
    tipo: str           # "lei" | "jurisprudencia" | "precedente"
    titulo: str
    conteudo: str
    relevancia: float
    artigo: Optional[str] = None
    diploma: Optional[str] = None       # "CP", "CPP", "CC", etc.
    instancias: List[str] = field(default_factory=list)  # ["TIC","TCCR"]
    embedding: Optional[List[float]] = None


# Mapeamento diploma → instâncias relevantes
DIPLOMA_INSTANCIAS: Dict[str, List[str]] = {
    "CP":   ["TIC", "TCCR", "TCIC"],
    "CPP":  ["TIC", "TCCR", "TCIC"],
    "CC":   ["TC_CIVEL", "TFM"],
    "CPC":  ["TC_CIVEL", "TFM", "TCOM"],
    "CT":   ["TRAB"],
    "CPT":  ["TRAB"],
    "CRP":  ["TC", "TIC", "TCCR", "TAF"],
    "CPTA": ["TAF"],
    "CPPT": ["TAF"],
    "CIRE": ["TCOM"],
    "CSC":  ["TCOM"],
    "RGPTC": ["TFM"],
    "LPCJP": ["TFM"],
}

# Palavras-chave por diploma
DIPLOMA_KEYWORDS: Dict[str, List[str]] = {
    "CP": ["código penal", "codigopenal", "crime", "pena", "arguido", "ilícito"],
    "CPP": ["processo penal", "inquérito", "instrução", "julgamento penal"],
    "CC": ["código civil", "codigocivil", "contrato", "obrigação", "propriedade"],
    "CPC": ["processo civil", "execução", "penhora", "citação"],
    "CT": ["código trabalho", "codigotrabalho", "trabalhador", "empregador"],
    "CRP": ["constituição", "direitos fundamentais", "artigo", "república"],
    "CPTA": ["administrativo", "contencioso", "tribunal administrativo"],
    "CIRE": ["insolvência", "recuperação", "credor", "devedor"],
}

STOPWORDS = {
    "a","o","as","os","um","uma","de","do","da","dos","das","em","no","na",
    "nos","nas","por","para","com","sem","sob","que","se","é","são","foi",
    "ser","ter","ao","à","aos","às","e","ou","mas","nem","não","sim","já",
    "ainda","também","quando","onde","como","porque","pois","porém","contudo",
    "n","nº","art","artigo","alínea","número","parágrafo",
}


class MotorRAG:
    def __init__(
        self,
        pasta_raiz: Path,
        modo: str = "bm25",
        embedding_modelo: str = "intfloat/multilingual-e5-large",
    ) -> None:
        self.pasta_raiz = pasta_raiz
        self.modo = modo
        self.embedding_modelo_nome = embedding_modelo
        self._indice: List[Fragmento] = []
        self._doc_freq: Dict[str, int] = {}
        self._indexado = False
        self._st_model: Optional[Any] = None

        cache_dir = pasta_raiz / "src" / "cache" / "data"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = cache_dir / "rag_index_v5.pkl"
        self._docfreq_path = cache_dir / "rag_docfreq_v5.pkl"

        # Tentar carregar modelo de embeddings se necessário
        if modo in ("hibrido",) and ST_OK and NUMPY_OK:
            try:
                self._st_model = SentenceTransformer(embedding_modelo)
            except Exception:
                self.modo = "bm25"  # fallback graceful

    # ── Indexação ────────────────────────────────────────────────────
    def indexar(self, forcar: bool = False) -> int:
        if not forcar and self._index_path.exists():
            try:
                with open(self._index_path, "rb") as f:
                    self._indice = pickle.load(f)
                with open(self._docfreq_path, "rb") as f:
                    self._doc_freq = pickle.load(f)
                self._indexado = True
                return len(self._indice)
            except Exception:
                pass

        self._indice = []
        self._doc_freq = {}

        for pasta, tipo in [
            (self.pasta_raiz / "data" / "leis",           "lei"),
            (self.pasta_raiz / "data" / "jurisprudencia", "jurisprudencia"),
            (self.pasta_raiz / "data" / "precedentes",    "precedente"),
        ]:
            if pasta.exists():
                for f in sorted(pasta.glob("*.txt")):
                    self._indice.extend(self._processar_ficheiro(f, tipo))

        # IDF
        for frag in self._indice:
            tokens = set(self._tokenizar(frag.conteudo + " " + frag.titulo))
            for t in tokens:
                self._doc_freq[t] = self._doc_freq.get(t, 0) + 1

        # Embeddings (se modo hibrido e modelo disponível)
        if self.modo == "hibrido" and self._st_model is not None and NUMPY_OK:
            self._computar_embeddings()

        self._indexado = True
        self._persistir()
        return len(self._indice)

    def _computar_embeddings(self) -> None:
        textos = [f.conteudo[:512] for f in self._indice]
        try:
            embs = self._st_model.encode(textos, batch_size=32, show_progress_bar=False)
            for i, frag in enumerate(self._indice):
                frag.embedding = embs[i].tolist()
        except Exception:
            pass

    def _persistir(self) -> None:
        try:
            with open(self._index_path, "wb") as f:
                pickle.dump(self._indice, f)
            with open(self._docfreq_path, "wb") as f:
                pickle.dump(self._doc_freq, f)
        except Exception:
            pass

    def _detectar_diploma(self, nome: str, texto: str) -> Optional[str]:
        nome_upper = nome.upper().replace(" ", "").replace("_", "")
        for diploma, kws in DIPLOMA_KEYWORDS.items():
            if diploma in nome_upper:
                return diploma
            if any(kw in texto[:500].lower() for kw in kws):
                return diploma
        return None

    def _processar_ficheiro(self, path: Path, tipo: str) -> List[Fragmento]:
        try:
            texto = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

        nome = path.stem.replace("_", " ").replace("-", " ")
        diploma = self._detectar_diploma(nome, texto)
        instancias_relevantes = DIPLOMA_INSTANCIAS.get(diploma or "", [])

        partes = re.split(
            r"\n(?=(?:Artigo\s+\d+|Art\.?\s+\d+|ARTIGO\s+\d+|Secção|CAPÍTULO|Título))",
            texto, flags=re.IGNORECASE,
        )

        fragmentos: List[Fragmento] = []

        if len(partes) > 1:
            for parte in partes:
                parte = parte.strip()
                if len(parte) < 40:
                    continue
                m = re.match(r"(Art(?:igo)?\.?\s+\d+\.?[º°]?[A-Za-z]?)", parte, re.IGNORECASE)
                artigo = m.group(1) if m else None
                titulo = parte[:100].split("\n")[0].strip()
                fragmentos.append(Fragmento(
                    fonte=nome, tipo=tipo, titulo=titulo,
                    conteudo=parte[:2000], relevancia=0.0,
                    artigo=artigo, diploma=diploma,
                    instancias=instancias_relevantes,
                ))
        else:
            blocos = [b.strip() for b in texto.split("\n\n") if len(b.strip()) > 80]
            for i, bloco in enumerate(blocos[:80]):
                fragmentos.append(Fragmento(
                    fonte=nome, tipo=tipo,
                    titulo=f"{nome} — bloco {i+1}",
                    conteudo=bloco[:2000], relevancia=0.0,
                    diploma=diploma, instancias=instancias_relevantes,
                ))

        return fragmentos

    # ── Tokenização ──────────────────────────────────────────────────
    def _tokenizar(self, texto: str) -> List[str]:
        palavras = re.findall(
            r"\b[a-záàâãéêíóôõúçA-ZÁÀÂÃÉÊÍÓÔÕÚÇ]{3,}\b", texto.lower()
        )
        return [p for p in palavras if p not in STOPWORDS]

    # ── BM25 ─────────────────────────────────────────────────────────
    def _score_bm25(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        if not doc_tokens or not query_tokens:
            return 0.0
        k1, b, avg_len = 1.5, 0.75, 250
        doc_len = len(doc_tokens)
        freq: Dict[str, int] = {}
        for t in doc_tokens:
            freq[t] = freq.get(t, 0) + 1
        N = max(len(self._indice), 1)
        score = 0.0
        for token in set(query_tokens):
            if token in freq:
                tf = freq[token]
                df = self._doc_freq.get(token, 1)
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                score += idf * (tf * (k1 + 1)) / (
                    tf + k1 * (1 - b + b * doc_len / avg_len)
                )
        return score

    # ── Cosine similarity ─────────────────────────────────────────────
    def _cosine(self, a: List[float], b: List[float]) -> float:
        if not NUMPY_OK:
            return 0.0
        va, vb = np.array(a), np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom > 0 else 0.0

    # ── Pesquisa principal ───────────────────────────────────────────
    def pesquisar(
        self,
        consulta: str,
        n_resultados: int = 6,
        instancia: Optional[str] = None,
        tipo_filtro: Optional[str] = None,
        diploma_filtro: Optional[str] = None,
    ) -> List[Fragmento]:
        if not self._indexado:
            self.indexar()
        if not self._indice:
            return []

        query_tokens = self._tokenizar(consulta)
        if not query_tokens:
            return []

        # Embedding da query (se modo hibrido)
        query_emb: Optional[List[float]] = None
        if self.modo == "hibrido" and self._st_model is not None and NUMPY_OK:
            try:
                query_emb = self._st_model.encode([consulta[:512]])[0].tolist()
            except Exception:
                pass

        resultados: List[Fragmento] = []

        for frag in self._indice:
            # Metadata filtering
            if tipo_filtro and frag.tipo != tipo_filtro:
                continue
            if diploma_filtro and frag.diploma != diploma_filtro:
                continue
            if instancia and frag.instancias and instancia not in frag.instancias:
                # Permitir fragmentos sem instância definida (CRP, etc.)
                continue

            doc_tokens = self._tokenizar(frag.conteudo + " " + frag.titulo)
            bm25 = self._score_bm25(query_tokens, doc_tokens)

            # Score híbrido
            if self.modo == "hibrido" and query_emb and frag.embedding:
                sem = self._cosine(query_emb, frag.embedding)
                score = 0.6 * bm25 + 0.4 * sem * 10  # normalizar escala
            else:
                score = bm25

            if score > 0:
                resultados.append(Fragmento(
                    fonte=frag.fonte, tipo=frag.tipo, titulo=frag.titulo,
                    conteudo=frag.conteudo, relevancia=round(score, 4),
                    artigo=frag.artigo, diploma=frag.diploma,
                    instancias=frag.instancias,
                ))

        resultados.sort(key=lambda f: f.relevancia, reverse=True)
        return resultados[:n_resultados]

    # ── Formatação do contexto ────────────────────────────────────────
    def formatar_contexto(
        self, fragmentos: List[Fragmento], max_chars: int = 3000
    ) -> str:
        if not fragmentos:
            return ""
        linhas = ["=== CONTEXTO JURÍDICO (RAG) ===\n"]
        total = 0
        for f in fragmentos:
            diploma_tag = f" [{f.diploma}]" if f.diploma else ""
            artigo_tag = f" — {f.artigo}" if f.artigo else ""
            bloco = (
                f"[{f.tipo.upper()}{diploma_tag}] {f.fonte}{artigo_tag}\n"
                f"Relevância: {f.relevancia:.3f}\n"
                f"{f.conteudo[:600]}\n{'─'*40}\n"
            )
            if total + len(bloco) > max_chars:
                break
            linhas.append(bloco)
            total += len(bloco)
        linhas.append("=== FIM CONTEXTO ===\n")
        return "\n".join(linhas)

    def tem_dados(self) -> bool:
        if not self._indexado:
            self.indexar()
        return bool(self._indice)

    def estatisticas(self) -> Dict:
        if not self._indexado:
            self.indexar()
        return {
            "total": len(self._indice),
            "leis": sum(1 for f in self._indice if f.tipo == "lei"),
            "jurisprudencia": sum(1 for f in self._indice if f.tipo == "jurisprudencia"),
            "precedentes": sum(1 for f in self._indice if f.tipo == "precedente"),
            "diplomas": list({f.diploma for f in self._indice if f.diploma}),
            "fontes": list({f.fonte for f in self._indice}),
            "modo": self.modo,
            "embeddings": self._st_model is not None,
        }

    def recarregar(self) -> int:
        self._indexado = False
        self._index_path.unlink(missing_ok=True)
        self._docfreq_path.unlink(missing_ok=True)
        return self.indexar(forcar=True)
