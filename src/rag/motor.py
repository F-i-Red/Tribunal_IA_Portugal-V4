"""
Motor RAG V6 — Híbrido BM25 + Embeddings + Reranking
══════════════════════════════════════════════════════
Pipeline:
  1. BM25  → top_k candidatos (rápido, lexical)
  2. Embeddings → score semântico por cosine similarity
  3. Fusão RRF (Reciprocal Rank Fusion) BM25 + semântico
  4. Cross-encoder Reranker → re-ordena top_k → devolve top_n

Modelos recomendados para Português:
  Embeddings:
    intfloat/multilingual-e5-large-instruct  (melhor, 560MB)
    intfloat/multilingual-e5-base            (equilibrado, 280MB)
    neuralmind/bert-base-portuguese-cased    (PT-específico)
    sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (leve, 118MB)
  Reranker:
    cross-encoder/ms-marco-MiniLM-L-6-v2    (universal, funciona em PT)
    cross-encoder/mmarco-mMiniLMv2-L12-H384 (multilingual, melhor para PT)
"""
from __future__ import annotations

import math
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Embeddings — optional
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import numpy as np
    ST_OK = True
except ImportError:
    ST_OK = False

# Tipos
try:
    from numpy.typing import NDArray
    FloatArray = NDArray[np.float32]
except Exception:
    FloatArray = list  # type: ignore


@dataclass
class Fragmento:
    fonte: str
    tipo: str
    titulo: str
    conteudo: str
    relevancia: float
    artigo: Optional[str] = None
    diploma: Optional[str] = None
    instancias: List[str] = field(default_factory=list)
    lingua: str = "pt"                      # "pt" | "en" (TEDH)
    embedding: Optional[List[float]] = None
    bm25_score: float = 0.0
    sem_score: float = 0.0
    rerank_score: float = 0.0


# Mapeamento diploma → instâncias relevantes
DIPLOMA_INSTANCIAS: Dict[str, List[str]] = {
    "CP":    ["TIC", "TCCR", "TCIC"],
    "CPP":   ["TIC", "TCCR", "TCIC"],
    "CC":    ["TC_CIVEL", "TFM"],
    "CPC":   ["TC_CIVEL", "TFM", "TCOM"],
    "CT":    ["TRAB"],
    "CPT":   ["TRAB"],
    "CRP":   ["TC", "TIC", "TCCR", "TAF"],
    "CPTA":  ["TAF"],
    "CPPT":  ["TAF"],
    "CIRE":  ["TCOM"],
    "CSC":   ["TCOM"],
    "RGPTC": ["TFM"],
    "LPCJP": ["TFM"],
    "ECHR":  [],  # Jurisprudência TEDH — todas as instâncias
    "TEDH":  [],
}

DIPLOMA_KEYWORDS: Dict[str, List[str]] = {
    "CP":   ["código penal", "crime", "arguido", "pena", "ilícito"],
    "CPP":  ["processo penal", "inquérito", "instrução criminal"],
    "CC":   ["código civil", "contrato", "obrigação", "propriedade"],
    "CPC":  ["processo civil", "execução", "penhora"],
    "CT":   ["código trabalho", "trabalhador", "empregador", "despedimento"],
    "CRP":  ["constituição", "direitos fundamentais", "república portuguesa"],
    "CPTA": ["administrativo", "contencioso administrativo"],
    "CIRE": ["insolvência", "recuperação", "credor", "devedor"],
    "ECHR": ["echr", "european court", "human rights", "article", "convention"],
    "TEDH": ["tedh", "tribunal europeu", "direitos humanos", "convenção europeia"],
}

STOPWORDS_PT = {
    "a","o","as","os","um","uma","de","do","da","dos","das","em","no","na",
    "nos","nas","por","para","com","sem","sob","que","se","é","são","foi",
    "ser","ter","ao","à","aos","às","e","ou","mas","nem","não","sim","já",
    "ainda","também","quando","onde","como","porque","pois","porém","contudo",
    "n","nº","art","artigo","alínea","número","parágrafo",
}

STOPWORDS_EN = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","as","is","was","are","were","be","been","being","have",
    "has","had","do","does","did","will","would","could","should","may",
    "might","shall","can","this","that","these","those","it","its","they",
    "them","their","we","our","you","your","he","she","his","her",
}


class MotorRAG:
    def __init__(
        self,
        pasta_raiz: Path,
        modo: str = "hibrido",
        embedding_modelo: str = "intfloat/multilingual-e5-large-instruct",
        reranker_modelo: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        usar_reranking: bool = True,
        top_k: int = 15,
        top_n: int = 6,
    ) -> None:
        self.pasta_raiz = pasta_raiz
        self.modo = modo if ST_OK else "bm25"
        self.embedding_modelo_nome = embedding_modelo
        self.reranker_modelo_nome = reranker_modelo
        self.usar_reranking = usar_reranking and ST_OK
        self.top_k = top_k
        self.top_n = top_n

        self._indice: List[Fragmento] = []
        self._doc_freq: Dict[str, int] = {}
        self._indexado = False
        self._embed_model: Optional[SentenceTransformer] = None
        self._rerank_model: Optional[CrossEncoder] = None

        cache_dir = pasta_raiz / "src" / "cache" / "data"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = cache_dir / "rag_index_v6.pkl"
        self._docfreq_path = cache_dir / "rag_docfreq_v6.pkl"

        # Carregar modelos se modo híbrido
        if self.modo == "hibrido" and ST_OK:
            self._carregar_modelos()

    # ── Carregamento de modelos ───────────────────────────────────────
    def _carregar_modelos(self) -> None:
        """Carrega embeddings e reranker — lazy loading."""
        if self._embed_model is None:
            try:
                self._embed_model = SentenceTransformer(
                    self.embedding_modelo_nome,
                    device="cpu",
                )
            except Exception as e:
                print(f"[RAG] Embeddings não carregados: {e}. Usando BM25.")
                self.modo = "bm25"
                return

        if self.usar_reranking and self._rerank_model is None:
            try:
                self._rerank_model = CrossEncoder(
                    self.reranker_modelo_nome,
                    device="cpu",
                    max_length=512,
                )
            except Exception as e:
                print(f"[RAG] Reranker não carregado: {e}. Reranking desactivado.")
                self.usar_reranking = False

    # ── Indexação ─────────────────────────────────────────────────────
    def indexar(self, forcar: bool = False) -> int:
        if not forcar and self._index_path.exists():
            try:
                with open(self._index_path, "rb") as f:
                    self._indice = pickle.load(f)
                with open(self._docfreq_path, "rb") as f:
                    self._doc_freq = pickle.load(f)
                self._indexado = True
                # Verificar se embeddings já estão calculados
                tem_embeddings = any(f.embedding for f in self._indice)
                if self.modo == "hibrido" and not tem_embeddings and ST_OK:
                    self._carregar_modelos()
                    if self._embed_model:
                        self._computar_embeddings()
                        self._persistir()
                return len(self._indice)
            except Exception:
                pass

        self._indice = []
        self._doc_freq = {}

        # Indexar todas as pastas
        pastas = [
            (self.pasta_raiz / "data" / "leis",           "lei",           "pt"),
            (self.pasta_raiz / "data" / "jurisprudencia", "jurisprudencia","pt"),
            (self.pasta_raiz / "data" / "precedentes",    "precedente",    "pt"),
            (self.pasta_raiz / "data" / "tedh",           "tedh",          "en"),
        ]
        for pasta, tipo, lingua in pastas:
            if pasta.exists():
                for f in sorted(pasta.glob("*.txt")):
                    self._indice.extend(self._processar_ficheiro(f, tipo, lingua))

        # IDF
        for frag in self._indice:
            tokens = set(self._tokenizar(frag.conteudo + " " + frag.titulo, frag.lingua))
            for t in tokens:
                self._doc_freq[t] = self._doc_freq.get(t, 0) + 1

        # Embeddings
        if self.modo == "hibrido" and ST_OK:
            self._carregar_modelos()
            if self._embed_model:
                self._computar_embeddings()

        self._indexado = True
        self._persistir()
        return len(self._indice)

    def _computar_embeddings(self) -> None:
        if not self._embed_model:
            return
        # Prefixo para modelos E5 (melhora qualidade)
        prefixo = ""
        nome = self.embedding_modelo_nome.lower()
        if "e5" in nome:
            prefixo = "passage: "

        textos = [prefixo + f.conteudo[:512] for f in self._indice]
        try:
            embs = self._embed_model.encode(
                textos, batch_size=32, show_progress_bar=False,
                normalize_embeddings=True,
            )
            for i, frag in enumerate(self._indice):
                frag.embedding = embs[i].tolist()
        except Exception as e:
            print(f"[RAG] Erro ao computar embeddings: {e}")

    def _persistir(self) -> None:
        try:
            with open(self._index_path, "wb") as f:
                pickle.dump(self._indice, f)
            with open(self._docfreq_path, "wb") as f:
                pickle.dump(self._doc_freq, f)
        except Exception:
            pass

    def _detectar_diploma(self, nome: str, texto: str) -> Optional[str]:
        nome_u = nome.upper().replace(" ", "").replace("_", "")
        for diploma, kws in DIPLOMA_KEYWORDS.items():
            if diploma in nome_u:
                return diploma
            if any(kw in texto[:500].lower() for kw in kws):
                return diploma
        return None

    def _processar_ficheiro(
        self, path: Path, tipo: str, lingua: str = "pt"
    ) -> List[Fragmento]:
        try:
            texto = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

        nome = path.stem.replace("_", " ").replace("-", " ")
        diploma = self._detectar_diploma(nome, texto)
        instancias = DIPLOMA_INSTANCIAS.get(diploma or "", [])

        # Dividir por artigos ou blocos
        partes = re.split(
            r"\n(?=(?:Artigo\s+\d+|Art\.?\s+\d+|ARTIGO\s+\d+|Article\s+\d+|§\s*\d+|CAPÍTULO|CHAPTER))",
            texto, flags=re.IGNORECASE,
        )
        fragmentos: List[Fragmento] = []

        if len(partes) > 1:
            for parte in partes:
                parte = parte.strip()
                if len(parte) < 40:
                    continue
                m = re.match(r"(Art(?:igo|icle)?\.?\s+\d+\.?[º°]?[A-Za-z]?)", parte, re.IGNORECASE)
                artigo = m.group(1) if m else None
                titulo = parte[:100].split("\n")[0].strip()
                fragmentos.append(Fragmento(
                    fonte=nome, tipo=tipo, titulo=titulo,
                    conteudo=parte[:2000], relevancia=0.0,
                    artigo=artigo, diploma=diploma,
                    instancias=instancias, lingua=lingua,
                ))
        else:
            blocos = [b.strip() for b in texto.split("\n\n") if len(b.strip()) > 80]
            for i, bloco in enumerate(blocos[:100]):
                fragmentos.append(Fragmento(
                    fonte=nome, tipo=tipo,
                    titulo=f"{nome} — bloco {i+1}",
                    conteudo=bloco[:2000], relevancia=0.0,
                    diploma=diploma, instancias=instancias, lingua=lingua,
                ))
        return fragmentos

    # ── Tokenização multilingual ──────────────────────────────────────
    def _tokenizar(self, texto: str, lingua: str = "pt") -> List[str]:
        palavras = re.findall(
            r"\b[a-záàâãéêíóôõúçA-ZÁÀÂÃÉÊÍÓÔÕÚÇ\w]{3,}\b", texto.lower()
        )
        stopwords = STOPWORDS_EN if lingua == "en" else STOPWORDS_PT
        return [p for p in palavras if p not in stopwords]

    # ── BM25 ──────────────────────────────────────────────────────────
    def _score_bm25(
        self, query_tokens: List[str], frag: Fragmento
    ) -> float:
        doc_tokens = self._tokenizar(
            frag.conteudo + " " + frag.titulo, frag.lingua
        )
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
        if not ST_OK:
            return 0.0
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom > 0 else 0.0

    # ── RRF — Reciprocal Rank Fusion ─────────────────────────────────
    @staticmethod
    def _rrf(bm25_rank: int, sem_rank: int, k: int = 60) -> float:
        return 1.0 / (k + bm25_rank) + 1.0 / (k + sem_rank)

    # ── Pesquisa principal ────────────────────────────────────────────
    def pesquisar(
        self,
        consulta: str,
        n_resultados: Optional[int] = None,
        instancia: Optional[str] = None,
        tipo_filtro: Optional[str] = None,
        diploma_filtro: Optional[str] = None,
        lingua_filtro: Optional[str] = None,
    ) -> List[Fragmento]:
        if not self._indexado:
            self.indexar()
        if not self._indice:
            return []

        top_k = self.top_k
        top_n = n_resultados or self.top_n

        # Filtrar candidatos
        candidatos = [
            f for f in self._indice
            if (not tipo_filtro or f.tipo == tipo_filtro)
            and (not diploma_filtro or f.diploma == diploma_filtro)
            and (not instancia or not f.instancias or instancia in f.instancias)
            and (not lingua_filtro or f.lingua == lingua_filtro)
        ]
        if not candidatos:
            return []

        # ── Fase 1: BM25 ──────────────────────────────────────────────
        query_tokens_pt = self._tokenizar(consulta, "pt")
        query_tokens_en = self._tokenizar(consulta, "en")

        bm25_scores: List[Tuple[float, Fragmento]] = []
        for frag in candidatos:
            qt = query_tokens_en if frag.lingua == "en" else query_tokens_pt
            s = self._score_bm25(qt, frag)
            if s > 0:
                bm25_scores.append((s, frag))
        bm25_scores.sort(key=lambda x: x[0], reverse=True)
        bm25_top = bm25_scores[:top_k * 2]

        if not bm25_top:
            return []

        # ── Fase 2: Embeddings semânticos ────────────────────────────
        if self.modo == "hibrido" and self._embed_model and ST_OK:
            nome_emb = self.embedding_modelo_nome.lower()
            prefixo_q = "query: " if "e5" in nome_emb else ""
            try:
                q_emb = self._embed_model.encode(
                    [prefixo_q + consulta[:512]],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )[0].tolist()

                sem_scores: List[Tuple[float, Fragmento]] = []
                for _, frag in bm25_top:
                    if frag.embedding:
                        s = self._cosine(q_emb, frag.embedding)
                        sem_scores.append((s, frag))
                    else:
                        sem_scores.append((0.0, frag))
                sem_scores.sort(key=lambda x: x[0], reverse=True)

                # RRF fusion
                bm25_rank = {id(f): i + 1 for i, (_, f) in enumerate(bm25_top)}
                sem_rank  = {id(f): i + 1 for i, (_, f) in enumerate(sem_scores)}

                rrf_scores: List[Tuple[float, Fragmento]] = []
                frags_vistos = set()
                for _, frag in bm25_top:
                    fid = id(frag)
                    if fid not in frags_vistos:
                        frags_vistos.add(fid)
                        rrf = self._rrf(
                            bm25_rank.get(fid, top_k * 2),
                            sem_rank.get(fid, top_k * 2),
                        )
                        rrf_scores.append((rrf, frag))
                rrf_scores.sort(key=lambda x: x[0], reverse=True)
                candidatos_reranking = rrf_scores[:top_k]
            except Exception:
                candidatos_reranking = [(s, f) for s, f in bm25_top[:top_k]]
        else:
            candidatos_reranking = [(s, f) for s, f in bm25_top[:top_k]]

        # ── Fase 3: Cross-encoder Reranking ──────────────────────────
        if self.usar_reranking and self._rerank_model and len(candidatos_reranking) > 1:
            try:
                pares = [
                    (consulta[:256], f.conteudo[:400])
                    for _, f in candidatos_reranking
                ]
                rerank_scores = self._rerank_model.predict(pares)
                reranked = sorted(
                    zip(rerank_scores, [f for _, f in candidatos_reranking]),
                    key=lambda x: x[0], reverse=True,
                )
                resultado_final = []
                for i, (score, frag) in enumerate(reranked[:top_n]):
                    resultado_final.append(Fragmento(
                        fonte=frag.fonte, tipo=frag.tipo, titulo=frag.titulo,
                        conteudo=frag.conteudo, artigo=frag.artigo,
                        diploma=frag.diploma, instancias=frag.instancias,
                        lingua=frag.lingua,
                        relevancia=round(float(score), 4),
                        rerank_score=round(float(score), 4),
                    ))
                return resultado_final
            except Exception as e:
                print(f"[RAG] Reranking falhou: {e}")

        # Fallback sem reranking
        resultado = []
        for i, (score, frag) in enumerate(candidatos_reranking[:top_n]):
            resultado.append(Fragmento(
                fonte=frag.fonte, tipo=frag.tipo, titulo=frag.titulo,
                conteudo=frag.conteudo, artigo=frag.artigo,
                diploma=frag.diploma, instancias=frag.instancias,
                lingua=frag.lingua, relevancia=round(score, 4),
            ))
        return resultado

    # ── Formatação ────────────────────────────────────────────────────
    def formatar_contexto(
        self,
        fragmentos: List[Fragmento],
        max_chars: int = 3500,
        incluir_tedh: bool = True,
    ) -> str:
        if not fragmentos:
            return ""
        pt_frags = [f for f in fragmentos if f.lingua == "pt"]
        en_frags = [f for f in fragmentos if f.lingua == "en" and incluir_tedh]

        linhas = ["=== CONTEXTO JURÍDICO (RAG V6) ===\n"]
        total = 0

        for f in pt_frags:
            diploma_tag = f" [{f.diploma}]" if f.diploma else ""
            artigo_tag = f" — {f.artigo}" if f.artigo else ""
            modo = "🔀 Híbrido" if f.rerank_score else "BM25"
            bloco = (
                f"[{f.tipo.upper()}{diploma_tag}] {f.fonte}{artigo_tag} "
                f"(rel={f.relevancia:.3f} {modo})\n"
                f"{f.conteudo[:600]}\n{'─'*40}\n"
            )
            if total + len(bloco) > max_chars:
                break
            linhas.append(bloco)
            total += len(bloco)

        if en_frags and total < max_chars:
            linhas.append("\n=== JURISPRUDÊNCIA TEDH / ECHR ===\n")
            for f in en_frags:
                bloco = (
                    f"[TEDH] {f.fonte} (rel={f.relevancia:.3f})\n"
                    f"{f.conteudo[:400]}\n{'─'*40}\n"
                )
                if total + len(bloco) > max_chars:
                    break
                linhas.append(bloco)
                total += len(bloco)

        linhas.append("=== FIM CONTEXTO ===\n")
        return "\n".join(linhas)

    # ── Utils ─────────────────────────────────────────────────────────
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
            "tedh": sum(1 for f in self._indice if f.lingua == "en"),
            "diplomas": sorted({f.diploma for f in self._indice if f.diploma}),
            "fontes": sorted({f.fonte for f in self._indice}),
            "modo": self.modo,
            "embeddings_computados": sum(1 for f in self._indice if f.embedding),
            "reranking": self.usar_reranking and self._rerank_model is not None,
            "modelo_embeddings": self.embedding_modelo_nome if self.modo == "hibrido" else "N/A",
            "modelo_reranker": self.reranker_modelo_nome if self.usar_reranking else "N/A",
        }

    def recarregar(self) -> int:
        self._indexado = False
        self._index_path.unlink(missing_ok=True)
        self._docfreq_path.unlink(missing_ok=True)
        return self.indexar(forcar=True)
