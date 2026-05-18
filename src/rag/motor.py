"""Motor RAG V7 — PORTULAN + neuralmind + E5 com prefixos correctos + Cohere/local rerank + RRF"""
from __future__ import annotations
import math, pickle, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import numpy as np
    ST_OK = True
except ImportError:
    ST_OK = False

try:
    import cohere as _cohere
    COHERE_OK = True
except ImportError:
    COHERE_OK = False


@dataclass
class Fragmento:
    fonte: str; tipo: str; titulo: str; conteudo: str; relevancia: float
    artigo: Optional[str] = None; diploma: Optional[str] = None
    instancias: List[str] = field(default_factory=list)
    lingua: str = "pt"; embedding: Optional[List[float]] = None
    bm25_score: float = 0.0; sem_score: float = 0.0; rerank_score: float = 0.0


DIPLOMA_INSTANCIAS: Dict[str, List[str]] = {
    "CP":["TIC","TCCR","TCIC"],"CPP":["TIC","TCCR","TCIC"],
    "CC":["TC_CIVEL","TFM"],"CPC":["TC_CIVEL","TFM","TCOM"],
    "CT":["TRAB"],"CPT":["TRAB"],"CRP":["TC","TIC","TCCR","TAF"],
    "CPTA":["TAF"],"CPPT":["TAF"],"CIRE":["TCOM"],"CSC":["TCOM"],
    "RGPTC":["TFM"],"LPCJP":["TFM"],"ECHR":[],"TEDH":[],
}

DIPLOMA_KEYWORDS: Dict[str, List[str]] = {
    "CP":["código penal","crime","arguido","pena"],
    "CPP":["processo penal","inquérito","instrução criminal"],
    "CC":["código civil","contrato","obrigação"],
    "CPC":["processo civil","execução","penhora"],
    "CT":["código trabalho","trabalhador","despedimento"],
    "CRP":["constituição","direitos fundamentais"],
    "CPTA":["administrativo","contencioso"],"CIRE":["insolvência","recuperação"],
    "ECHR":["echr","european court","human rights"],"TEDH":["tedh","tribunal europeu"],
}

STOPWORDS_PT = {"a","o","as","os","um","uma","de","do","da","dos","das","em","no","na",
    "nos","nas","por","para","com","sem","sob","que","se","é","são","foi",
    "ser","ter","ao","à","aos","às","e","ou","mas","nem","não","sim","já",
    "n","nº","art","artigo","alínea","número","parágrafo"}
STOPWORDS_EN = {"the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","as","is","was","are","were","be","been","have","has","had",
    "this","that","it","its","they","them","their","we","you"}


class MotorRAG:
    def __init__(self, pasta_raiz: Path, modo: str="hibrido",
                 embedding_modelo: str="intfloat/multilingual-e5-large-instruct",
                 reranker_modelo: str="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 usar_reranking: bool=True, reranker_backend: str="local",
                 cohere_api_key: str="", top_k: int=15, top_n: int=6) -> None:
        self.pasta_raiz = pasta_raiz
        self.modo = modo if ST_OK else "bm25"
        self.embedding_modelo_nome = embedding_modelo
        self.reranker_modelo_nome = reranker_modelo
        self.usar_reranking = usar_reranking and ST_OK
        self.reranker_backend = reranker_backend
        self.cohere_api_key = cohere_api_key
        self.top_k = top_k; self.top_n = top_n
        self._indice: List[Fragmento] = []
        self._doc_freq: Dict[str, int] = {}
        self._indexado = False
        self._embed_model: Optional[SentenceTransformer] = None
        self._rerank_model: Optional[CrossEncoder] = None
        self._cohere_client = None
        # Prefixos correctos por arquitectura
        self._prefixo_query: str = ""
        self._prefixo_passage: str = ""
        self._definir_prefixos()
        cache_dir = pasta_raiz / "src" / "cache" / "data"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = cache_dir / "rag_index_v7.pkl"
        self._docfreq_path = cache_dir / "rag_docfreq_v7.pkl"
        if self.modo == "hibrido" and ST_OK:
            self._carregar_modelos()

    def _definir_prefixos(self) -> None:
        """Define prefixos correctos por arquitectura — E5 requer prefixos, PORTULAN/neuralmind não."""
        nome = self.embedding_modelo_nome.lower()
        if "e5" in nome:
            self._prefixo_query = "query: "
            self._prefixo_passage = "passage: "
        else:
            self._prefixo_query = ""
            self._prefixo_passage = ""

    def _carregar_modelos(self) -> None:
        if self._embed_model is None:
            try:
                self._embed_model = SentenceTransformer(self.embedding_modelo_nome, device="cpu")
            except Exception as e:
                print(f"[RAG V7] Embeddings falhou ({e}) — fallback BM25")
                self.modo = "bm25"; return
        if self.usar_reranking:
            if self.reranker_backend == "cohere" and self.cohere_api_key and COHERE_OK:
                try:
                    self._cohere_client = _cohere.Client(self.cohere_api_key)
                except Exception:
                    self._carregar_cross_encoder()
            else:
                self._carregar_cross_encoder()

    def _carregar_cross_encoder(self) -> None:
        if self._rerank_model is None:
            try:
                self._rerank_model = CrossEncoder(self.reranker_modelo_nome, device="cpu", max_length=512)
            except Exception:
                self.usar_reranking = False

    def indexar(self, forcar: bool=False) -> int:
        if not forcar and self._index_path.exists():
            try:
                with open(self._index_path,"rb") as f: self._indice = pickle.load(f)
                with open(self._docfreq_path,"rb") as f: self._doc_freq = pickle.load(f)
                self._indexado = True
                if self.modo=="hibrido" and ST_OK and not any(fr.embedding for fr in self._indice):
                    if self._embed_model: self._computar_embeddings(); self._persistir()
                return len(self._indice)
            except Exception: pass
        self._indice = []; self._doc_freq = {}
        for pasta, tipo, lingua in [
            (self.pasta_raiz/"data"/"leis","lei","pt"),
            (self.pasta_raiz/"data"/"jurisprudencia","jurisprudencia","pt"),
            (self.pasta_raiz/"data"/"precedentes","precedente","pt"),
            (self.pasta_raiz/"data"/"tedh","tedh","en"),
        ]:
            if pasta.exists():
                for f in sorted(pasta.glob("*.txt")):
                    self._indice.extend(self._processar_ficheiro(f, tipo, lingua))
        for frag in self._indice:
            for t in set(self._tokenizar(frag.conteudo+" "+frag.titulo, frag.lingua)):
                self._doc_freq[t] = self._doc_freq.get(t, 0) + 1
        if self.modo=="hibrido" and ST_OK and self._embed_model:
            self._computar_embeddings()
        self._indexado = True; self._persistir()
        return len(self._indice)

    def _computar_embeddings(self) -> None:
        if not self._embed_model: return
        textos = [self._prefixo_passage + f.conteudo[:512] for f in self._indice]
        try:
            embs = self._embed_model.encode(textos, batch_size=32,
                show_progress_bar=False, normalize_embeddings=True)
            for i, frag in enumerate(self._indice):
                frag.embedding = embs[i].tolist()
        except Exception as e:
            print(f"[RAG V7] Embeddings error: {e}")

    def _persistir(self) -> None:
        try:
            with open(self._index_path,"wb") as f: pickle.dump(self._indice, f)
            with open(self._docfreq_path,"wb") as f: pickle.dump(self._doc_freq, f)
        except Exception: pass

    def _detectar_diploma(self, nome: str, texto: str) -> Optional[str]:
        nu = nome.upper().replace(" ","").replace("_","")
        for diploma, kws in DIPLOMA_KEYWORDS.items():
            if diploma in nu: return diploma
            if any(kw in texto[:500].lower() for kw in kws): return diploma
        return None

    def _processar_ficheiro(self, path: Path, tipo: str, lingua: str) -> List[Fragmento]:
        try: texto = path.read_text(encoding="utf-8", errors="replace")
        except Exception: return []
        nome = path.stem.replace("_"," ").replace("-"," ")
        diploma = self._detectar_diploma(nome, texto)
        instancias = DIPLOMA_INSTANCIAS.get(diploma or "", [])
        partes = re.split(r"\n(?=(?:Artigo\s+\d+|Art\.?\s+\d+|Article\s+\d+))",
                         texto, flags=re.IGNORECASE)
        frags: List[Fragmento] = []
        if len(partes) > 1:
            for parte in partes:
                parte = parte.strip()
                if len(parte) < 40: continue
                m = re.match(r"(Art(?:igo|icle)?\.?\s+\d+\.?[º°]?[A-Za-z]?)", parte, re.IGNORECASE)
                frags.append(Fragmento(fonte=nome, tipo=tipo,
                    titulo=parte[:100].split("\n")[0].strip(),
                    conteudo=parte[:2000], relevancia=0.0,
                    artigo=m.group(1) if m else None,
                    diploma=diploma, instancias=instancias, lingua=lingua))
        else:
            for i, bloco in enumerate([b.strip() for b in texto.split("\n\n") if len(b.strip())>80][:100]):
                frags.append(Fragmento(fonte=nome, tipo=tipo,
                    titulo=f"{nome} — bloco {i+1}", conteudo=bloco[:2000],
                    relevancia=0.0, diploma=diploma, instancias=instancias, lingua=lingua))
        return frags

    def _tokenizar(self, texto: str, lingua: str="pt") -> List[str]:
        palavras = re.findall(r"\b[a-záàâãéêíóôõúçA-ZÁÀÂÃÉÊÍÓÔÕÚÇ\w]{3,}\b", texto.lower())
        sw = STOPWORDS_EN if lingua=="en" else STOPWORDS_PT
        return [p for p in palavras if p not in sw]

    def _score_bm25(self, query_tokens: List[str], frag: Fragmento) -> float:
        doc_tokens = self._tokenizar(frag.conteudo+" "+frag.titulo, frag.lingua)
        if not doc_tokens or not query_tokens: return 0.0
        k1, b, avg_len = 1.5, 0.75, 250
        freq: Dict[str, int] = {}
        for t in doc_tokens: freq[t] = freq.get(t,0)+1
        N = max(len(self._indice), 1); score = 0.0
        for token in set(query_tokens):
            if token in freq:
                tf = freq[token]; df = self._doc_freq.get(token,1)
                idf = math.log((N-df+0.5)/(df+0.5)+1)
                score += idf*(tf*(k1+1))/(tf+k1*(1-b+b*len(doc_tokens)/avg_len))
        return score

    def _cosine(self, a: List[float], b: List[float]) -> float:
        if not ST_OK: return 0.0
        va = np.array(a, dtype=np.float32); vb = np.array(b, dtype=np.float32)
        d = float(np.linalg.norm(va)*np.linalg.norm(vb))
        return float(np.dot(va,vb)/d) if d>0 else 0.0

    @staticmethod
    def _rrf(r1: int, r2: int, k: int=60) -> float:
        return 1.0/(k+r1)+1.0/(k+r2)

    def _rerank_cohere(self, consulta: str, cands: List[Tuple[float,"Fragmento"]]) -> List["Fragmento"]:
        if not self._cohere_client or not COHERE_OK:
            return [f for _,f in cands[:self.top_n]]
        try:
            docs = [f.conteudo[:500] for _,f in cands]
            resp = self._cohere_client.rerank(query=consulta, documents=docs,
                model="rerank-multilingual-v3.0", top_n=self.top_n)
            return [Fragmento(fonte=cands[r.index][1].fonte, tipo=cands[r.index][1].tipo,
                titulo=cands[r.index][1].titulo, conteudo=cands[r.index][1].conteudo,
                artigo=cands[r.index][1].artigo, diploma=cands[r.index][1].diploma,
                instancias=cands[r.index][1].instancias, lingua=cands[r.index][1].lingua,
                relevancia=round(float(r.relevance_score),4),
                rerank_score=round(float(r.relevance_score),4))
                for r in resp.results]
        except Exception:
            return self._rerank_local(consulta, cands)

    def _rerank_local(self, consulta: str, cands: List[Tuple[float,"Fragmento"]]) -> List["Fragmento"]:
        if not self._rerank_model: return [f for _,f in cands[:self.top_n]]
        try:
            scores = self._rerank_model.predict([(consulta[:256], f.conteudo[:400]) for _,f in cands])
            reranked = sorted(zip(scores,[f for _,f in cands]), key=lambda x:x[0], reverse=True)
            return [Fragmento(fonte=f.fonte,tipo=f.tipo,titulo=f.titulo,conteudo=f.conteudo,
                artigo=f.artigo,diploma=f.diploma,instancias=f.instancias,lingua=f.lingua,
                relevancia=round(float(s),4),rerank_score=round(float(s),4))
                for s,f in reranked[:self.top_n]]
        except Exception:
            return [f for _,f in cands[:self.top_n]]

    def pesquisar(self, consulta: str, n_resultados: Optional[int]=None,
                  instancia: Optional[str]=None, tipo_filtro: Optional[str]=None,
                  diploma_filtro: Optional[str]=None, lingua_filtro: Optional[str]=None) -> List[Fragmento]:
        if not self._indexado: self.indexar()
        if not self._indice: return []
        top_n = n_resultados or self.top_n
        cands = [f for f in self._indice
                 if (not tipo_filtro or f.tipo==tipo_filtro)
                 and (not diploma_filtro or f.diploma==diploma_filtro)
                 and (not instancia or not f.instancias or instancia in f.instancias)
                 and (not lingua_filtro or f.lingua==lingua_filtro)]
        if not cands: return []
        qt_pt = self._tokenizar(consulta,"pt"); qt_en = self._tokenizar(consulta,"en")
        bm25: List[Tuple[float,Fragmento]] = []
        for frag in cands:
            s = self._score_bm25(qt_en if frag.lingua=="en" else qt_pt, frag)
            if s>0: bm25.append((s,frag))
        bm25.sort(key=lambda x:x[0],reverse=True)
        top = bm25[:self.top_k*2]
        if not top: return []
        if self.modo=="hibrido" and self._embed_model and ST_OK:
            try:
                q_emb = self._embed_model.encode([self._prefixo_query+consulta[:512]],
                    normalize_embeddings=True,show_progress_bar=False)[0].tolist()
                sem = [(self._cosine(q_emb,f.embedding) if f.embedding else 0.0, f) for _,f in top]
                sem.sort(key=lambda x:x[0],reverse=True)
                br = {id(f):i+1 for i,(_,f) in enumerate(top)}
                sr = {id(f):i+1 for i,(_,f) in enumerate(sem)}
                seen: set = set(); rrf: List[Tuple[float,Fragmento]] = []
                for _,frag in top:
                    fid=id(frag)
                    if fid not in seen:
                        seen.add(fid)
                        rrf.append((self._rrf(br.get(fid,self.top_k*2),sr.get(fid,self.top_k*2)),frag))
                rrf.sort(key=lambda x:x[0],reverse=True)
                top = rrf[:self.top_k]
            except Exception:
                top = top[:self.top_k]
        else:
            top = top[:self.top_k]
        if self.usar_reranking and len(top)>1:
            if self.reranker_backend=="cohere" and self._cohere_client:
                return self._rerank_cohere(consulta,top)[:top_n]
            elif self._rerank_model:
                return self._rerank_local(consulta,top)[:top_n]
        return [Fragmento(fonte=f.fonte,tipo=f.tipo,titulo=f.titulo,conteudo=f.conteudo,
            artigo=f.artigo,diploma=f.diploma,instancias=f.instancias,lingua=f.lingua,
            relevancia=round(s,4)) for s,f in top[:top_n]]

    def formatar_contexto(self, fragmentos: List[Fragmento], max_chars: int=3500,
                          incluir_tedh: bool=True) -> str:
        if not fragmentos: return ""
        sep = "-"*40
        pt_frags = [f for f in fragmentos if f.lingua=="pt"]
        en_frags = [f for f in fragmentos if f.lingua=="en" and incluir_tedh]
        linhas = ["=== CONTEXTO JURIDICO (RAG V7) ===\n"]
        total = 0
        for f in pt_frags:
            diploma_tag = f" [{f.diploma}]" if f.diploma else ""
            artigo_tag = f" - {f.artigo}" if f.artigo else ""
            bloco = (f"[{f.tipo.upper()}{diploma_tag}] {f.fonte}{artigo_tag} "
                     f"(rel={f.relevancia:.3f})\n{f.conteudo[:600]}\n{sep}\n")
            if total+len(bloco) > max_chars: break
            linhas.append(bloco); total += len(bloco)
        if en_frags and total < max_chars:
            linhas.append("\n=== TEDH / ECHR ===\n")
            for f in en_frags:
                bloco = f"[TEDH] {f.fonte} (rel={f.relevancia:.3f})\n{f.conteudo[:400]}\n{sep}\n"
                if total+len(bloco) > max_chars: break
                linhas.append(bloco); total += len(bloco)
        linhas.append("=== FIM ===\n")
        return "\n".join(linhas)

    def tem_dados(self) -> bool:
        if not self._indexado: self.indexar()
        return bool(self._indice)

    def estatisticas(self) -> Dict[str, object]:
        if not self._indexado: self.indexar()
        return {
            "total": len(self._indice),
            "leis": sum(1 for f in self._indice if f.tipo=="lei"),
            "jurisprudencia": sum(1 for f in self._indice if f.tipo=="jurisprudencia"),
            "precedentes": sum(1 for f in self._indice if f.tipo=="precedente"),
            "tedh": sum(1 for f in self._indice if f.lingua=="en"),
            "diplomas": sorted({f.diploma for f in self._indice if f.diploma}),
            "fontes": sorted({f.fonte for f in self._indice}),
            "modo": self.modo,
            "embeddings_computados": sum(1 for f in self._indice if f.embedding),
            "reranking": self.usar_reranking,
            "reranker_backend": self.reranker_backend,
            "modelo_embeddings": self.embedding_modelo_nome,
            "modelo_reranker": self.reranker_modelo_nome,
            "prefixo_query": self._prefixo_query,
        }

    def recarregar(self) -> int:
        self._indexado = False
        self._index_path.unlink(missing_ok=True)
        self._docfreq_path.unlink(missing_ok=True)
        return self.indexar(forcar=True)
