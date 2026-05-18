"""API REST V7 — JWT + Rate limiting + OpenAPI + Observability"""
from __future__ import annotations
import threading, time as _t
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Depends, status, Request, Response, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import PlainTextResponse
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, Field
    FASTAPI_OK = True
except ImportError:
    FASTAPI_OK = False

try:
    from jose import JWTError, jwt as _jwt
    JOSE_OK = True
except ImportError:
    JOSE_OK = False

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    SLOWAPI_OK = True
except ImportError:
    SLOWAPI_OK = False

try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    PROM_OK = True
except ImportError:
    PROM_OK = False

if FASTAPI_OK:
    class TokenResponse(BaseModel):
        access_token: str; token_type: str = "bearer"; expires_in: int

    class LoginRequest(BaseModel):
        username: str = Field(..., examples=["tribunal_user"])
        password: str = Field(..., examples=["senha_aqui"])

    class PedidoCaso(BaseModel):
        descricao: str = Field(..., min_length=20,
            examples=["Fui despedido sem justa causa após 8 anos. A empresa contratou outra pessoa 2 semanas depois."])
        instancia: Optional[str] = Field(None, examples=["TRAB"])
        modelo: Optional[str] = Field(None, examples=["openrouter/free"])
        dados_instrucao: Optional[Dict[str, Any]] = None
        intervencao_utilizador: Optional[str] = None
        gerar_pdf: bool = True

    class PedidoInstrucao(BaseModel):
        descricao: str = Field(..., min_length=20)
        instancia: Optional[str] = None

    class PedidoContraditorio(BaseModel):
        case_id: str; argumento: str = Field(..., min_length=10); avaliar: bool = True

    class RespostaSaude(BaseModel):
        status: str; versao: str; modelo: str; backend: str
        rag_modo: str; reranking: bool; orquestracao: str
        observability: bool; timestamp: str

    security = HTTPBearer(auto_error=False)
    if SLOWAPI_OK:
        limiter = Limiter(key_func=get_remote_address)
    else:
        limiter = None

    _proc_lock = threading.Lock()
    _proc_inst: Any = None

    def _get_proc() -> Any:
        global _proc_inst
        with _proc_lock:
            if _proc_inst is None:
                from ..pipeline import CaseProcessor
                _proc_inst = CaseProcessor()
        return _proc_inst

    def _criar_token(user: str, secret: str, exp_min: int) -> str:
        if not JOSE_OK:
            return f"demo_{user}"
        exp = datetime.now(timezone.utc) + timedelta(minutes=exp_min)
        return _jwt.encode({"sub": user, "exp": exp}, secret, algorithm="HS256")

    def _verificar(token: str, secret: str) -> Optional[str]:
        if not JOSE_OK:
            return token.replace("demo_","") if token.startswith("demo_") else None
        try:
            p = _jwt.decode(token, secret, algorithms=["HS256"])
            return p.get("sub")
        except JWTError:
            return None

    def criar_app() -> FastAPI:
        from ..utils.config import get_config
        cfg = get_config()

        app = FastAPI(
            title="Tribunal IA Portugal V7",
            description=(
                "API REST para simulação judicial.\n\n"
                "## Auth\nUsa `POST /auth/token` para JWT Bearer.\n\n"
                f"## Rate Limit\n{cfg.api_rate_limit} req/min por IP.\n\n"
                "## ⚠️ Fins exclusivamente educativos."
            ),
            version="7.0.0",
            docs_url="/docs", redoc_url="/redoc",
        )
        app.add_middleware(CORSMiddleware, allow_origins=["*"],
                          allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
        if SLOWAPI_OK and limiter:
            app.state.limiter = limiter
            app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

        from ..observability import metrics, setup_tracing
        setup_tracing(cfg.otel_service_name, cfg.otel_exporter_otlp_endpoint)
        metrics.registar_info("7.0.0", cfg.modelo_activo, cfg.rag_modo)

        def _auth(creds: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)]) -> str:
            if not creds:
                raise HTTPException(401, "Token necessário", headers={"WWW-Authenticate":"Bearer"})
            u = _verificar(creds.credentials, cfg.api_secret_key)
            if not u:
                raise HTTPException(401, "Token inválido", headers={"WWW-Authenticate":"Bearer"})
            return u

        @app.get("/", include_in_schema=False)
        async def raiz() -> Dict[str, str]:
            return {"versao":"7.0.0","docs":"/docs","metricas":"/metrics",
                    "aviso":"Fins exclusivamente educativos"}

        @app.get("/saude", response_model=RespostaSaude, tags=["Sistema"])
        async def saude() -> RespostaSaude:
            return RespostaSaude(status="ok", versao="7.0.0",
                modelo=cfg.modelo_activo, backend=cfg.backend,
                rag_modo=cfg.rag_modo, reranking=cfg.rag_reranking,
                orquestracao=cfg.orquestracao, observability=cfg.observability_enabled,
                timestamp=datetime.now(timezone.utc).isoformat())

        @app.get("/metrics", tags=["Observability"], summary="Prometheus metrics")
        async def prometheus_metrics() -> Response:
            if not PROM_OK:
                return PlainTextResponse("# prometheus_client nao instalado\n")
            return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

        @app.get("/metricas/resumo", tags=["Observability"])
        async def metricas_resumo(_: Annotated[str, Depends(_auth)]) -> Dict[str, Any]:
            from ..observability import metrics as m
            return m.resumo()

        @app.post("/auth/token", response_model=TokenResponse, tags=["Autenticação"],
                  summary="Obter JWT")
        async def login(p: LoginRequest) -> TokenResponse:
            token = _criar_token(p.username, cfg.api_secret_key, cfg.api_access_token_expire_minutes)
            return TokenResponse(access_token=token, expires_in=cfg.api_access_token_expire_minutes*60)

        @app.post("/instrucao", tags=["Processo"])
        async def instrucao(p: PedidoInstrucao, _: Annotated[str,Depends(_auth)]) -> Dict[str,Any]:
            try:
                return {"sucesso":True, "instancia": p.instancia or "TIC",
                        "perguntas": _get_proc().gerar_perguntas_instrucao(p.descricao, p.instancia or "TIC")}
            except Exception as e:
                raise HTTPException(500, str(e))

        @app.post("/processar", tags=["Processo"],
                  summary="Processa caso completo (7 agentes + consistência + TEDH)")
        async def processar(p: PedidoCaso, _user: Annotated[str,Depends(_auth)],
                            bg: BackgroundTasks) -> Dict[str, Any]:
            import os
            if p.modelo:
                os.environ["MODELO"] = p.modelo
                from ..utils.config import reset_config
                from ..utils.brain import reset_brain
                reset_config(); reset_brain()
            from ..observability import metrics as m, MetricasCaso
            start = _t.time()
            try:
                result = _get_proc().process(
                    case_description=p.descricao, instancia_codigo=p.instancia,
                    dados_instrucao=p.dados_instrucao, gerar_pdf=p.gerar_pdf,
                    intervencao_utilizador=p.intervencao_utilizador,
                )
                elapsed = _t.time() - start
                bg.add_task(m.registar_caso, MetricasCaso(
                    case_id=result.case_id, instancia=result.instancia_codigo,
                    modelo=result.modelo_usado, backend=result.backend_usado,
                    rag_modo=cfg.rag_modo, orquestracao=cfg.orquestracao,
                    duracao_total_s=elapsed, custo_usd=result.custo_total_usd,
                    n_entidades_anonimizadas=len(result.entities_found),
                    grau_incerteza=result.grau_incerteza,
                    tem_tedh=bool(result.analise_tedh),
                    tem_contraditorio=bool(p.intervencao_utilizador),
                ))
                return {
                    "sucesso": True, "case_id": result.case_id,
                    "trace_id": result.trace_id, "tribunal": result.instancia_nome,
                    "modelo": result.modelo_usado, "grau_incerteza": result.grau_incerteza,
                    "custo_usd": result.custo_total_usd, "doc_hash": result.doc_hash,
                    "entidades_anonimizadas": len(result.entities_found),
                    "duracao_s": round(elapsed, 2), "orquestracao": cfg.orquestracao,
                    "pecas": {"detetive": result.detetive_report, "acusacao": result.acusacao,
                              "defesa": result.defesa, "sentenca_rigorosa": result.sentenca_rigorosa,
                              "sentenca_garantista": result.sentenca_garantista,
                              "sentenca_equilibrada": result.sentenca_equilibrada},
                    "relatorio_consistencia": result.relatorio_consistencia,
                    "analise_tedh": result.analise_tedh,
                    "ata_path": str(result.ata_path) if result.ata_path else None,
                }
            except Exception as e:
                m.registar_erro("api","processar",str(e))
                raise HTTPException(500, str(e))

        @app.get("/ata/{case_id}/pdf", tags=["Documentos"])
        async def download_pdf(case_id: str, _: Annotated[str,Depends(_auth)]) -> Response:
            p2 = cfg.pasta_atas / f"{case_id}.pdf"
            if not p2.exists(): raise HTTPException(404, "PDF não encontrado")
            return Response(content=p2.read_bytes(), media_type="application/pdf",
                           headers={"Content-Disposition":f"attachment; filename={case_id}.pdf"})

        @app.get("/ata/{case_id}/txt", tags=["Documentos"])
        async def download_txt(case_id: str, _: Annotated[str,Depends(_auth)]) -> Response:
            p2 = cfg.pasta_atas / f"{case_id}.txt"
            if not p2.exists(): raise HTTPException(404, "Ata não encontrada")
            return Response(content=p2.read_text(encoding="utf-8"),
                           media_type="text/plain; charset=utf-8")

        @app.post("/contraditorio", tags=["Contraditório"])
        async def contraditorio(p: PedidoContraditorio, _: Annotated[str,Depends(_auth)]) -> Dict[str,Any]:
            try:
                from ..contraditorio import get_gestor_contraditorio
                iv = get_gestor_contraditorio().submeter_argumento(p.case_id, p.argumento, p.avaliar)
                return {"sucesso":True,"numero":iv.numero,"argumento":iv.argumento,"feedback":iv.feedback_juridico}
            except ValueError as e: raise HTTPException(404, str(e))
            except Exception as e: raise HTTPException(500, str(e))

        @app.get("/historico", tags=["Histórico"])
        async def historico(_: Annotated[str,Depends(_auth)], query:str="",
                            instancia:Optional[str]=None, limite:int=20) -> Dict[str,Any]:
            from ..historico import get_historico
            h = get_historico()
            regs = h.pesquisar(query=query, instancia=instancia, limite=limite)
            s = h.estatisticas()
            return {"total":s["total"], "registos":[
                {"id":r.id,"timestamp":r.timestamp,"tribunal":r.instancia_codigo,
                 "grau_incerteza":r.grau_incerteza,"custo_usd":r.custo_usd} for r in regs]}

        @app.get("/rag/stats", tags=["RAG"])
        async def rag_stats(_: Annotated[str,Depends(_auth)]) -> Dict[str,Any]:
            return dict(_get_proc().rag.estatisticas())

        @app.get("/rag/pesquisar", tags=["RAG"])
        async def rag_pesquisar(q:str, _: Annotated[str,Depends(_auth)],
                                instancia:Optional[str]=None, n:int=5) -> Dict[str,Any]:
            frags = _get_proc().rag.pesquisar(q, n_resultados=n, instancia=instancia)
            return {"query":q,"n":len(frags),"resultados":[
                {"fonte":f.fonte,"tipo":f.tipo,"diploma":f.diploma,"lingua":f.lingua,
                 "relevancia":f.relevancia,"excerto":f.conteudo[:300]} for f in frags]}

        @app.get("/instancias", tags=["Sistema"])
        async def instancias() -> Dict[str,Any]:
            from ..pipeline.instancias import INSTANCIAS
            return {c:{"nome":i.nome,"materia":i.materia,"diploma":i.diploma_principal}
                    for c,i in INSTANCIAS.items()}

        return app

    app = criar_app()
else:
    app = None
