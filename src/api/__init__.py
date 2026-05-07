"""
API REST V6 — FastAPI
══════════════════════
Endpoints:
  POST /processar          → processa um caso completo
  POST /instrucao          → gera perguntas de instrução
  POST /contraditorio      → submete argumento de defesa
  GET  /historico          → lista histórico de casos
  GET  /rag/stats          → estatísticas do RAG
  GET  /saude              → health check
  GET  /modelo             → info do modelo activo
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, Response
    from pydantic import BaseModel, Field
    FASTAPI_OK = True
except ImportError:
    FASTAPI_OK = False

if FASTAPI_OK:

    # ── Modelos Pydantic ──────────────────────────────────────────────
    class PedidoCaso(BaseModel):
        descricao: str = Field(..., min_length=20, description="Descrição do caso")
        instancia: Optional[str] = Field(None, description="Código do tribunal")
        modelo: Optional[str] = Field(None, description="Override do modelo")
        dados_instrucao: Optional[Dict] = None
        intervencao_utilizador: Optional[str] = None
        gerar_pdf: bool = True

    class PedidoInstrucao(BaseModel):
        descricao: str = Field(..., min_length=20)
        instancia: Optional[str] = None

    class PedidoContraditorio(BaseModel):
        case_id: str
        argumento: str = Field(..., min_length=10)
        avaliar: bool = True

    class RespostaSaude(BaseModel):
        status: str
        versao: str
        modelo: str
        backend: str
        rag_modo: str
        orquestracao: str
        timestamp: str

    # ── App ───────────────────────────────────────────────────────────
    def criar_app() -> FastAPI:
        app = FastAPI(
            title="Tribunal IA Portugal V6 — API",
            description=(
                "API REST para simulação judicial com IA.\n\n"
                "⚠️ **Aviso Legal:** Fins exclusivamente educativos. "
                "Não constitui parecer jurídico."
            ),
            version="6.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Cache do processor (thread-safe)
        _processor_lock = threading.Lock()
        _processor_instance = None

        def get_processor():
            nonlocal _processor_instance
            with _processor_lock:
                if _processor_instance is None:
                    from ..pipeline import CaseProcessor
                    _processor_instance = CaseProcessor()
            return _processor_instance

        # ── Health check ──────────────────────────────────────────────
        @app.get("/saude", response_model=RespostaSaude, tags=["Sistema"])
        async def saude():
            from ..utils.config import get_config
            cfg = get_config()
            return RespostaSaude(
                status="ok",
                versao="6.0.0",
                modelo=cfg.modelo_activo,
                backend=cfg.backend,
                rag_modo=cfg.rag_modo,
                orquestracao=cfg.orquestracao,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        @app.get("/", tags=["Sistema"])
        async def raiz():
            return {
                "nome": "Tribunal IA Portugal V6",
                "versao": "6.0.0",
                "docs": "/docs",
                "aviso": "Fins exclusivamente educativos — não constitui parecer jurídico",
            }

        # ── Instrução ─────────────────────────────────────────────────
        @app.post("/instrucao", tags=["Processo"])
        async def instrucao(pedido: PedidoInstrucao):
            try:
                proc = get_processor()
                perguntas = proc.gerar_perguntas_instrucao(
                    pedido.descricao,
                    pedido.instancia or "TIC",
                )
                return {
                    "sucesso": True,
                    "instancia": pedido.instancia or "TIC",
                    "perguntas": perguntas,
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # ── Processar caso ────────────────────────────────────────────
        @app.post("/processar", tags=["Processo"])
        async def processar(pedido: PedidoCaso):
            if pedido.modelo:
                os.environ["MODELO"] = pedido.modelo
                from ..utils.config import reset_config
                from ..utils.brain import reset_brain
                reset_config()
                reset_brain()

            try:
                proc = get_processor()
                result = proc.process(
                    case_description=pedido.descricao,
                    instancia_codigo=pedido.instancia,
                    dados_instrucao=pedido.dados_instrucao,
                    gerar_pdf=pedido.gerar_pdf,
                    intervencao_utilizador=pedido.intervencao_utilizador,
                )
                resposta = {
                    "sucesso": True,
                    "case_id": result.case_id,
                    "trace_id": result.trace_id,
                    "tribunal": result.instancia_nome,
                    "modelo": result.modelo_usado,
                    "grau_incerteza": result.grau_incerteza,
                    "custo_usd": result.custo_total_usd,
                    "doc_hash": result.doc_hash,
                    "entidades_anonimizadas": len(result.entities_found),
                    "pecas": {
                        "detetive": result.detetive_report,
                        "acusacao": result.acusacao,
                        "defesa": result.defesa,
                        "sentenca_rigorosa": result.sentenca_rigorosa,
                        "sentenca_garantista": result.sentenca_garantista,
                        "sentenca_equilibrada": result.sentenca_equilibrada,
                    },
                    "relatorio_consistencia": result.relatorio_consistencia,
                    "analise_tedh": result.analise_tedh,
                    "ata_path": str(result.ata_path) if result.ata_path else None,
                }
                return resposta
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # ── Download PDF ──────────────────────────────────────────────
        @app.get("/ata/{case_id}/pdf", tags=["Documentos"])
        async def download_pdf(case_id: str):
            from ..utils.config import get_config
            cfg = get_config()
            pdf_path = cfg.pasta_atas / f"{case_id}.pdf"
            if not pdf_path.exists():
                raise HTTPException(status_code=404, detail="PDF não encontrado")
            return Response(
                content=pdf_path.read_bytes(),
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename={case_id}.pdf"},
            )

        @app.get("/ata/{case_id}/txt", tags=["Documentos"])
        async def download_txt(case_id: str):
            from ..utils.config import get_config
            cfg = get_config()
            txt_path = cfg.pasta_atas / f"{case_id}.txt"
            if not txt_path.exists():
                raise HTTPException(status_code=404, detail="Ata não encontrada")
            return Response(
                content=txt_path.read_text(encoding="utf-8"),
                media_type="text/plain; charset=utf-8",
                headers={"Content-Disposition": f"attachment; filename={case_id}.txt"},
            )

        # ── Modo Contraditório ────────────────────────────────────────
        @app.post("/contraditorio", tags=["Contraditório"])
        async def contraditorio(pedido: PedidoContraditorio):
            try:
                from ..contraditorio import get_gestor_contraditorio
                gestor = get_gestor_contraditorio()
                iv = gestor.submeter_argumento(
                    pedido.case_id, pedido.argumento, pedido.avaliar
                )
                return {
                    "sucesso": True,
                    "numero_intervencao": iv.numero,
                    "argumento": iv.argumento,
                    "feedback_juridico": iv.feedback_juridico,
                }
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # ── Histórico ─────────────────────────────────────────────────
        @app.get("/historico", tags=["Histórico"])
        async def historico(
            query: str = "",
            instancia: Optional[str] = None,
            limite: int = 20,
        ):
            from ..historico import get_historico
            hist = get_historico()
            registos = hist.pesquisar(query=query, instancia=instancia, limite=limite)
            stats = hist.estatisticas()
            return {
                "total": stats["total"],
                "registos": [
                    {
                        "id": r.id,
                        "timestamp": r.timestamp,
                        "tribunal": r.instancia_codigo,
                        "tribunal_nome": r.instancia_nome,
                        "resumo": r.resumo,
                        "grau_incerteza": r.grau_incerteza,
                        "custo_usd": r.custo_usd,
                        "modelo": r.modelo,
                    }
                    for r in registos
                ],
            }

        # ── RAG Stats ─────────────────────────────────────────────────
        @app.get("/rag/stats", tags=["RAG"])
        async def rag_stats():
            proc = get_processor()
            return proc.rag.estatisticas()

        # ── Instâncias ────────────────────────────────────────────────
        @app.get("/instancias", tags=["Sistema"])
        async def instancias():
            from ..pipeline.instancias import INSTANCIAS
            return {
                cod: {
                    "nome": inst.nome,
                    "materia": inst.materia,
                    "diploma": inst.diploma_principal,
                }
                for cod, inst in INSTANCIAS.items()
            }

        return app

    app = criar_app()

else:
    app = None  # type: ignore
