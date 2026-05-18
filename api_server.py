#!/usr/bin/env python3
"""
Tribunal IA Portugal V6 — Servidor API REST (FastAPI)
Uso: python api_server.py [--host 0.0.0.0] [--port 8000]

Documentação interactiva: http://localhost:8000/docs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tribunal IA Portugal V6 — API REST")
    parser.add_argument("--host", default=None, help="Host (default: .env API_HOST)")
    parser.add_argument("--port", type=int, default=None, help="Porto (default: .env API_PORT)")
    parser.add_argument("--reload", action="store_true", help="Hot reload (dev)")
    parser.add_argument("--workers", type=int, default=1, help="Workers uvicorn")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("❌ uvicorn não instalado. Instala com: pip install uvicorn[standard]")
        sys.exit(1)

    try:
        from src.utils.config import get_config
        cfg = get_config()
    except Exception as e:
        print(f"❌ Configuração inválida: {e}")
        sys.exit(1)

    host = args.host or cfg.api_host
    port = args.port or cfg.api_port

    try:
        from src.api import app, FASTAPI_OK
        if not FASTAPI_OK or app is None:
            print("❌ FastAPI não instalado. Instala com: pip install fastapi")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Erro ao carregar API: {e}")
        sys.exit(1)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  🏛️  TRIBUNAL IA PORTUGAL V7 — API REST                      ║
╚══════════════════════════════════════════════════════════════╝

  Modelo    : {cfg.modelo_activo} [{cfg.backend}]
  RAG       : {cfg.rag_modo} | Reranking: {cfg.rag_reranking}
  Orquestração: {'LangGraph' if cfg.usar_langgraph else 'Imperativo'}

  URL       : http://{host}:{port}
  Docs      : http://{host}:{port}/docs
  ReDoc     : http://{host}:{port}/redoc
  Saúde     : http://{host}:{port}/saude

  ⚠️  Fins exclusivamente educativos e de simulação
""")

    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
