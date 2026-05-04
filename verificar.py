#!/usr/bin/env python3
"""
verificar.py V5 — Diagnóstico completo do ambiente.
Uso: python verificar.py [--api] [--rag] [--ollama] [--modelos]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
    def ok(m):  console.print(f"  [green]✅[/green]  {m}")
    def err(m): console.print(f"  [red]❌[/red]  {m}")
    def wrn(m): console.print(f"  [yellow]⚠️ [/yellow]  {m}")
    def inf(m): console.print(f"  [dim]ℹ️ [/dim]  {m}")
    def sec(t): console.print(f"\n[bold #1a3a5c]{'═'*55}[/bold #1a3a5c]\n  [bold]{t}[/bold]")
except ImportError:
    console = None
    def ok(m):  print(f"  ✅  {m}")
    def err(m): print(f"  ❌  {m}")
    def wrn(m): print(f"  ⚠️   {m}")
    def inf(m): print(f"  ℹ️   {m}")
    def sec(t): print(f"\n{'═'*55}\n  {t}")


def verificar_python():
    sec("PYTHON")
    v = sys.version_info
    if v >= (3, 10):
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        err(f"Python {v.major}.{v.minor} — recomendado 3.10+")


def verificar_deps():
    sec("DEPENDÊNCIAS")
    deps = {
        "httpx":             ("httpx", True),
        "dotenv":            ("python-dotenv", True),
        "streamlit":         ("streamlit", True),
        "pydantic":          ("pydantic>=2.7", True),
        "pydantic_settings": ("pydantic-settings", True),
        "tenacity":          ("tenacity", True),
        "typer":             ("typer", False),
        "rich":              ("rich", False),
        "structlog":         ("structlog", False),
        "fitz":              ("PyMuPDF (leitura PDF)", False),
        "reportlab":         ("reportlab (exportação PDF)", False),
        "sentence_transformers": ("sentence-transformers (RAG híbrido)", False),
        "numpy":             ("numpy (RAG híbrido)", False),
    }
    todos_criticos_ok = True
    for mod, (pkg, critico) in deps.items():
        try:
            __import__(mod)
            ok(f"{pkg}")
        except ImportError:
            if critico:
                err(f"{pkg}  [OBRIGATÓRIO]  →  pip install {pkg.split('>=')[0]}")
                todos_criticos_ok = False
            else:
                wrn(f"{pkg}  [opcional]  →  pip install {pkg.split(' ')[0]}")

    if not todos_criticos_ok:
        inf("Instala tudo:  pip install -r requirements.txt")


def verificar_env():
    sec("CONFIGURAÇÃO (.env)")
    from dotenv import load_dotenv
    import os
    load_dotenv()

    backend = os.getenv("BACKEND", "openrouter")
    inf(f"BACKEND = {backend}")

    if backend == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key or "cola" in api_key.lower() or api_key == "sem-chave":
            err("OPENROUTER_API_KEY não configurada")
            inf("Obtém chave gratuita em: https://openrouter.ai/keys")
        else:
            ok(f"OPENROUTER_API_KEY = {api_key[:8]}…{api_key[-4:]}")
        modelo = os.getenv("MODELO", "openrouter/auto")
        is_free = modelo.endswith(":free") or "free" in modelo.lower() or modelo == "openrouter/auto"
        ok(f"MODELO = {modelo}  {'[GRÁTIS]' if is_free else '[PAGO]'}")
    else:
        ok(f"BACKEND = ollama (sem chave OpenRouter necessária)")
        ok(f"OLLAMA_URL = {os.getenv('OLLAMA_URL','http://localhost:11434')}")
        ok(f"OLLAMA_MODELO = {os.getenv('OLLAMA_MODELO','llama3.3:70b')}")

    for var, default in [
        ("MAX_RETRIES", "5"), ("REQUEST_TIMEOUT", "180"),
        ("CACHE_ENABLED", "true"), ("PARALELISMO", "false"),
        ("MODO_ECONOMICO", "true"), ("HISTORICO_ENABLED", "true"),
        ("EXPORTAR_PDF", "true"), ("CONSISTENCIA_CHECK", "true"),
        ("RAG_MODO", "bm25"),
    ]:
        inf(f"{var} = {os.getenv(var, default)}")


def verificar_pastas():
    sec("PASTAS")
    pastas = [
        ("data/leis", True),
        ("data/jurisprudencia", False),
        ("data/precedentes", False),
        ("output_atas", False),
        ("logs", False),
        ("src/cache/data", False),
        ("src/historico/data", False),
    ]
    for pasta, obrigatoria in pastas:
        p = Path(pasta)
        if p.exists():
            n = len(list(p.glob("*")))
            ok(f"{pasta}/  ({n} ficheiro(s))")
        else:
            p.mkdir(parents=True, exist_ok=True)
            if obrigatoria:
                wrn(f"{pasta}/  — criada (adiciona ficheiros .txt de leis)")
            else:
                inf(f"{pasta}/  — criada automaticamente")


def verificar_rag(detalhe: bool = False):
    sec("RAG — BASE DE CONHECIMENTO")
    pasta = Path("data/leis")
    ficheiros = list(pasta.glob("*.txt")) if pasta.exists() else []
    if not ficheiros:
        wrn("Nenhum ficheiro .txt em data/leis/")
        inf("O RAG funciona sem ficheiros mas não adiciona contexto jurídico.")
        inf("Adiciona: Codigo_Penal.txt, Codigo_Civil.txt, Codigo_do_Trabalho.txt, etc.")
        return

    total_chars = 0
    for f in ficheiros:
        chars = len(f.read_text(encoding="utf-8", errors="replace"))
        total_chars += chars
        if detalhe:
            ok(f"{f.name}  ({chars:,} chars)")

    ok(f"{len(ficheiros)} ficheiro(s) em data/leis/ — {total_chars:,} chars total")

    try:
        from src.rag import MotorRAG
        rag = MotorRAG(Path("."))
        n = rag.indexar()
        stats = rag.estatisticas()
        ok(f"Indexado: {n} fragmentos  |  diplomas: {stats['diplomas']}")
        ok(f"Modo: {stats['modo']}  |  Embeddings: {stats['embeddings']}")

        if detalhe:
            frags = rag.pesquisar("furto arguido crime penal", n_resultados=3)
            if frags:
                ok(f"Pesquisa de teste: {len(frags)} resultado(s) — top score {frags[0].relevancia:.3f}")
            else:
                wrn("Pesquisa de teste sem resultados")
    except Exception as e:
        err(f"Erro ao indexar RAG: {e}")


def verificar_ollama():
    sec("OLLAMA (BACKEND LOCAL)")
    import os
    url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    try:
        import httpx
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{url}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                modelos = [m["name"] for m in data.get("models", [])]
                ok(f"Ollama disponível em {url}")
                if modelos:
                    ok(f"Modelos instalados: {', '.join(modelos[:5])}")
                else:
                    wrn("Nenhum modelo instalado. Executa: ollama pull llama3.3:70b")
            else:
                err(f"Ollama respondeu HTTP {resp.status_code}")
    except Exception as e:
        err(f"Ollama não disponível em {url}: {e}")
        inf("Para iniciar: ollama serve")
        inf("Para instalar modelo: ollama pull llama3.3:70b")


def verificar_api():
    sec("LIGAÇÃO À API OPENROUTER")
    import os
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    modelo = os.getenv("MODELO", "openrouter/auto")

    if not api_key or "cola" in api_key.lower():
        err("Sem chave API — salta teste")
        return

    inf(f"A testar: {modelo} (pode demorar até 60s em modelos gratuitos)...")
    try:
        import httpx, time
        start = time.time()
        with httpx.Client(timeout=90) as client:
            resp = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://tribunal-ia.gov.pt",
                    "X-Title": "Tribunal IA Portugal V5 — verificar",
                },
                json={
                    "model": modelo,
                    "messages": [{"role": "user", "content": "Responde apenas: OK"}],
                    "max_tokens": 5, "temperature": 0,
                },
            )
        elapsed = time.time() - start
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"].strip()
            ok(f"API OK em {elapsed:.1f}s — resposta: '{content}'")
        else:
            err(f"HTTP {resp.status_code}: {resp.text[:150]}")
    except Exception as e:
        err(f"Falha: {e}")


def listar_modelos():
    sec("MODELOS DISPONÍVEIS")
    from src.utils.config import FREE_MODELS, PAID_MODELS
    if console:
        t = Table(border_style="dim")
        t.add_column("Modelo", style="cyan")
        t.add_column("Tipo")
        t.add_column("Custo estimado/caso")
        for m in sorted(FREE_MODELS):
            t.add_row(m, "[green]GRÁTIS[/green]", "€0.00")
        for m, (ip, op) in list(PAID_MODELS.items()):
            est = round((ip * 3000 + op * 5000) / 1_000_000, 4)
            t.add_row(m, "[blue]PAGO[/blue]", f"~${est:.4f}")
        console.print(t)
    else:
        for m in FREE_MODELS:
            print(f"  🆓  {m}")
        for m in PAID_MODELS:
            print(f"  💳  {m}")


def main():
    parser = argparse.ArgumentParser(description="Diagnóstico V5")
    parser.add_argument("--api",     action="store_true", help="Testar API OpenRouter")
    parser.add_argument("--rag",     action="store_true", help="Detalhe do RAG")
    parser.add_argument("--ollama",  action="store_true", help="Verificar Ollama")
    parser.add_argument("--modelos", action="store_true", help="Listar modelos")
    args = parser.parse_args()

    if console:
        console.print(Panel(
            "[bold #1a3a5c]TRIBUNAL IA PORTUGAL V5[/bold #1a3a5c] — Diagnóstico",
            border_style="blue",
        ))
    else:
        print("\n🏛️  TRIBUNAL IA PORTUGAL V5 — Diagnóstico\n")

    if args.modelos:
        listar_modelos()
        return

    verificar_python()
    verificar_deps()
    verificar_env()
    verificar_pastas()
    verificar_rag(detalhe=args.rag)

    if args.ollama:
        verificar_ollama()
    if args.api:
        verificar_api()

    if not args.api and not args.ollama:
        inf("Usa --api para testar OpenRouter  |  --ollama para testar Ollama local")

    print()
    inf("Interface web:  streamlit run app.py")
    inf("CLI:            python main.py processar")
    inf("Modelos:        python verificar.py --modelos")
    print()


if __name__ == "__main__":
    main()
