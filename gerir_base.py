#!/usr/bin/env python3
"""
gerir_base.py V5 — Gerir a base de conhecimento jurídica (RAG).
Uso:
  python gerir_base.py --stats
  python gerir_base.py --reindexar
  python gerir_base.py --pesquisar "texto" [--instancia TIC]
  python gerir_base.py --limpar-cache
  python gerir_base.py --historico [--query "texto"]
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
    RICH = True
except ImportError:
    console = None
    RICH = False


def main():
    parser = argparse.ArgumentParser(description="Gerir base de conhecimento V5")
    parser.add_argument("--reindexar",   action="store_true", help="Força reindexação")
    parser.add_argument("--stats",       action="store_true", help="Estatísticas")
    parser.add_argument("--pesquisar",   "-p", default=None,  help="Pesquisa de teste")
    parser.add_argument("--instancia",   "-i", default=None,  help="Filtrar por instância")
    parser.add_argument("--diploma",     "-d", default=None,  help="Filtrar por diploma (CP, CC...)")
    parser.add_argument("--limpar-cache",action="store_true", help="Limpa cache de respostas LLM")
    parser.add_argument("--historico",   action="store_true", help="Ver histórico de casos")
    parser.add_argument("--query",       "-q", default="",    help="Pesquisa no histórico")
    parser.add_argument("--modo",        default=None,        help="Modo RAG: bm25|hibrido|api")
    args = parser.parse_args()

    from src.rag import MotorRAG

    modo = args.modo or "bm25"
    rag = MotorRAG(Path("."), modo=modo)

    # Stats (default se nenhum argumento)
    if args.stats or not any([
        args.reindexar, args.pesquisar, args.limpar_cache, args.historico
    ]):
        n = rag.indexar()
        stats = rag.estatisticas()

        if RICH:
            t = Table(title="Base de Conhecimento RAG — V5", border_style="blue")
            t.add_column("Métrica", style="bold")
            t.add_column("Valor", style="cyan")
            t.add_row("Total fragmentos", str(stats["total"]))
            t.add_row("Leis", str(stats["leis"]))
            t.add_row("Jurisprudência", str(stats["jurisprudencia"]))
            t.add_row("Precedentes", str(stats["precedentes"]))
            t.add_row("Diplomas indexados", ", ".join(sorted(stats["diplomas"])) or "—")
            t.add_row("Modo RAG", stats["modo"])
            t.add_row("Embeddings activos", "Sim" if stats["embeddings"] else "Não")
            console.print(t)

            if stats["fontes"]:
                t2 = Table(title="Fontes", border_style="dim")
                t2.add_column("Fonte")
                for f in sorted(stats["fontes"]):
                    t2.add_row(f)
                console.print(t2)
            else:
                console.print("\n[yellow]⚠️  Nenhuma fonte indexada.[/yellow]")
                console.print("Adiciona ficheiros .txt em data/leis/ para activar o RAG.")
        else:
            print(f"\nFragmentos: {stats['total']} (leis={stats['leis']}, jurisprudência={stats['jurisprudencia']})")
            print(f"Diplomas: {', '.join(stats['diplomas']) or 'nenhum'}")
            print(f"Modo: {stats['modo']}  |  Embeddings: {stats['embeddings']}")

    if args.reindexar:
        print("🔄 A reindexar...")
        n = rag.recarregar()
        print(f"✅ {n} fragmentos reindexados")

    if args.pesquisar:
        rag.indexar()
        frags = rag.pesquisar(
            args.pesquisar,
            n_resultados=8,
            instancia=args.instancia,
            diploma_filtro=args.diploma,
        )
        print(f"\n🔍 Pesquisa: '{args.pesquisar}'")
        if args.instancia:
            print(f"   Filtro instância: {args.instancia}")
        if args.diploma:
            print(f"   Filtro diploma: {args.diploma}")
        print(f"   {len(frags)} resultado(s)\n")

        if RICH and frags:
            t = Table(border_style="dim")
            t.add_column("Rel.", style="cyan", width=6)
            t.add_column("Tipo", width=12)
            t.add_column("Diploma", width=6)
            t.add_column("Fonte", width=25)
            t.add_column("Artigo", width=12)
            t.add_column("Excerto", width=40)
            for f in frags:
                t.add_row(
                    str(f.relevancia),
                    f.tipo,
                    f.diploma or "—",
                    f.fonte[:25],
                    f.artigo or "—",
                    f.conteudo[:80].replace("\n", " ") + "…",
                )
            console.print(t)
        elif frags:
            for i, f in enumerate(frags, 1):
                print(f"  [{i}] {f.relevancia:.3f} | {f.fonte} | {f.artigo or ''}")
                print(f"       {f.conteudo[:150]}...\n")
        else:
            print("  Sem resultados.")

    if args.limpar_cache:
        from src.cache import limpar_cache
        n = limpar_cache(dias=0)
        print(f"🗑️  Cache limpo: {n} entradas removidas")

    if args.historico:
        from src.historico import get_historico
        hist = get_historico()
        stats_h = hist.estatisticas()
        registos = hist.pesquisar(query=args.query, limite=20)

        if RICH:
            t = Table(title=f"Histórico — {stats_h['total']} casos", border_style="blue")
            t.add_column("ID", style="dim", width=22)
            t.add_column("Tribunal", width=16)
            t.add_column("Incerteza", width=10)
            t.add_column("Custo", width=10)
            t.add_column("Data", width=11)
            t.add_column("Resumo", width=35)
            for r in registos:
                t.add_row(
                    r.id, r.instancia_codigo, r.grau_incerteza,
                    f"${r.custo_usd:.4f}", r.timestamp[:10],
                    r.resumo[:35] + "…",
                )
            console.print(t)
        else:
            for r in registos:
                print(f"  {r.id} | {r.instancia_codigo} | {r.grau_incerteza} | {r.timestamp[:10]}")
                print(f"    {r.resumo[:80]}")


if __name__ == "__main__":
    main()
