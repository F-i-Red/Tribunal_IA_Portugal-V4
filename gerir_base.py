#!/usr/bin/env python3
"""
gerir_base.py V6 — Gestão da base de conhecimento jurídica.
Uso:
  python gerir_base.py --stats
  python gerir_base.py --reindexar
  python gerir_base.py --pesquisar "despedimento ilícito" [--instancia TRAB]
  python gerir_base.py --pesquisar "fair trial" --lingua en
  python gerir_base.py --historico [--query "furto"]
  python gerir_base.py --limpar-cache
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    RICH = True
except ImportError:
    console = None
    RICH = False


def main():
    parser = argparse.ArgumentParser(description="Gerir base RAG V6")
    parser.add_argument("--reindexar",   action="store_true")
    parser.add_argument("--stats",       action="store_true")
    parser.add_argument("--pesquisar",   "-p", default=None)
    parser.add_argument("--instancia",   "-i", default=None)
    parser.add_argument("--diploma",     "-d", default=None)
    parser.add_argument("--lingua",      "-l", default=None, choices=["pt","en"],
                        help="Filtrar por língua (pt=leis PT, en=TEDH)")
    parser.add_argument("--limpar-cache",action="store_true")
    parser.add_argument("--historico",   action="store_true")
    parser.add_argument("--query",       "-q", default="")
    args = parser.parse_args()

    from src.utils.config import get_config
    cfg = get_config()

    from src.rag.motor import MotorRAG
    rag = MotorRAG(
        Path("."),
        modo=cfg.rag_modo,
        embedding_modelo=cfg.rag_embedding_modelo,
        reranker_modelo=cfg.rag_reranker_modelo,
        usar_reranking=cfg.rag_reranking,
        top_k=cfg.rag_top_k,
        top_n=cfg.rag_top_n,
    )

    # Stats (default)
    if args.stats or not any([args.reindexar, args.pesquisar, args.limpar_cache, args.historico]):
        n = rag.indexar()
        s = rag.estatisticas()
        if RICH:
            t = Table(title=f"RAG V6 — {n} fragmentos", border_style="blue")
            t.add_column("Métrica", style="bold"); t.add_column("Valor", style="cyan")
            t.add_row("Total fragmentos",    str(s["total"]))
            t.add_row("Leis PT",             str(s["leis"]))
            t.add_row("Jurisprudência PT",   str(s["jurisprudencia"]))
            t.add_row("Precedentes PT",      str(s["precedentes"]))
            t.add_row("TEDH / ECHR (EN)",    str(s["tedh"]))
            t.add_row("Diplomas",            ", ".join(s["diplomas"]) or "—")
            t.add_row("Modo",                s["modo"])
            t.add_row("Embeddings",          f"{s['embeddings_computados']}/{s['total']}")
            t.add_row("Reranking",           "✅ activo" if s["reranking"] else "❌ inactivo")
            t.add_row("Modelo embeddings",   s["modelo_embeddings"])
            t.add_row("Modelo reranker",     s["modelo_reranker"])
            console.print(t)
            if s["fontes"]:
                t2 = Table(title="Fontes indexadas", border_style="dim")
                t2.add_column("Fonte")
                for f in sorted(s["fontes"]):
                    t2.add_row(f)
                console.print(t2)
            else:
                console.print("\n[yellow]⚠️  Sem fontes indexadas.[/yellow]")
                console.print("Adiciona .txt em data/leis/ (PT) ou data/tedh/ (ECHR)")
        else:
            print(f"\nTotal: {s['total']} | Leis: {s['leis']} | TEDH: {s['tedh']}")
            print(f"Modo: {s['modo']} | Embeddings: {s['embeddings_computados']}")

    if args.reindexar:
        print("🔄 A reindexar...")
        n = rag.recarregar()
        print(f"✅ {n} fragmentos reindexados")

    if args.pesquisar:
        rag.indexar()
        frags = rag.pesquisar(
            args.pesquisar,
            instancia=args.instancia,
            diploma_filtro=args.diploma,
            lingua_filtro=args.lingua,
        )
        print(f"\n🔍 '{args.pesquisar}' → {len(frags)} resultado(s)")
        if args.instancia: print(f"   Instância: {args.instancia}")
        if args.diploma:   print(f"   Diploma: {args.diploma}")
        if args.lingua:    print(f"   Língua: {args.lingua}")
        print()

        if RICH and frags:
            t = Table(border_style="dim")
            t.add_column("Rel.", style="cyan", width=7)
            t.add_column("Tipo", width=12)
            t.add_column("Dip.", width=6)
            t.add_column("Lng", width=4)
            t.add_column("Fonte", width=28)
            t.add_column("Artigo", width=12)
            t.add_column("Excerto", width=45)
            for f in frags:
                t.add_row(
                    str(f.relevancia), f.tipo, f.diploma or "—", f.lingua,
                    f.fonte[:28], f.artigo or "—",
                    f.conteudo[:60].replace("\n"," ") + "…",
                )
            console.print(t)
        elif frags:
            for i, f in enumerate(frags, 1):
                print(f"  [{i}] {f.relevancia:.4f} | {f.tipo} | {f.lingua} | {f.fonte}")
                print(f"       {f.conteudo[:150]}...\n")
        else:
            print("  Sem resultados.")

    if args.limpar_cache:
        from src.cache import limpar_cache
        n = limpar_cache(dias=0)
        print(f"🗑️  Cache limpo: {n} entradas removidas")

    if args.auditoria or args.verificar_cadeia:
        from src.auditoria import get_cadeia_auditoria
        cadeia = get_cadeia_auditoria()
        if args.verificar_cadeia:
            ok, erros = cadeia.verificar_integridade()
            print(f"\nIntegridade da cadeia: {'✅ OK' if ok else '❌ COMPROMETIDA'}")
            if erros:
                for e in erros: print(f"  ⚠️  {e}")
            print(f"Total de blocos: {cadeia.resumo()['total_blocos']}")
        else:
            txt = cadeia.exportar_auditoria()
            print(txt)
            output = "cadeia_auditoria.txt"
            open(output, "w", encoding="utf-8").write(txt)
            print(f"\n✅ Cadeia exportada para: {output}")

    if args.historico:
        from src.historico import get_historico
        hist = get_historico()
        s = hist.estatisticas()
        registos = hist.pesquisar(query=args.query, limite=20)
        if RICH:
            t = Table(title=f"Histórico — {s['total']} casos", border_style="blue")
            t.add_column("ID", style="dim", width=24)
            t.add_column("Tribunal", width=10)
            t.add_column("Incerteza", width=12)
            t.add_column("Custo", width=10)
            t.add_column("Data", width=11)
            t.add_column("Resumo", width=35)
            for r in registos:
                t.add_row(r.id, r.instancia_codigo, r.grau_incerteza,
                          f"${r.custo_usd:.4f}", r.timestamp[:10], r.resumo[:35]+"…")
            console.print(t)
        else:
            for r in registos:
                print(f"  {r.id} | {r.instancia_codigo} | {r.grau_incerteza} | {r.timestamp[:10]}")


if __name__ == "__main__":
    main()
