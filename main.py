#!/usr/bin/env python3
"""
Tribunal IA Portugal V6 — CLI
Uso:
  python main.py processar "caso..."
  python main.py processar --instancia TRAB --contraditorio
  python main.py historico [--query "texto"] [--instancia TIC]
  python main.py rag --stats
  python main.py modelos
  python main.py instancias
  python main.py api [--port 8000]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_OK = True
except ImportError:
    RICH_OK = False

console = Console() if RICH_OK else None


def _banner() -> None:
    if console:
        console.print(Panel(
            "[bold #1a3a5c]TRIBUNAL IA PORTUGAL V6[/bold #1a3a5c]\n"
            "[dim]Simulador judicial · RAG Híbrido + Reranking · LangGraph[/dim]\n"
            "[yellow]⚠️  Fins exclusivamente educativos[/yellow]",
            border_style="bold blue",
        ))
    else:
        print("\n🏛️  TRIBUNAL IA PORTUGAL V6\n")


def _cfg():
    from src.utils.config import get_config, ConfigError
    try:
        return get_config()
    except ConfigError as e:
        if console:
            console.print(f"[bold red]❌ Configuração:[/bold red] {e}")
        else:
            print(f"ERRO: {e}")
        sys.exit(1)


if RICH_OK:
    app = typer.Typer(name="tribunal", help="Tribunal IA Portugal V6", add_completion=False)

    @app.command("processar")
    def cmd_processar(
        caso: str = typer.Argument(None, help="Descrição do caso"),
        instancia: str = typer.Option(None, "-i", "--instancia"),
        modelo: str = typer.Option(None, "-m", "--modelo"),
        backend: str = typer.Option(None, "-b", "--backend"),
        contraditorio: bool = typer.Option(False, "--contraditorio", help="Activar modo contraditório"),
        sem_instrucao: bool = typer.Option(False, "--sem-instrucao"),
        sem_pdf: bool = typer.Option(False, "--sem-pdf"),
    ):
        """Processa um caso judicial completo."""
        _banner()
        if modelo: os.environ["MODELO"] = modelo
        if backend: os.environ["BACKEND"] = backend
        cfg = _cfg()

        if not caso:
            console.print("\n[bold]📝 Descreve o caso[/bold] [dim](linha vazia para terminar)[/dim]")
            linhas = []
            while True:
                try:
                    linha = input("> ")
                    if not linha.strip() and linhas:
                        break
                    if linha.strip():
                        linhas.append(linha)
                except (EOFError, KeyboardInterrupt):
                    raise typer.Exit()
            caso = "\n".join(linhas)

        if not caso.strip():
            console.print("[red]Descrição vazia.[/red]"); raise typer.Exit(1)

        from src.pipeline.instancias import detectar_instancia_por_keywords, INSTANCIAS
        if not instancia:
            instancia = detectar_instancia_por_keywords(caso)
        inst = INSTANCIAS.get(instancia, INSTANCIAS["TIC"])

        console.print(f"\n[bold]🏛️  Tribunal:[/bold] {inst.nome}")
        console.print(f"[bold]🤖 Modelo:[/bold]   {cfg.modelo_activo} [{cfg.backend}]")
        orq = "LangGraph" if cfg.usar_langgraph else "Imperativo"
        console.print(f"[bold]🔀 Orquestração:[/bold] {orq} | RAG: {cfg.rag_modo}")

        # Instrução
        dados_instrucao = None
        if not sem_instrucao:
            with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prog:
                t = prog.add_task("A gerar perguntas de instrução...", total=None)
                try:
                    from src.pipeline.case_processor import CaseProcessor
                    proc = CaseProcessor()
                    perguntas = proc.gerar_perguntas_instrucao(caso, instancia)
                    prog.update(t, description=f"✅ {len(perguntas.get('perguntas',[]))} perguntas geradas")
                except Exception as e:
                    prog.update(t, description=f"⚠️  Instrução falhou: {str(e)[:50]}")
                    perguntas = {"perguntas": []}

            respostas = {}
            for p in perguntas.get("perguntas", []):
                badge = {"critica":"🔴","relevante":"🟡","complementar":"🟢"}.get(p.get("importancia",""),"⚪")
                console.print(f"\n  {badge} [{p.get('categoria','?')}] {p.get('texto','')}")
                if p.get("razao"):
                    console.print(f"  [dim]  ↳ {p['razao']}[/dim]")
                resp = input("  ➜ ").strip()
                respostas[p["id"]] = {
                    "pergunta": p.get("texto",""),
                    "categoria": p.get("categoria",""),
                    "resposta": resp or "Sem resposta",
                }
            if respostas:
                dados_instrucao = {"respostas": respostas, "materiais": []}

        # Contraditório
        intervencao = None
        if contraditorio:
            console.print("\n[bold yellow]⚔️  MODO CONTRADITÓRIO[/bold yellow]")
            console.print("A gerar acusação para o contraditório...")

            # Gerar acusação primeiro
            from src.pipeline.case_processor import CaseProcessor
            from src.utils import anonymize_text
            proc_c = CaseProcessor()
            anon, _ = anonymize_text(caso)
            ctx = proc_c._rag_ctx(anon, instancia=instancia)
            inst_obj = INSTANCIAS.get(instancia, INSTANCIAS["TIC"])
            det = proc_c._detetive.executar(anon, "", ctx, inst_obj)
            acus = proc_c._acusacao.executar(anon, det, ctx, inst_obj)

            console.print("\n[bold]📄 ACUSAÇÃO:[/bold]")
            console.print(Panel(acus[:800] + "..." if len(acus) > 800 else acus,
                                border_style="red"))

            console.print("\n[bold]🛡️  O teu argumento de defesa:[/bold] [dim](linha vazia para terminar)[/dim]")
            linhas_arg = []
            while True:
                try:
                    linha = input("  > ")
                    if not linha.strip() and linhas_arg:
                        break
                    if linha.strip():
                        linhas_arg.append(linha)
                except (EOFError, KeyboardInterrupt):
                    break
            intervencao = "\n".join(linhas_arg) if linhas_arg else None

        # Processar
        console.print()
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prog:
            t2 = prog.add_task("⚖️  Processo judicial em curso (V6)...", total=None)
            try:
                start = time.time()
                from src.pipeline.case_processor import CaseProcessor
                proc2 = CaseProcessor()
                result = proc2.process(
                    case_description=caso,
                    instancia_codigo=instancia,
                    dados_instrucao=dados_instrucao,
                    gerar_pdf=not sem_pdf,
                    intervencao_utilizador=intervencao,
                )
                elapsed = time.time() - start
                prog.update(t2, description="✅ Concluído")
            except Exception as e:
                prog.update(t2, description=f"❌ Erro")
                console.print(f"\n[red]{e}[/red]"); raise typer.Exit(1)

        custo = "Gratuito" if result.custo_total_usd == 0 else f"${result.custo_total_usd:.4f}"
        orq_real = "LangGraph" if cfg.usar_langgraph else "Imperativo"
        console.print(Panel(
            f"[bold]ID:[/bold] {result.case_id}\n"
            f"[bold]Tribunal:[/bold] {result.instancia_nome}\n"
            f"[bold]Orquestração:[/bold] {orq_real}\n"
            f"[bold]Tempo:[/bold] {elapsed:.1f}s | [bold]Custo:[/bold] {custo}\n"
            f"[bold]Entidades RGPD:[/bold] {len(result.entities_found)}\n"
            f"[bold]Grau de incerteza:[/bold] {result.grau_incerteza}\n"
            f"[bold]TEDH:[/bold] {'✅' if result.analise_tedh else '—'}\n"
            f"[bold]Ata:[/bold] {result.ata_path or 'não guardada'}",
            title="✅ Processo Concluído", border_style="green",
        ))

        import re
        def disp(txt: str) -> str:
            if not txt: return "N/D"
            m = re.search(r"(?:CONDENA|ABSOLVE|JULGA)[^.]*\.", txt, re.IGNORECASE)
            return (m.group(0) if m else txt[:120]) + "..."

        t3 = Table(title="Resumo das Decisões", border_style="dim")
        t3.add_column("Perfil", style="bold", width=15)
        t3.add_column("Dispositivo")
        t3.add_row("🔴 Rigoroso",    disp(result.sentenca_rigorosa))
        t3.add_row("🟢 Garantista",  disp(result.sentenca_garantista))
        t3.add_row("🔵 Equilibrado", disp(result.sentenca_equilibrada))
        console.print(t3)

        if result.analise_tedh:
            console.print("\n[bold]🌍 Análise TEDH (resumo):[/bold]")
            console.print(result.analise_tedh[:400] + "...")

    @app.command("historico")
    def cmd_historico(
        query: str = typer.Option("", "-q", "--query"),
        instancia: str = typer.Option(None, "-i"),
        limite: int = typer.Option(10, "-n"),
    ):
        """Histórico de casos processados."""
        _banner(); _cfg()
        from src.historico import get_historico
        hist = get_historico()
        registos = hist.pesquisar(query=query, instancia=instancia, limite=limite)
        stats = hist.estatisticas()
        console.print(f"\n[bold]📋 Total:[/bold] {stats['total']} casos\n")
        if not registos:
            console.print("[dim]Sem resultados.[/dim]"); return
        t = Table(border_style="dim")
        t.add_column("ID", style="dim", width=24)
        t.add_column("Tribunal", width=10)
        t.add_column("Incerteza", width=12)
        t.add_column("Custo", width=10)
        t.add_column("Data", width=12)
        for r in registos:
            t.add_row(r.id, r.instancia_codigo, r.grau_incerteza,
                      f"${r.custo_usd:.4f}", r.timestamp[:10])
        console.print(t)

    @app.command("rag")
    def cmd_rag(
        stats: bool = typer.Option(True, "--stats"),
        pesquisar: str = typer.Option(None, "-p", "--pesquisar"),
        instancia: str = typer.Option(None, "-i"),
        reindexar: bool = typer.Option(False, "--reindexar"),
    ):
        """Gestão do motor RAG."""
        _banner(); cfg = _cfg()
        from src.rag.motor import MotorRAG
        rag = MotorRAG(Path("."), modo=cfg.rag_modo,
                       embedding_modelo=cfg.rag_embedding_modelo,
                       reranker_modelo=cfg.rag_reranker_modelo,
                       usar_reranking=cfg.rag_reranking,
                       top_k=cfg.rag_top_k, top_n=cfg.rag_top_n)

        if reindexar:
            with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prog:
                t = prog.add_task("A reindexar...", total=None)
                n = rag.recarregar()
                prog.update(t, description=f"✅ {n} fragmentos")

        if stats or not pesquisar:
            n = rag.indexar()
            s = rag.estatisticas()
            t2 = Table(title=f"RAG V6 — {n} fragmentos", border_style="blue")
            t2.add_column("Métrica"); t2.add_column("Valor", style="cyan")
            t2.add_row("Total", str(s["total"]))
            t2.add_row("Leis", str(s["leis"]))
            t2.add_row("Jurisprudência", str(s["jurisprudencia"]))
            t2.add_row("TEDH / ECHR", str(s["tedh"]))
            t2.add_row("Diplomas", ", ".join(s["diplomas"]) or "—")
            t2.add_row("Modo", s["modo"])
            t2.add_row("Embeddings computados", str(s["embeddings_computados"]))
            t2.add_row("Reranking activo", "✅" if s["reranking"] else "❌")
            t2.add_row("Modelo embeddings", s["modelo_embeddings"])
            t2.add_row("Modelo reranker", s["modelo_reranker"])
            console.print(t2)

        if pesquisar:
            rag.indexar()
            frags = rag.pesquisar(pesquisar, instancia=instancia)
            console.print(f"\n🔍 '{pesquisar}' → {len(frags)} resultado(s)\n")
            t3 = Table(border_style="dim")
            t3.add_column("Rel.", width=6, style="cyan")
            t3.add_column("Tipo", width=12)
            t3.add_column("Diploma", width=8)
            t3.add_column("Lingua", width=6)
            t3.add_column("Fonte", width=25)
            t3.add_column("Excerto", width=45)
            for f in frags:
                t3.add_row(str(f.relevancia), f.tipo, f.diploma or "—",
                           f.lingua, f.fonte[:25],
                           f.conteudo[:60].replace("\n"," ") + "…")
            console.print(t3)

    @app.command("modelos")
    def cmd_modelos():
        """Lista modelos disponíveis."""
        _banner()
        from src.utils.config import FREE_MODELS, PAID_MODELS
        t = Table(title="Modelos OpenRouter", border_style="blue")
        t.add_column("Modelo", style="cyan"); t.add_column("Tipo"); t.add_column("Custo/caso")
        for m in sorted(FREE_MODELS):
            t.add_row(m, "[green]GRÁTIS[/green]", "€0.00")
        for m, (ip, op) in list(PAID_MODELS.items()):
            est = round((ip*3000+op*5000)/1_000_000, 4)
            t.add_row(m, "[blue]PAGO[/blue]", f"~${est:.4f}")
        console.print(t)
        console.print("\n[dim]Para Ollama local: BACKEND=ollama no .env[/dim]")

    @app.command("instancias")
    def cmd_instancias():
        """Lista instâncias judiciais."""
        _banner()
        from src.pipeline.instancias import INSTANCIAS
        t = Table(title="Instâncias Judiciais", border_style="blue")
        t.add_column("Código", style="bold cyan", width=12)
        t.add_column("Tribunal")
        t.add_column("Matéria")
        t.add_column("Diploma", style="dim")
        for cod, inst in INSTANCIAS.items():
            t.add_row(cod, inst.nome_curto, inst.materia, inst.diploma_principal)
        console.print(t)

    @app.command("api")
    def cmd_api(
        host: str = typer.Option(None, "--host"),
        port: int = typer.Option(None, "--port"),
        reload: bool = typer.Option(False, "--reload"),
    ):
        """Inicia o servidor API REST."""
        _banner()
        import subprocess
        args = [sys.executable, "api_server.py"]
        if host: args += ["--host", host]
        if port: args += ["--port", str(port)]
        if reload: args += ["--reload"]
        subprocess.run(args)

    @app.command("verificar")
    def cmd_verificar():
        """Diagnóstico do ambiente."""
        _banner()
        import subprocess
        subprocess.run([sys.executable, "verificar.py"], check=False)

    if __name__ == "__main__":
        app()

else:
    # Fallback simples sem typer/rich
    if __name__ == "__main__":
        print("\n🏛️  TRIBUNAL IA PORTUGAL V6\n")
        cfg = _cfg()
        print(f"Modelo: {cfg.modelo_activo} | RAG: {cfg.rag_modo}")
        print("\nDescreve o caso (linha vazia para terminar):")
        linhas = []
        while True:
            try:
                linha = input("> ")
                if not linha.strip() and linhas: break
                if linha.strip(): linhas.append(linha)
            except (EOFError, KeyboardInterrupt): sys.exit(0)
        caso = "\n".join(linhas)
        if not caso.strip(): sys.exit(1)
        from src.pipeline.case_processor import CaseProcessor
        start = time.time()
        result = CaseProcessor().process(caso)
        print(f"\n✅ {result.case_id} | {time.time()-start:.1f}s | {result.grau_incerteza}")
