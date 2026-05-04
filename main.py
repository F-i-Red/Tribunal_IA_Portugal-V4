#!/usr/bin/env python3
"""
Tribunal IA Portugal V5 — CLI moderna com typer + rich
Uso:
  python main.py processar "descrição do caso"
  python main.py processar --instancia TRAB --sem-instrucao
  python main.py historico
  python main.py modelos
  python main.py verificar
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.markdown import Markdown
    from rich import print as rprint
    RICH_OK = True
except ImportError:
    RICH_OK = False
    # Fallback mínimo sem typer/rich
    import argparse

console = Console() if RICH_OK else None

BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║  🏛️  TRIBUNAL IA PORTUGAL V5  🇵🇹                                    ║
║  Simulador judicial — Direito Português                              ║
║  ⚠️  Fins exclusivamente educativos e de simulação                  ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def _banner():
    if console:
        console.print(Panel(
            "[bold #1a3a5c]TRIBUNAL IA PORTUGAL V5[/bold #1a3a5c]\n"
            "[dim]Simulador judicial · Direito Português 🇵🇹[/dim]\n"
            "[yellow]⚠️  Fins exclusivamente educativos e de simulação[/yellow]",
            border_style="bold blue",
        ))
    else:
        print(BANNER)


def _verificar_config():
    from src.utils.config import get_config, ConfigError
    try:
        return get_config()
    except ConfigError as e:
        if console:
            console.print(f"[bold red]❌ CONFIGURAÇÃO:[/bold red] {e}")
        else:
            print(f"ERRO: {e}")
        print("\nCria .env com:")
        print("  OPENROUTER_API_KEY=a_tua_chave")
        print("  MODELO=openrouter/auto")
        print("\nOu para Ollama local:")
        print("  BACKEND=ollama")
        print("  OLLAMA_MODELO=llama3.3:70b")
        sys.exit(1)


def _mostrar_instancias():
    from src.pipeline.instancias import INSTANCIAS
    if console:
        t = Table(title="Instâncias Judiciais Disponíveis", border_style="blue")
        t.add_column("Código", style="bold cyan", width=12)
        t.add_column("Tribunal", style="white")
        t.add_column("Matéria", style="dim")
        for cod, inst in INSTANCIAS.items():
            t.add_row(cod, inst.nome_curto, inst.materia)
        console.print(t)
    else:
        from src.pipeline.instancias import listar_instancias_menu
        print(listar_instancias_menu())


def _mostrar_modelos():
    from src.utils.config import FREE_MODELS, PAID_MODELS
    if console:
        t = Table(title="Modelos OpenRouter", border_style="blue")
        t.add_column("Modelo", style="cyan")
        t.add_column("Tipo", style="bold")
        t.add_column("Custo estimado/caso", style="dim")
        for m in FREE_MODELS:
            t.add_row(m, "[green]GRÁTIS[/green]", "€0.00")
        for m, (ip, op) in list(PAID_MODELS.items())[:8]:
            est = round((ip * 3000 + op * 5000) / 1_000_000, 4)
            t.add_row(m, "[blue]PAGO[/blue]", f"~${est:.4f}")
        console.print(t)
        console.print("\n[dim]Para Ollama local: BACKEND=ollama no .env[/dim]")
    else:
        print("\nModelos gratuitos (openrouter):")
        for m in FREE_MODELS:
            print(f"  🆓 {m}")
        print("\nModelos pagos (openrouter):")
        for m in list(PAID_MODELS.keys())[:8]:
            print(f"  💳 {m}")


if RICH_OK:
    app = typer.Typer(
        name="tribunal",
        help="Tribunal IA Portugal V5 — Simulador judicial",
        add_completion=False,
    )

    @app.command("processar")
    def cmd_processar(
        caso: str = typer.Argument(None, help="Descrição do caso (ou interactivo)"),
        instancia: str = typer.Option(None, "--instancia", "-i", help="Código do tribunal"),
        modelo: str = typer.Option(None, "--modelo", "-m", help="Override do modelo"),
        backend: str = typer.Option(None, "--backend", "-b", help="openrouter ou ollama"),
        sem_instrucao: bool = typer.Option(False, "--sem-instrucao", help="Salta instrução"),
        sem_pdf: bool = typer.Option(False, "--sem-pdf", help="Não gerar PDF"),
    ):
        """Processa um caso judicial completo."""
        import os
        _banner()
        cfg = _verificar_config()

        if modelo:
            os.environ["MODELO"] = modelo
        if backend:
            os.environ["BACKEND"] = backend

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
                    console.print("\n[yellow]Cancelado.[/yellow]")
                    raise typer.Exit()
            caso = "\n".join(linhas)

        if not caso.strip():
            console.print("[red]Descrição vazia.[/red]")
            raise typer.Exit(1)

        from src.pipeline.instancias import detectar_instancia_por_keywords, INSTANCIAS
        if not instancia:
            instancia = detectar_instancia_por_keywords(caso)
        inst = INSTANCIAS.get(instancia, INSTANCIAS["TIC"])
        console.print(f"\n[bold]🏛️  Tribunal:[/bold] {inst.nome}")
        console.print(f"[bold]🤖 Modelo:[/bold]   {cfg.modelo_activo} [{cfg.backend}]")

        # Instrução
        dados_instrucao = None
        if not sem_instrucao:
            with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prog:
                t = prog.add_task("A gerar perguntas de instrução...", total=None)
                try:
                    from src.pipeline.case_processor import CaseProcessor
                    proc = CaseProcessor()
                    perguntas = proc.gerar_perguntas_instrucao(caso, instancia)
                    prog.update(t, description="✅ Perguntas geradas")
                except Exception as e:
                    prog.update(t, description=f"⚠️  Instrução falhou: {str(e)[:60]}")
                    perguntas = {"perguntas": []}

            respostas = {}
            for p in perguntas.get("perguntas", []):
                badge = {"critica": "🔴", "relevante": "🟡", "complementar": "🟢"}.get(p.get("importancia", ""), "⚪")
                console.print(f"\n  {badge} [{p.get('categoria','?')}] {p.get('texto','')}")
                resp = input("  ➜ ").strip()
                respostas[p["id"]] = {
                    "pergunta": p.get("texto", ""),
                    "categoria": p.get("categoria", ""),
                    "resposta": resp or "Sem resposta",
                }
            if respostas:
                dados_instrucao = {"respostas": respostas, "materiais": []}

        # Processar
        console.print("\n")
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prog:
            t = prog.add_task("⚖️  Processo judicial em curso...", total=None)
            try:
                from src.pipeline.case_processor import CaseProcessor
                import time
                start = time.time()
                proc = CaseProcessor()
                result = proc.process(
                    case_description=caso,
                    instancia_codigo=instancia,
                    dados_instrucao=dados_instrucao,
                    gerar_pdf=not sem_pdf,
                )
                elapsed = time.time() - start
                prog.update(t, description="✅ Concluído")
            except Exception as e:
                prog.update(t, description=f"❌ Erro: {str(e)[:80]}")
                console.print(f"\n[red]{e}[/red]")
                raise typer.Exit(1)

        # Resultado
        custo = "Gratuito" if result.custo_total_usd == 0 else f"${result.custo_total_usd:.4f}"
        console.print(Panel(
            f"[bold]ID:[/bold] {result.case_id}\n"
            f"[bold]Tribunal:[/bold] {result.instancia_nome}\n"
            f"[bold]Tempo:[/bold] {elapsed:.1f}s | [bold]Custo:[/bold] {custo}\n"
            f"[bold]Entidades anonimizadas:[/bold] {len(result.entities_found)}\n"
            f"[bold]Grau de incerteza:[/bold] {result.grau_incerteza}\n"
            f"[bold]Ata:[/bold] {result.ata_path or 'não guardada'}",
            title="✅ Processo Concluído",
            border_style="green",
        ))

        import re
        def disp(txt):
            if not txt:
                return "N/D"
            m = re.search(r"(?:CONDENA|ABSOLVE|JULGA)[^.]*\.", txt, re.IGNORECASE)
            return (m.group(0) if m else txt[:120]) + "..."

        t2 = Table(title="Resumo das Decisões", border_style="dim")
        t2.add_column("Perfil", style="bold")
        t2.add_column("Dispositivo")
        t2.add_row("🔴 Rigoroso", disp(result.sentenca_rigorosa))
        t2.add_row("🟢 Garantista", disp(result.sentenca_garantista))
        t2.add_row("🔵 Equilibrado", disp(result.sentenca_equilibrada))
        console.print(t2)

        if result.pdf_bytes:
            console.print(f"[green]📄 PDF:[/green] {result.ata_path.with_suffix('.pdf') if result.ata_path else 'gerado'}")

    @app.command("historico")
    def cmd_historico(
        query: str = typer.Option("", "--query", "-q", help="Pesquisa"),
        instancia: str = typer.Option(None, "--instancia", "-i"),
        limite: int = typer.Option(10, "--limite", "-n"),
    ):
        """Lista o histórico de casos processados."""
        _banner()
        _verificar_config()
        from src.historico import get_historico
        hist = get_historico()
        registos = hist.pesquisar(query=query, instancia=instancia, limite=limite)
        stats = hist.estatisticas()

        console.print(f"\n[bold]📋 Histórico:[/bold] {stats['total']} casos\n")
        if not registos:
            console.print("[dim]Sem resultados.[/dim]")
            return
        t = Table(border_style="dim")
        t.add_column("ID", style="dim", width=22)
        t.add_column("Tribunal", width=18)
        t.add_column("Incerteza", width=10)
        t.add_column("Custo", width=10)
        t.add_column("Data", width=12)
        for r in registos:
            t.add_row(
                r.id, r.instancia_codigo, r.grau_incerteza,
                f"${r.custo_usd:.4f}", r.timestamp[:10],
            )
        console.print(t)

    @app.command("modelos")
    def cmd_modelos():
        """Lista modelos disponíveis."""
        _banner()
        _mostrar_modelos()

    @app.command("instancias")
    def cmd_instancias():
        """Lista instâncias judiciais disponíveis."""
        _banner()
        _mostrar_instancias()

    @app.command("verificar")
    def cmd_verificar():
        """Diagnóstico do ambiente."""
        _banner()
        import subprocess
        subprocess.run([sys.executable, "verificar.py"], check=False)

    if __name__ == "__main__":
        app()

else:
    # Fallback sem typer
    def main_fallback():
        _banner()
        cfg = _verificar_config()
        print(f"\nModelo: {cfg.modelo_activo} [{cfg.backend}]")
        print("\nDescreve o caso (linha vazia para terminar):")
        linhas = []
        while True:
            try:
                linha = input("> ")
                if not linha.strip() and linhas:
                    break
                if linha.strip():
                    linhas.append(linha)
            except (EOFError, KeyboardInterrupt):
                sys.exit(0)
        caso = "\n".join(linhas)
        if not caso.strip():
            sys.exit(1)

        from src.pipeline.case_processor import CaseProcessor
        import time
        print("\n⚖️  A processar...")
        start = time.time()
        proc = CaseProcessor()
        result = proc.process(caso)
        elapsed = time.time() - start
        print(f"\n✅ {result.case_id} | {elapsed:.1f}s | Incerteza: {result.grau_incerteza}")
        print(f"Ata: {result.ata_path}")

    if __name__ == "__main__":
        main_fallback()
