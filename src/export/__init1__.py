"""
Exportação de Atas em PDF V5
─────────────────────────────
Usa ReportLab para gerar PDF formatado, com:
• Cabeçalho oficial
• Secções numeradas
• Watermark / disclaimer
• Metadados do documento
• Fallback para TXT se ReportLab não disponível
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        BaseDocTemplate, Frame, HRFlowable, PageTemplate,
        Paragraph, Spacer, Table, TableStyle,
    )
    from reportlab.platypus.tableofcontents import TableOfContents
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False


def exportar_pdf(result: "CaseResult", destino: Optional[Path] = None) -> bytes:  # type: ignore[name-defined]
    """
    Gera PDF da ata de um caso.
    Retorna bytes do PDF.
    Se ReportLab não estiver instalado, retorna TXT como bytes.
    """
    if not REPORTLAB_OK:
        return _exportar_txt_bytes(result)
    return _exportar_pdf_reportlab(result, destino)


def _exportar_txt_bytes(result: "CaseResult") -> bytes:  # type: ignore[name-defined]
    return (result.ata_final or "").encode("utf-8")


def _exportar_pdf_reportlab(result: "CaseResult", destino: Optional[Path]) -> bytes:  # type: ignore[name-defined]
    buffer = io.BytesIO()

    # Estilos
    styles = getSampleStyleSheet()

    s_titulo = ParagraphStyle(
        "titulo",
        parent=styles["Title"],
        fontSize=16,
        spaceAfter=6,
        textColor=colors.HexColor("#1a3a5c"),
        alignment=TA_CENTER,
    )
    s_subtitulo = ParagraphStyle(
        "subtitulo",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#555555"),
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    s_secao = ParagraphStyle(
        "secao",
        parent=styles["Heading2"],
        fontSize=11,
        textColor=colors.HexColor("#1a3a5c"),
        spaceBefore=14,
        spaceAfter=4,
        borderPad=3,
    )
    s_corpo = ParagraphStyle(
        "corpo",
        parent=styles["Normal"],
        fontSize=9,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    s_aviso = ParagraphStyle(
        "aviso",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#c0392b"),
        alignment=TA_CENTER,
        borderColor=colors.HexColor("#c0392b"),
        borderWidth=1,
        borderPad=6,
        backColor=colors.HexColor("#fff5f5"),
    )
    s_meta = ParagraphStyle(
        "meta",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.grey,
    )
    s_watermark = ParagraphStyle(
        "watermark",
        parent=styles["Normal"],
        fontSize=7,
        textColor=colors.HexColor("#aaaaaa"),
        alignment=TA_CENTER,
    )

    doc = BaseDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2.2 * cm,
        leftMargin=2.2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
        title=f"Ata Judicial — {result.case_id}",
        author="Tribunal IA Portugal V5",
        subject=f"Simulação Judicial — {result.instancia_nome}",
    )

    frame = Frame(
        doc.leftMargin, doc.bottomMargin,
        doc.width, doc.height,
        id="normal",
    )

    def _cabecalho_rodape(canvas, doc):
        canvas.saveState()
        # Cabeçalho
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(colors.HexColor("#1a3a5c"))
        canvas.drawString(2.2 * cm, A4[1] - 1.5 * cm,
                          "🏛 TRIBUNAL IA PORTUGAL V5 — SIMULAÇÃO EDUCATIVA")
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(A4[0] - 2.2 * cm, A4[1] - 1.5 * cm,
                               f"{result.case_id}")
        # Linha separadora
        canvas.setStrokeColor(colors.HexColor("#1a3a5c"))
        canvas.setLineWidth(0.5)
        canvas.line(2.2 * cm, A4[1] - 1.7 * cm,
                    A4[0] - 2.2 * cm, A4[1] - 1.7 * cm)
        # Rodapé
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(
            A4[0] / 2, 1.5 * cm,
            f"Pág. {doc.page} — DOCUMENTO DE SIMULAÇÃO SEM VALOR JURÍDICO — {result.trace_id}",
        )
        canvas.restoreState()

    doc.addPageTemplates([
        PageTemplate(id="main", frames=frame, onPage=_cabecalho_rodape)
    ])

    elementos = []

    # Aviso legal
    elementos.append(Paragraph(
        "⚠️ AVISO LEGAL — DOCUMENTO DE SIMULAÇÃO EDUCATIVA — NÃO CONSTITUI PARECER JURÍDICO",
        s_aviso,
    ))
    elementos.append(Spacer(1, 0.3 * cm))

    # Título
    elementos.append(Paragraph("TRIBUNAL IA PORTUGAL V5", s_titulo))
    elementos.append(Paragraph("Ata de Simulação Judicial", s_subtitulo))
    elementos.append(HRFlowable(width="100%", thickness=1.5,
                                color=colors.HexColor("#1a3a5c")))
    elementos.append(Spacer(1, 0.3 * cm))

    # Metadados em tabela
    now_str = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    meta_data = [
        ["Processo Nº", result.case_id, "Tribunal", result.instancia_nome],
        ["Trace ID", result.trace_id, "Matéria", result.instancia_codigo],
        ["Data", now_str, "Modelo", result.modelo_usado],
        ["Custo", f"{'Gratuito' if result.custo_total_usd == 0 else f'${result.custo_total_usd:.4f}'}",
         "Entidades anonimizadas", str(len(result.entities_found))],
    ]
    tabela = Table(meta_data, colWidths=[3.5 * cm, 5 * cm, 3.5 * cm, 5 * cm])
    tabela.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8eef5")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#e8eef5")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    elementos.append(tabela)
    elementos.append(Spacer(1, 0.5 * cm))

    def secao(titulo: str, conteudo: str) -> None:
        elementos.append(HRFlowable(width="100%", thickness=0.5,
                                    color=colors.HexColor("#cccccc")))
        elementos.append(Paragraph(titulo, s_secao))
        for linha in (conteudo or "").split("\n"):
            linha = linha.strip()
            if not linha:
                elementos.append(Spacer(1, 0.15 * cm))
                continue
            # Detectar cabeçalhos de secção dentro do texto
            if linha.startswith("##"):
                elementos.append(Paragraph(
                    f"<b>{linha.replace('##','').strip()}</b>",
                    ParagraphStyle("sh", parent=s_corpo, fontSize=9,
                                   textColor=colors.HexColor("#1a3a5c"),
                                   spaceBefore=6)
                ))
            elif linha.startswith("==") and linha.endswith("=="):
                elementos.append(Paragraph(
                    f"<b>{linha.replace('=','').strip()}</b>",
                    ParagraphStyle("sh2", parent=s_corpo, fontSize=9,
                                   textColor=colors.HexColor("#2d6a9f"),
                                   spaceBefore=6)
                ))
            else:
                # Escapar caracteres especiais do ReportLab
                linha = linha.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                elementos.append(Paragraph(linha, s_corpo))

    # Secções principais
    secao("I — DESCRIÇÃO DO CASO (ANONIMIZADO — RGPD)",
          result.anonymized_description)
    secao("II — RELATÓRIO DE INSTRUÇÃO FACTUAL", result.detetive_report)
    secao("III — ALEGAÇÕES DA ACUSAÇÃO", result.acusacao)
    secao("IV — ALEGAÇÕES DA DEFESA", result.defesa)
    secao(f"V — {result.instancia_codigo}: SENTENÇA RIGOROSA", result.sentenca_rigorosa)
    secao(f"VI — {result.instancia_codigo}: SENTENÇA GARANTISTA", result.sentenca_garantista)
    secao(f"VII — {result.instancia_codigo}: SENTENÇA EQUILIBRADA", result.sentenca_equilibrada)

    if result.relatorio_consistencia:
        secao("VIII — RELATÓRIO DE CONSISTÊNCIA E INCERTEZA",
              result.relatorio_consistencia)

    # Watermark final
    elementos.append(Spacer(1, 0.5 * cm))
    elementos.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#cccccc")))
    elementos.append(Paragraph(
        f"Hash do documento: {result.doc_hash} | "
        f"Gerado por Tribunal IA Portugal V5 | "
        f"Apenas para fins educativos e de simulação",
        s_watermark,
    ))
    elementos.append(Paragraph(
        "Para situações reais consulte um Advogado inscrito na Ordem dos Advogados: www.oa.pt",
        s_watermark,
    ))

    doc.build(elementos)
    pdf_bytes = buffer.getvalue()

    if destino:
        destino.write_bytes(pdf_bytes)

    return pdf_bytes


def extrair_texto_pdf(ficheiro_bytes: bytes) -> tuple[str, str]:
    """
    Extrai texto de um PDF enviado pelo utilizador.
    Retorna (texto_extraido, tipo_detectado).
    Requer PyMuPDF (fitz).
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=ficheiro_bytes, filetype="pdf")
        textos = []
        for page in doc:
            textos.append(page.get_text())
        doc.close()
        texto_completo = "\n\n".join(textos).strip()

        # Detectar tipo de documento
        tipo = _detectar_tipo_documento(texto_completo)
        return texto_completo, tipo

    except ImportError:
        return "", "PyMuPDF não instalado (pip install PyMuPDF)"
    except Exception as e:
        return "", f"Erro ao processar PDF: {e}"


def _detectar_tipo_documento(texto: str) -> str:
    t = texto.lower()
    if any(k in t for k in ["contrato", "acordo", "outorgante"]):
        return "contrato"
    if any(k in t for k in ["factura", "fatura", "invoice", "recibo"]):
        return "documento financeiro"
    if any(k in t for k in ["queixa", "participação", "denúncia"]):
        return "queixa-crime"
    if any(k in t for k in ["escritura", "registo predial", "conservatória"]):
        return "documento predial"
    if any(k in t for k in ["laudo", "relatório médico", "clínico"]):
        return "relatório médico"
    if any(k in t for k in ["certidão", "assento", "registo civil"]):
        return "certidão de registo civil"
    if any(k in t for k in ["deliberação", "despacho", "aviso"]):
        return "acto administrativo"
    return "documento jurídico"
