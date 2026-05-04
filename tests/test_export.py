"""Testes de exportação V5."""
import pytest
from src.export import extrair_texto_pdf, _detectar_tipo_documento


def test_detectar_tipo_contrato():
    t = "Este contrato é celebrado entre os outorgantes abaixo identificados."
    assert _detectar_tipo_documento(t) == "contrato"


def test_detectar_tipo_financeiro():
    t = "FATURA Nº 2024/001 — Total a pagar: 1.500,00 EUR"
    assert _detectar_tipo_documento(t) == "documento financeiro"


def test_detectar_tipo_queixa():
    t = "Queixa-crime apresentada contra X por furto."
    assert _detectar_tipo_documento(t) == "queixa-crime"


def test_detectar_tipo_predial():
    t = "Certidão da Conservatória do Registo Predial — escritura pública."
    assert _detectar_tipo_documento(t) == "documento predial"


def test_detectar_tipo_medico():
    t = "Relatório médico — o utente apresenta lesões compatíveis com o laudo."
    assert _detectar_tipo_documento(t) == "relatório médico"


def test_detectar_tipo_generico():
    t = "Texto sem palavras-chave específicas de nenhuma categoria."
    assert _detectar_tipo_documento(t) == "documento jurídico"


def test_extrair_texto_pdf_bytes_invalidos():
    """Bytes inválidos devem retornar erro gracioso."""
    texto, tipo = extrair_texto_pdf(b"nao_e_um_pdf")
    # Deve retornar string (possivelmente vazia ou com msg de erro)
    assert isinstance(texto, str)
    assert isinstance(tipo, str)


def test_exportar_pdf_fallback_sem_reportlab():
    """Se ReportLab não estiver instalado, deve retornar TXT como bytes."""
    from unittest.mock import patch, MagicMock
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class MockResult:
        case_id: str = "caso_test"
        trace_id: str = "trace123"
        anonymized_description: str = "Caso de teste anonimizado"
        instancia_codigo: str = "TIC"
        instancia_nome: str = "Tribunal de Instrução Criminal"
        modelo_usado: str = "test"
        backend_usado: str = "test"
        entities_found: list = None
        detetive_report: str = "Relatório de instrução"
        acusacao: str = "Alegações da acusação"
        defesa: str = "Alegações da defesa"
        sentenca_rigorosa: str = "Sentença rigorosa"
        sentenca_garantista: str = "Sentença garantista"
        sentenca_equilibrada: str = "Sentença equilibrada"
        relatorio_consistencia: str = "Relatório de consistência"
        ata_final: str = "Ata completa do processo"
        ata_path: Optional[object] = None
        pdf_bytes: Optional[bytes] = None
        custo_total_usd: float = 0.0
        doc_hash: str = "abc123"

        def __post_init__(self):
            if self.entities_found is None:
                self.entities_found = []

    result = MockResult()

    with patch("src.export.REPORTLAB_OK", False):
        from src.export import exportar_pdf
        pdf_bytes = exportar_pdf(result)
        assert isinstance(pdf_bytes, bytes)
        assert b"Ata completa" in pdf_bytes
