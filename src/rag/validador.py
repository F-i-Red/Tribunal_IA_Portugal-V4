"""
Validador de citações jurídicas — V4 (sem alterações funcionais).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple


class ValidadorCitacoes:
    DIPLOMAS = {
        "CP": "Código Penal", "CPP": "Código de Processo Penal",
        "CC": "Código Civil", "CPC": "Código de Processo Civil",
        "CT": "Código do Trabalho", "CRP": "Constituição da República Portuguesa",
        "CPTA": "Código de Processo nos Tribunais Administrativos",
    }

    def __init__(self, pasta_leis: Path) -> None:
        self.pasta_leis = pasta_leis
        self._artigos_conhecidos: Dict[str, set] = {}
        self._indexado = False

    def _indexar(self) -> None:
        self._artigos_conhecidos = {}
        if not self.pasta_leis.exists():
            self._indexado = True
            return
        for ficheiro in self.pasta_leis.glob("*.txt"):
            nome = ficheiro.stem.upper()
            texto = ficheiro.read_text(encoding="utf-8", errors="replace")
            artigos = set()
            for m in re.finditer(r"Art(?:igo)?\.?\s+(\d+)\.?[º°]?", texto, re.IGNORECASE):
                artigos.add(m.group(1))
            self._artigos_conhecidos[nome] = artigos
        self._indexado = True

    def extrair_citacoes(self, texto: str) -> List[Tuple[str, str]]:
        if not self._indexado:
            self._indexar()
        citacoes = []
        for m in re.finditer(
            r"art(?:igo)?\.?\s*(\d+)\.?[º°]?[A-Za-z]?"
            r"(?:\s+(?:do|da|n\.?[º°]\s*\d+))*"
            r"(?:\s+(?:do\s+)?([A-Z]{2,5}))?",
            texto, re.IGNORECASE,
        ):
            num = m.group(1)
            diploma = (m.group(2) or "").upper()
            if num and 1 <= int(num) <= 999:
                citacoes.append((num, diploma))
        for m in re.finditer(r"\[art\.?\?(?:\s+([A-Z]{2,5}))?\]", texto):
            citacoes.append(("?", m.group(1) or ""))
        return list(set(citacoes))

    def validar_texto(self, texto: str) -> Tuple[str, List[Dict]]:
        if not self._indexado:
            self._indexar()
        citacoes = self.extrair_citacoes(texto)
        problemas: List[Dict] = []
        mapa = {
            "CP": ["PENAL"], "CPP": ["PROCESSO_PENAL", "PROCESSO"],
            "CC": ["CIVIL"], "CPC": ["PROCESSO_CIVIL"],
            "CT": ["TRABALHO"], "CRP": ["CONSTITUICAO"],
            "CPTA": ["ADMINISTRATIVO"],
        }
        for num, diploma in citacoes:
            if num == "?":
                problemas.append({"artigo": f"[art.? {diploma}]", "status": "incerto",
                                  "mensagem": "Artigo marcado como incerto pelo modelo"})
                continue
            verificado = any(
                ((diploma and any(t in n for t in mapa.get(diploma, [diploma]))) or (not diploma))
                and num in arts
                for n, arts in self._artigos_conhecidos.items()
            )
            if not verificado and self._artigos_conhecidos:
                problemas.append({
                    "artigo": f"art. {num}.º {diploma}".strip(),
                    "status": "nao_verificado",
                    "mensagem": f"Art. {num} não encontrado nos ficheiros locais",
                })
        return texto, problemas

    def relatorio_citacoes(self, problemas: List[Dict]) -> str:
        if not problemas:
            return "✅ Todas as citações verificadas nos ficheiros locais."
        linhas = [f"⚠️  {len(problemas)} citação(ões) a verificar:"]
        for p in problemas:
            linhas.append(f"  • {p['artigo']}: {p['mensagem']}")
        linhas.append("\nNota: Citações não verificadas podem estar correctas mas ausentes dos ficheiros locais.")
        return "\n".join(linhas)
