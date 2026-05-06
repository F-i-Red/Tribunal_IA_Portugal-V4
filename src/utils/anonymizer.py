"""
Anonimização de entidades sensíveis — RGPD.
Mascarar dados ANTES de enviar para APIs de terceiros.
V4: sem alterações funcionais — o modelo da V3 é robusto.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Entity:
    text: str
    start: int
    end: int
    label: str
    score: float


class PortugueseLegalAnonymizer:
    STRUCTURED_PATTERNS = {
        "EMAIL":         re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "IBAN":          re.compile(r"\bPT\d{2}[\s\d]{20,26}\b"),
        "CODIGO_POSTAL": re.compile(r"\b\d{4}-\d{3}\b"),
        "CC":            re.compile(r"\b\d{8}\s*[A-Za-z]{2}\d?\b"),
        "TELEFONE":      re.compile(r"\b(?:\+351[\s-]?)?(?:9[1236]\d{7}|2\d{8})\b"),
        "NIF":           re.compile(r"\b[1235689]\d{8}\b"),
        "NISS":          re.compile(r"\b\d{11}\b"),
        "PROCESSO":      re.compile(r"\b\d{3,4}/\d{2}[.\d]*\b"),
    }

    CIDADES = {
        "lisboa", "porto", "braga", "coimbra", "faro", "aveiro", "guarda", "leiria",
        "viseu", "bragança", "évora", "beja", "portalegre", "setúbal", "santarém",
        "viana do castelo", "vila real", "funchal", "ponta delgada", "amadora",
        "almada", "oeiras", "sintra", "cascais", "loures", "odivelas",
        "vila nova de gaia", "matosinhos", "maia", "gondomar", "portimão", "tavira",
        "evora", "setubal", "santarem",
    }

    PREFIXOS_FORMAIS = [
        r"arguid[oa]\s+", r"r[eé]u\s+", r"autor[ea]?\s+", r"vítima\s+",
        r"ofendid[oa]\s+", r"testemunha\s+", r"sr\.?\s+", r"sra\.?\s+",
        r"dr\.?\s+", r"dra\.?\s+", r"doutor[ea]?\s+", r"professor[ea]?\s+",
        r"advogad[oa]?\s+", r"juiz[a]?\s+", r"senhor[a]?\s+",
        r"menor\s+", r"cônjuge\s+", r"companheiro[a]?\s+",
    ]

    PADROES_INFORMAIS = [
        r"(?:sou\s+(?:o|a)\s+|chamo[- ]me\s+)([A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+(?:\s+[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+){0,3})",
        r"(?:o\s+vizinho|a\s+vizinha|o\s+senhorio|a\s+senhora|o\s+senhor)\s+(?:\w+\s+)?\(([A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+(?:\s+[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+){0,3})\)",
        r"\(([A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+(?:\s+[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+){0,3})\)",
        r"(?:denominad[oa]|alcunhad[oa]|conhecid[oa]\s+(?:por|como))\s+([A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+(?:\s+[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+){0,3})",
    ]

    NAO_NOMES = {
        "tribunal", "juiz", "advogado", "procurador", "ministerio", "direito",
        "lei", "codigo", "artigo", "processo", "sentenca", "nacional", "republica",
        "portuguesa", "europeu", "estado", "comarca", "distrito", "concelho",
        "qualificado", "furto", "delito", "crime", "testemunhas", "antecedentes",
        "em", "na", "no", "de", "do", "da", "para", "por", "foi", "tem", "sao",
        "era", "esta", "fica", "uma", "um", "duas", "dois", "tres", "quatro",
        "cinco", "seis", "sete", "oito", "nove", "dez", "aqui", "ali", "isso",
        "isto", "aquilo", "quando", "onde", "como", "porque", "mas", "pois",
    }

    def __init__(self, salt: str = "tribunal_ia_v4"):
        self.salt = salt

    def _pseudonimo(self, label: str, original: str) -> str:
        key = f"{self.salt}:{label}:{original.lower().strip()}"
        h = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 9000 + 1000
        mapa = {
            "PESSOA":        f"[PESSOA_{h}]",
            "LOCAL":         f"[LOCAL_{h}]",
            "MORADA":        f"[MORADA_{h}]",
            "ORGANIZACAO":   f"[ENTIDADE_{h}]",
            "NIF":           "[NIF_REMOVIDO]",
            "CC":            "[CC_REMOVIDO]",
            "NISS":          "[NISS_REMOVIDO]",
            "TELEFONE":      "[TELEFONE_REMOVIDO]",
            "EMAIL":         "[EMAIL_REMOVIDO]",
            "IBAN":          "[IBAN_REMOVIDO]",
            "CODIGO_POSTAL": "[CP_REMOVIDO]",
            "PROCESSO":      f"[PROCESSO_{h}]",
            "DATA_NASCIMENTO": "[DATA_NASC_REMOVIDA]",
        }
        return mapa.get(label, f"[{label}_{h}]")

    def _valido_nome(self, nome: str) -> bool:
        palavras = nome.split()
        if len(nome) < 3 or len(palavras) > 5:
            return False
        if palavras[0].lower() in self.NAO_NOMES:
            return False
        meio_stop = {"por", "em", "de", "para", "com", "sem", "sob", "foi",
                     "tem", "são", "era", "está", "fica", "e", "ou"}
        for w in palavras[1:]:
            if w.lower() in meio_stop:
                return False
        return True

    def _encontrar_estruturados(self, text: str) -> List[Entity]:
        entities: List[Entity] = []
        for label in ["EMAIL", "IBAN", "CODIGO_POSTAL", "CC", "TELEFONE", "NIF", "NISS", "PROCESSO"]:
            for m in self.STRUCTURED_PATTERNS[label].finditer(text):
                entities.append(Entity(m.group(), m.start(), m.end(), label, 0.97))
        return entities

    def _encontrar_nomes_formais(self, text: str) -> List[Entity]:
        entities: List[Entity] = []
        nome_re = r"([A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+(?:\s+[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+){0,4})"
        for prefix in self.PREFIXOS_FORMAIS:
            pat = re.compile(f"(?:{prefix}){nome_re}", re.IGNORECASE)
            for m in pat.finditer(text):
                nome = m.group(1)
                if self._valido_nome(nome):
                    entities.append(Entity(nome, m.start(1), m.end(1), "PESSOA", 0.90))
        return entities

    def _encontrar_nomes_informais(self, text: str) -> List[Entity]:
        entities: List[Entity] = []
        for padrao in self.PADROES_INFORMAIS:
            for m in re.finditer(padrao, text, re.IGNORECASE):
                nome = m.group(1)
                if self._valido_nome(nome):
                    entities.append(Entity(nome, m.start(1), m.end(1), "PESSOA", 0.85))
        return entities

    def _encontrar_locais(self, text: str) -> List[Entity]:
        entities: List[Entity] = []
        for m in re.finditer(
            r"Tribunal\s+(?:(?:da|do|de|Central)\s+)?[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+"
            r"(?:\s+(?:de\s+)?[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+)*",
            text, re.IGNORECASE
        ):
            entities.append(Entity(m.group(), m.start(), m.end(), "LOCAL", 0.95))
        for m in re.finditer(
            r"(?:Rua|Avenida|Av\.?|Praça|Largo|Travessa|Estrada|Alameda|Calçada|Beco)"
            r"\s+[^,.\n]{3,60}(?:,\s*[^,.\n]{2,40}){0,3}",
            text, re.IGNORECASE
        ):
            if len(m.group().split()) >= 2:
                entities.append(Entity(m.group().strip(), m.start(), m.end(), "MORADA", 0.85))
        for cidade in self.CIDADES:
            for m in re.finditer(rf"\b{re.escape(cidade)}\b", text, re.IGNORECASE):
                entities.append(Entity(m.group(), m.start(), m.end(), "LOCAL", 0.80))
        return entities

    def anonymize(self, text: str) -> Tuple[str, List[Entity]]:
        todas: List[Entity] = []
        todas.extend(self._encontrar_estruturados(text))
        todas.extend(self._encontrar_nomes_formais(text))
        todas.extend(self._encontrar_nomes_informais(text))
        todas.extend(self._encontrar_locais(text))

        todas.sort(key=lambda e: e.start)
        filtradas: List[Entity] = []
        ultimo_fim = -1
        for ent in todas:
            if ent.start >= ultimo_fim:
                filtradas.append(ent)
                ultimo_fim = ent.end

        resultado = text
        for ent in reversed(filtradas):
            pseudo = self._pseudonimo(ent.label, ent.text)
            resultado = resultado[:ent.start] + pseudo + resultado[ent.end:]

        return resultado, filtradas


def anonymize_text(text: str) -> Tuple[str, List[Entity]]:
    return PortugueseLegalAnonymizer().anonymize(text)
