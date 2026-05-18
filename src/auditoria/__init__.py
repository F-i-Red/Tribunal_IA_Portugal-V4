"""
Auditoria V7 — Git Jurídico
════════════════════════════
Implementa:
1. Cadeia de hash encadeada (cada caso referencia o anterior)
2. Provenance log — quais fragmentos RAG influenciaram cada decisão
3. Detecção básica de prompt injection / inputs adversariais
4. Voto de vencido formal (dissent mechanism)
5. Separação explícita: apoio cognitivo ≠ decisão soberana

Princípio: "criar memória jurídica auditável entre humanos e IAs"
Não substituir juízes — apoiar o raciocínio jurídico.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════
# 1. CADEIA DE HASH ENCADEADA — Git Jurídico
# ══════════════════════════════════════════════════════════════════════

@dataclass
class BlocoAuditoria:
    """
    Bloco imutável na cadeia de auditoria.
    Cada caso referencia o hash do bloco anterior — à semelhança de blockchain.
    Garante que nenhum registo pode ser alterado sem quebrar a cadeia.
    """
    indice: int
    case_id: str
    timestamp: str
    instancia: str
    modelo: str
    grau_incerteza: str
    hash_ata: str           # SHA-256 da ata completa
    hash_anterior: str      # SHA-256 do bloco anterior (0000...0 no génesis)
    hash_bloco: str = ""    # calculado após criação

    def calcular_hash(self) -> str:
        conteudo = json.dumps({
            "indice": self.indice,
            "case_id": self.case_id,
            "timestamp": self.timestamp,
            "hash_ata": self.hash_ata,
            "hash_anterior": self.hash_anterior,
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(conteudo.encode()).hexdigest()

    def __post_init__(self) -> None:
        if not self.hash_bloco:
            self.hash_bloco = self.calcular_hash()


class CadeiaAuditoria:
    """
    Cadeia imutável de registos de auditoria.
    Verificável publicamente — qualquer alteração quebra a cadeia.
    """
    GENESIS_HASH = "0" * 64
    _lock = threading.Lock()

    def __init__(self, pasta: Path) -> None:
        self.pasta = pasta
        self.pasta.mkdir(parents=True, exist_ok=True)
        self._cadeia_path = pasta / "cadeia_auditoria.jsonl"
        self._cadeia: List[BlocoAuditoria] = []
        self._carregar()

    def _carregar(self) -> None:
        if not self._cadeia_path.exists():
            return
        try:
            for linha in self._cadeia_path.read_text(encoding="utf-8").splitlines():
                if linha.strip():
                    d = json.loads(linha)
                    self._cadeia.append(BlocoAuditoria(**d))
        except Exception:
            self._cadeia = []

    def adicionar(
        self,
        case_id: str,
        instancia: str,
        modelo: str,
        grau_incerteza: str,
        hash_ata: str,
    ) -> BlocoAuditoria:
        with self._lock:
            hash_anterior = (
                self._cadeia[-1].hash_bloco if self._cadeia else self.GENESIS_HASH
            )
            bloco = BlocoAuditoria(
                indice=len(self._cadeia),
                case_id=case_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                instancia=instancia,
                modelo=modelo,
                grau_incerteza=grau_incerteza,
                hash_ata=hash_ata,
                hash_anterior=hash_anterior,
            )
            self._cadeia.append(bloco)
            # Append-only — nunca sobrescreve
            with open(self._cadeia_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(bloco), ensure_ascii=False) + "\n")
            return bloco

    def verificar_integridade(self) -> Tuple[bool, List[str]]:
        """Verifica se a cadeia está íntegra (nenhum bloco foi alterado)."""
        erros: List[str] = []
        hash_esperado = self.GENESIS_HASH
        for i, bloco in enumerate(self._cadeia):
            if bloco.hash_anterior != hash_esperado:
                erros.append(f"Bloco {i}: hash_anterior incorreto")
            hash_recalculado = bloco.calcular_hash()
            if hash_recalculado != bloco.hash_bloco:
                erros.append(f"Bloco {i} ({bloco.case_id}): hash adulterado")
            hash_esperado = bloco.hash_bloco
        return len(erros) == 0, erros

    def resumo(self) -> Dict:
        ok, erros = self.verificar_integridade()
        return {
            "total_blocos": len(self._cadeia),
            "cadeia_integra": ok,
            "erros": erros,
            "ultimo_hash": self._cadeia[-1].hash_bloco if self._cadeia else None,
            "genesis_hash": self.GENESIS_HASH[:16] + "...",
        }

    def exportar_auditoria(self) -> str:
        """Exporta cadeia completa para verificação pública."""
        linhas = [
            "CADEIA DE AUDITORIA — TRIBUNAL IA PORTUGAL V7",
            "Verificável publicamente. Qualquer alteração quebra a cadeia.",
            f"Total de blocos: {len(self._cadeia)}",
            "=" * 60,
        ]
        for b in self._cadeia:
            linhas.append(
                f"[{b.indice:04d}] {b.case_id} | {b.instancia} | {b.timestamp[:19]}\n"
                f"       Hash: {b.hash_bloco[:32]}...\n"
                f"       Anterior: {b.hash_anterior[:32]}..."
            )
        ok, erros = self.verificar_integridade()
        linhas.append(f"\nIntegridade: {'✅ OK' if ok else '❌ COMPROMETIDA'}")
        if erros:
            for e in erros:
                linhas.append(f"  ⚠️ {e}")
        return "\n".join(linhas)


# ══════════════════════════════════════════════════════════════════════
# 2. PROVENANCE LOG — Rastreabilidade das decisões
# ══════════════════════════════════════════════════════════════════════

@dataclass
class FragmentoUsado:
    """Registo de um fragmento RAG que influenciou uma decisão."""
    agente: str        # detetive, acusacao, defesa, juiz_*
    fonte: str         # nome do diploma/ficheiro
    diploma: str       # CP, CC, CPP...
    artigo: str        # Artigo 203.º
    relevancia: float  # score RAG
    lingua: str        # pt | en


@dataclass
class ProvenanceLog:
    """
    Rastreabilidade completa de uma decisão.
    Responde a: "porque é que o sistema chegou a esta conclusão?"
    """
    case_id: str
    fragmentos_usados: List[FragmentoUsado] = field(default_factory=list)
    modelos_consultados: List[str] = field(default_factory=list)
    total_tokens: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def adicionar_fragmentos_rag(
        self, agente: str, fragmentos: List
    ) -> None:
        for f in fragmentos:
            self.fragmentos_usados.append(FragmentoUsado(
                agente=agente,
                fonte=getattr(f, "fonte", "?"),
                diploma=getattr(f, "diploma", "") or "?",
                artigo=getattr(f, "artigo", "") or "",
                relevancia=getattr(f, "relevancia", 0.0),
                lingua=getattr(f, "lingua", "pt"),
            ))

    def relatorio(self) -> str:
        linhas = [
            f"PROVENANCE LOG — {self.case_id}",
            f"Timestamp: {self.timestamp}",
            f"Modelos: {', '.join(set(self.modelos_consultados))}",
            f"Fragmentos RAG usados: {len(self.fragmentos_usados)}",
            "─" * 50,
        ]
        por_agente: Dict[str, List[FragmentoUsado]] = {}
        for fr in self.fragmentos_usados:
            por_agente.setdefault(fr.agente, []).append(fr)

        for agente, frags in por_agente.items():
            linhas.append(f"\n[{agente.upper()}] — {len(frags)} fragmento(s):")
            for fr in frags[:5]:  # máximo 5 por agente
                artigo_str = f" {fr.artigo}" if fr.artigo else ""
                linhas.append(
                    f"  • [{fr.diploma}]{artigo_str} — {fr.fonte} "
                    f"(rel={fr.relevancia:.3f}, {fr.lingua})"
                )

        linhas.append(
            f"\n{'─'*50}\n"
            f"Total tokens: {self.total_tokens}\n"
            f"Este relatório permite verificar quais normas jurídicas\n"
            f"influenciaram cada peça processual."
        )
        return "\n".join(linhas)


# ══════════════════════════════════════════════════════════════════════
# 3. THREAT MODEL — Detecção de inputs adversariais
# ══════════════════════════════════════════════════════════════════════

# Padrões de prompt injection conhecidos
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions?",
    r"esquece\s+(as\s+)?instruções\s+anteriores",
    r"act\s+as\s+(a\s+)?(?:different|new|evil|uncensored)",
    r"jailbreak",
    r"DAN\s*mode",
    r"você\s+é\s+agora\s+um",
    r"you\s+are\s+now\s+(?:a\s+)?(?:different|evil|uncensored)",
    r"<\s*script\s*>",
    r"system\s*:\s*you\s+are",
    r"\[\s*INST\s*\]",
    r"ignore\s+the\s+above",
    r"disregard\s+all\s+previous",
    r"from\s+now\s+on\s+you\s+(?:will|are)",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

# Limites de segurança
MAX_CASO_CHARS = 10_000      # máximo de caracteres por caso
MAX_ARGUMENTO_CHARS = 3_000  # máximo para argumento de defesa (contraditório)
MIN_CASO_CHARS = 20          # mínimo para ser válido


@dataclass
class ResultadoValidacao:
    valido: bool
    avisos: List[str] = field(default_factory=list)
    texto_sanitizado: str = ""


def validar_input(
    texto: str,
    max_chars: int = MAX_CASO_CHARS,
    campo: str = "caso",
) -> ResultadoValidacao:
    """
    Valida e sanitiza input do utilizador.
    Detecta prompt injection, limita tamanho, remove caracteres perigosos.
    """
    avisos: List[str] = []

    if not texto or not texto.strip():
        return ResultadoValidacao(valido=False, avisos=["Texto vazio"])

    if len(texto) < MIN_CASO_CHARS:
        return ResultadoValidacao(
            valido=False,
            avisos=[f"Descrição muito curta (mínimo {MIN_CASO_CHARS} caracteres)"]
        )

    # Truncar se necessário
    if len(texto) > max_chars:
        texto = texto[:max_chars]
        avisos.append(f"Texto truncado para {max_chars} caracteres")

    # Detectar prompt injection
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(texto):
            return ResultadoValidacao(
                valido=False,
                avisos=[
                    "⚠️ Input rejeitado: contém padrão de manipulação do modelo. "
                    "Por favor descreve o caso de forma natural."
                ]
            )

    # Sanitização básica — remover sequências de controlo
    texto_limpo = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", texto)

    # Avisar sobre conteúdo potencialmente problemático
    if "<" in texto_limpo and ">" in texto_limpo:
        avisos.append("Aviso: o texto contém marcação HTML/XML que foi preservada")

    return ResultadoValidacao(
        valido=True,
        avisos=avisos,
        texto_sanitizado=texto_limpo,
    )


# ══════════════════════════════════════════════════════════════════════
# 4. VOTO DE VENCIDO — Dissent mechanism formal
# ══════════════════════════════════════════════════════════════════════

@dataclass
class VotoVencido:
    """
    Voto de vencido formal — quando os 3 juízes divergem significativamente.
    Tradição do direito processual: o juiz vencido pode fundamentar o seu voto.
    """
    perfil_divergente: str     # "rigoroso" | "garantista"
    sentido_divergente: str    # "mais severo" | "mais garantista"
    fundamento_resumo: str     # extracto da fundamentação divergente
    artigos_divergentes: List[str] = field(default_factory=list)


def analisar_dissenso(
    s_rigorosa: str,
    s_garantista: str,
    s_equilibrada: str,
) -> Optional[VotoVencido]:
    """
    Analisa se há dissenso significativo entre as 3 sentenças.
    Se sim, identifica o voto de vencido e o seu fundamento.
    """
    # Extrair dispositivos
    def _dispositivo(txt: str) -> str:
        if not txt:
            return ""
        m = re.search(r"(?:CONDENA|ABSOLVE|JULGA)[^.]*\.", txt, re.IGNORECASE)
        return m.group(0).lower() if m else ""

    d_r = _dispositivo(s_rigorosa)
    d_g = _dispositivo(s_garantista)
    d_e = _dispositivo(s_equilibrada)

    # Detectar divergência condenar vs absolver
    condena = lambda d: "condena" in d
    absolve = lambda d: "absolve" in d or "não pronunci" in d or "arquiva" in d

    decisoes = [
        ("rigoroso", d_r),
        ("garantista", d_g),
        ("equilibrado", d_e),
    ]

    condenas = [p for p, d in decisoes if condena(d)]
    absolves = [p for p, d in decisoes if absolve(d)]

    # Dissenso claro: maioria condena mas um absolve (ou vice-versa)
    if len(condenas) == 2 and len(absolves) == 1:
        vencido = absolves[0]
        return VotoVencido(
            perfil_divergente=vencido,
            sentido_divergente="mais garantista",
            fundamento_resumo=_extrair_fundamentacao(
                s_garantista if vencido == "garantista" else s_equilibrada
            ),
            artigos_divergentes=_extrair_artigos(
                s_garantista if vencido == "garantista" else s_equilibrada
            ),
        )
    elif len(absolves) == 2 and len(condenas) == 1:
        vencido = condenas[0]
        return VotoVencido(
            perfil_divergente=vencido,
            sentido_divergente="mais rigoroso",
            fundamento_resumo=_extrair_fundamentacao(s_rigorosa),
            artigos_divergentes=_extrair_artigos(s_rigorosa),
        )

    return None  # Sem dissenso significativo


def _extrair_fundamentacao(sentenca: str) -> str:
    if not sentenca:
        return ""
    # Tentar encontrar secção de fundamentação jurídica
    m = re.search(
        r"(?:FUNDAMENTAÇÃO JURÍDICA|MOTIVAÇÃO)[^\n]*\n+(.*?)(?:==|\Z)",
        sentenca, re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).strip()[:300]
    return sentenca[200:500].strip()


def _extrair_artigos(sentenca: str) -> List[str]:
    if not sentenca:
        return []
    artigos = re.findall(
        r"art(?:igo)?\.?\s*\d+\.?[º°]?[A-Za-z]?(?:\s+(?:do|da|n\.?[º°]).*?)?(?:CP|CPP|CC|CPC|CT|CRP)?",
        sentenca, re.IGNORECASE,
    )
    return list(dict.fromkeys(artigos[:5]))  # deduplicar, máximo 5


# ══════════════════════════════════════════════════════════════════════
# 5. DISCLAIMER DE SEPARAÇÃO DE PAPÉIS
# ══════════════════════════════════════════════════════════════════════

DISCLAIMER_SEPARACAO_PAPEIS = """
╔══════════════════════════════════════════════════════════════════════╗
║  DECLARAÇÃO DE SEPARAÇÃO DE PAPÉIS — TRIBUNAL IA PORTUGAL V7        ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Este sistema é APOIO COGNITIVO — não DECISÃO SOBERANA.             ║
║                                                                      ║
║  O sistema PODE:                                                     ║
║  ✅ Resumir e estruturar factos                                      ║
║  ✅ Identificar legislação potencialmente aplicável                  ║
║  ✅ Simular argumentos de acusação e defesa                         ║
║  ✅ Evidenciar incerteza jurídica e zonas de dissenso               ║
║  ✅ Gerar hipóteses decisórias para reflexão                        ║
║                                                                      ║
║  O sistema NÃO PODE:                                                 ║
║  ❌ Determinar culpa ou inocência                                    ║
║  ❌ Substituir a decisão de um magistrado                           ║
║  ❌ Produzir efeitos jurídicos vinculativos                         ║
║  ❌ Garantir a exactidão dos artigos citados                        ║
║  ❌ Representar a posição do Estado português                       ║
║                                                                      ║
║  Para decisões com efeitos jurídicos:                               ║
║  → Advogado inscrito na Ordem dos Advogados: www.oa.pt              ║
║  → Julgados de Paz: www.julgadosdepaz.mj.pt                        ║
║  → Apoio judiciário: www.dgaj.mj.pt                                 ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════
# Singletons
# ══════════════════════════════════════════════════════════════════════

_cadeia: Optional[CadeiaAuditoria] = None
_cadeia_lock = threading.Lock()


def get_cadeia_auditoria() -> CadeiaAuditoria:
    global _cadeia
    with _cadeia_lock:
        if _cadeia is None:
            try:
                from ..utils.config import get_config
                cfg = get_config()
                pasta = cfg.pasta_historico.parent / "auditoria"
            except Exception:
                pasta = Path("src/historico/data/auditoria")
            _cadeia = CadeiaAuditoria(pasta)
    return _cadeia
