"""
Definição completa das instâncias judiciais portuguesas.
V4: sem alterações — modelo da V3 é completo e correcto.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class InstanciaJudicial:
    codigo: str
    nome: str
    nome_curto: str
    materia: str
    descricao: str
    termo_acusado: str
    termo_vitima: str
    termo_mp: str
    termo_defesa: str
    termo_decisao: str
    diploma_principal: str
    keywords: List[str]


INSTANCIAS: Dict[str, InstanciaJudicial] = {

    "TIC": InstanciaJudicial(
        codigo="TIC",
        nome="Tribunal de Instrução Criminal",
        nome_curto="T. Instrução Criminal",
        materia="Penal — Fase de Instrução",
        descricao="Controla a legalidade da investigação criminal e decide sobre pronúncia ou não pronúncia.",
        termo_acusado="arguido", termo_vitima="assistente / ofendido",
        termo_mp="Magistrado do Ministério Público", termo_defesa="defensor / patrono",
        termo_decisao="despacho de pronúncia / não pronúncia",
        diploma_principal="Código de Processo Penal (CPP)",
        keywords=["crime", "arguido", "penal", "furto", "roubo", "ofensa corporal",
                  "homicídio", "ameaça", "coação", "difamação", "violação", "tráfico",
                  "corrupção", "instrução criminal", "queixa", "denúncia", "inquérito"],
    ),

    "TCCR": InstanciaJudicial(
        codigo="TCCR",
        nome="Tribunal Coletivo / Singular Criminal",
        nome_curto="Tribunal Criminal",
        materia="Penal — Julgamento",
        descricao="Julgamento de crimes. Singular até 5 anos; Coletivo acima.",
        termo_acusado="arguido", termo_vitima="assistente / ofendido",
        termo_mp="Procurador do Ministério Público", termo_defesa="defensor",
        termo_decisao="sentença / acórdão",
        diploma_principal="CPP + Código Penal (CP)",
        keywords=["julgamento criminal", "acusação formal", "pronúncia", "audiência"],
    ),

    "TCIC": InstanciaJudicial(
        codigo="TCIC",
        nome="Tribunal Central de Instrução Criminal",
        nome_curto="TCIC",
        materia="Penal — Grande Criminalidade",
        descricao="Especializado para criminalidade organizada, terrorismo, corrupção de grande dimensão.",
        termo_acusado="arguido", termo_vitima="assistente / ofendido",
        termo_mp="Procurador da República / DCIAP", termo_defesa="defensor",
        termo_decisao="despacho de pronúncia",
        diploma_principal="CPP + Lei de Organização do Sistema Judiciário",
        keywords=["terrorismo", "criminalidade organizada", "lavagem", "branqueamento",
                  "dciap", "financeiro", "peculato"],
    ),

    "TC_CIVEL": InstanciaJudicial(
        codigo="TC_CIVEL",
        nome="Tribunal Judicial de Comarca — Juízo Cível",
        nome_curto="Tribunal Cível",
        materia="Cível — Direito Privado",
        descricao="Litígios entre privados: contratos, propriedade, responsabilidade civil.",
        termo_acusado="réu / demandado", termo_vitima="autor / demandante",
        termo_mp="Ministério Público (incapazes)", termo_defesa="mandatário judicial",
        termo_decisao="sentença",
        diploma_principal="Código Civil (CC) + Código de Processo Civil (CPC)",
        keywords=["contrato", "dívida", "indemnização", "danos", "propriedade",
                  "arrendamento", "responsabilidade civil", "incumprimento",
                  "execução", "penhora", "cível"],
    ),

    "TFM": InstanciaJudicial(
        codigo="TFM",
        nome="Tribunal de Família e Menores",
        nome_curto="T. Família e Menores",
        materia="Família / Menores",
        descricao="Divórcio, responsabilidades parentais, alimentos, adopção.",
        termo_acusado="requerido", termo_vitima="requerente",
        termo_mp="Ministério Público", termo_defesa="mandatário",
        termo_decisao="sentença / decisão",
        diploma_principal="CC + CPC + RGPTC + LPCJP",
        keywords=["divórcio", "separação", "filho", "menor", "alimentos",
                  "responsabilidades parentais", "guarda", "tutela", "adopção",
                  "família", "casamento", "união de facto"],
    ),

    "TRAB": InstanciaJudicial(
        codigo="TRAB",
        nome="Tribunal do Trabalho",
        nome_curto="T. Trabalho",
        materia="Laboral",
        descricao="Litígios laborais: despedimentos, salários, acidentes de trabalho.",
        termo_acusado="réu / entidade empregadora", termo_vitima="autor / trabalhador",
        termo_mp="Ministério Público", termo_defesa="mandatário",
        termo_decisao="sentença",
        diploma_principal="Código do Trabalho (CT) + Código de Processo do Trabalho (CPT)",
        keywords=["trabalho", "laboral", "despedimento", "salário", "vencimento",
                  "contrato de trabalho", "férias", "subsídio", "acidente trabalho",
                  "assédio laboral", "sindicato", "empregador"],
    ),

    "TAF": InstanciaJudicial(
        codigo="TAF",
        nome="Tribunal Administrativo e Fiscal",
        nome_curto="T. Administrativo",
        materia="Administrativo / Fiscal",
        descricao="Litígios com o Estado: actos administrativos, impostos, urbanismo.",
        termo_acusado="entidade demandada / administração", termo_vitima="autor / contribuinte",
        termo_mp="Ministério Público", termo_defesa="mandatário",
        termo_decisao="sentença / acórdão",
        diploma_principal="CPTA + CPPT + CPA",
        keywords=["estado", "câmara municipal", "imposto", "irs", "irc", "iva",
                  "coima", "multa", "licença", "urbanismo", "expropriação",
                  "funcionário público", "acto administrativo"],
    ),

    "TCOM": InstanciaJudicial(
        codigo="TCOM",
        nome="Tribunal de Comércio",
        nome_curto="T. Comércio",
        materia="Comercial / Insolvência",
        descricao="Insolvências, recuperação de empresas, propriedade industrial.",
        termo_acusado="insolvente / réu", termo_vitima="credor / autor",
        termo_mp="Ministério Público", termo_defesa="mandatário",
        termo_decisao="sentença / despacho",
        diploma_principal="CIRE + CSC + CódComercial",
        keywords=["insolvência", "falência", "empresa", "sociedade", "per", "peap",
                  "credor", "recuperação", "liquidação", "dissolução", "gerente",
                  "propriedade industrial", "marca", "patente", "comercial"],
    ),

    "TR": InstanciaJudicial(
        codigo="TR",
        nome="Tribunal da Relação",
        nome_curto="Tribunal da Relação",
        materia="2ª Instância — Recurso",
        descricao="Segunda instância para recursos de decisões de 1ª instância.",
        termo_acusado="recorrido / arguido", termo_vitima="recorrente / assistente",
        termo_mp="Procurador-Geral Adjunto", termo_defesa="mandatário",
        termo_decisao="acórdão",
        diploma_principal="CPP / CPC (consoante a matéria)",
        keywords=["recurso", "apelação", "relação", "segunda instância", "impugnar"],
    ),

    "STJ": InstanciaJudicial(
        codigo="STJ",
        nome="Supremo Tribunal de Justiça",
        nome_curto="STJ",
        materia="3ª Instância — Direito",
        descricao="Última instância cível e penal. Conhece apenas de direito.",
        termo_acusado="recorrido", termo_vitima="recorrente",
        termo_mp="Procurador-Geral da República", termo_defesa="mandatário",
        termo_decisao="acórdão",
        diploma_principal="CPP / CPC + Estatuto STJ",
        keywords=["supremo tribunal", "stj", "terceira instância", "revista",
                  "uniformização"],
    ),

    "TC": InstanciaJudicial(
        codigo="TC",
        nome="Tribunal Constitucional",
        nome_curto="T. Constitucional",
        materia="Constitucional",
        descricao="Fiscaliza a constitucionalidade das normas jurídicas.",
        termo_acusado="entidade que aplica a norma", termo_vitima="recorrente / requerente",
        termo_mp="Procurador-Geral da República", termo_defesa="mandatário",
        termo_decisao="acórdão",
        diploma_principal="Constituição da República Portuguesa (CRP) + LTC",
        keywords=["constitucional", "inconstitucionalidade", "direitos fundamentais",
                  "crp", "tribunal constitucional", "fiscalização"],
    ),
}


def listar_instancias_menu() -> str:
    linhas: List[str] = []
    grupos = [
        ("⚖️  PENAL", ["TIC", "TCCR", "TCIC"]),
        ("📋 CÍVEL", ["TC_CIVEL"]),
        ("👨‍👩‍👧 FAMÍLIA", ["TFM"]),
        ("💼 TRABALHO", ["TRAB"]),
        ("🏛️  ADMINISTRATIVO / FISCAL", ["TAF"]),
        ("🏢 COMERCIAL", ["TCOM"]),
        ("🔼 RECURSOS", ["TR", "STJ"]),
        ("📜 CONSTITUCIONAL", ["TC"]),
    ]
    for grupo, codigos in grupos:
        linhas.append(f"\n  {grupo}")
        for c in codigos:
            inst = INSTANCIAS[c]
            linhas.append(f"    [{c:8s}] {inst.nome_curto:<35} — {inst.materia}")
    return "\n".join(linhas)


def detectar_instancia_por_keywords(texto: str) -> str:
    t = texto.lower()
    scores: Dict[str, int] = {k: 0 for k in INSTANCIAS}
    for codigo, inst in INSTANCIAS.items():
        for kw in inst.keywords:
            if kw in t:
                scores[codigo] += 1
    for exc in ["TR", "STJ", "TC", "TCIC"]:
        scores[exc] = 0
    melhor = max(scores, key=lambda k: scores[k])
    return melhor if scores[melhor] > 0 else "TIC"
