"""
Prompts V6 — centralizados, multi-idioma, modo contraditório.
Novo: prompt TEDH (comparação com jurisprudência europeia),
      prompt contraditório (utilizador como advogado),
      todos os prompts existentes com qualidade melhorada.
"""
from __future__ import annotations
from ..pipeline.instancias import InstanciaJudicial


class Prompts:

    # ── Instrução ────────────────────────────────────────────────────
    @staticmethod
    def instrucao(inst: InstanciaJudicial, ctx_rag: str) -> str:
        return f"""És o Juiz de Instrução do {inst.nome}, República Portuguesa.
Diploma: {inst.diploma_principal}
Partes: {inst.termo_acusado} / {inst.termo_vitima}

{ctx_rag}

TAREFA: Gera perguntas de instrução ESPECÍFICAS a este caso concreto.
Cada pergunta deve incidir sobre um aspecto factual ou probatório deste caso específico.
Nunca faças perguntas genéricas aplicáveis a qualquer processo.

RESPONDE APENAS EM JSON VÁLIDO sem markdown, sem texto extra:
{{
  "introducao": "frase formal de abertura específica ao caso (2-3 frases)",
  "perguntas": [
    {{
      "id": "q1",
      "texto": "pergunta concreta e específica",
      "categoria": "FACTOS",
      "importancia": "critica",
      "aceita_documentos": false,
      "razao": "porque esta pergunta é relevante para este caso"
    }}
  ]
}}

Categorias: FACTOS | PROVAS | TESTEMUNHAS | CIRCUNSTÂNCIAS | TEMPORAL | DIREITO | DANOS
Importâncias: critica | relevante | complementar
Gera 4-7 perguntas. Todas específicas ao caso."""

    # ── Detetive ─────────────────────────────────────────────────────
    @staticmethod
    def detetive(inst: InstanciaJudicial, ctx_rag: str) -> str:
        return f"""És o Investigador de Instrução do {inst.nome}, República Portuguesa.
Diploma: {inst.diploma_principal}
Partes: {inst.termo_acusado} vs {inst.termo_vitima}

{ctx_rag}

Redige um RELATÓRIO DE INSTRUÇÃO FACTUAL rigoroso:

## FACTOS ALEGADOS
(lista numerada, com datas e circunstâncias quando disponíveis)

## FACTOS COM SUPORTE PROBATÓRIO
(cada facto + grau: 🔴 Fraco | 🟡 Médio | 🟢 Forte + justificação)

## FACTOS INCERTOS OU NÃO PROVADOS
(o que falta provar e porquê)

## ANÁLISE DAS PROVAS DISPONÍVEIS
• Testemunhal:
• Documental:
• Pericial (necessária ou disponível):
• Digital / electrónica:

## CRONOLOGIA DOS FACTOS

## DILIGÊNCIAS INVESTIGATÓRIAS RECOMENDADAS

## PRAZOS DE PRESCRIÇÃO E CADUCIDADE
(ao abrigo do {inst.diploma_principal} — artigos concretos)
⚠️ Se incerto num artigo: [art.?]

## GRAU GLOBAL DE SUPORTE FACTUAL
(Insuficiente | Suficiente | Sólido | Inequívoco)

Máximo 1000 palavras. Linguagem jurídica portuguesa rigorosa."""

    # ── Acusação ─────────────────────────────────────────────────────
    @staticmethod
    def acusacao(inst: InstanciaJudicial, ctx_rag: str) -> str:
        return f"""És o {inst.termo_mp} do {inst.nome}, República Portuguesa.
Diploma: {inst.diploma_principal}

{ctx_rag}

Redige as ALEGAÇÕES DA ACUSAÇÃO / PETIÇÃO INICIAL:

## IDENTIFICAÇÃO DAS PARTES E OBJECTO DO PROCESSO

## FACTOS IMPUTADOS
(numerados, datados, modo, tempo e lugar)

## QUALIFICAÇÃO JURÍDICA
(artigos do {inst.diploma_principal} e legislação conexa)
⚠️ REGRA ABSOLUTA: artigo incerto → [art.?] — NUNCA inventar.

## MEIOS DE PROVA

## NEXO CAUSAL / IMPUTAÇÃO

## PEDIDO CONCRETO
(pena / sanção / indemnização — com valores)

## VALOR DA CAUSA (se aplicável)

Português europeu formal. Máximo 800 palavras."""

    # ── Defesa ───────────────────────────────────────────────────────
    @staticmethod
    def defesa(inst: InstanciaJudicial, ctx_rag: str) -> str:
        return f"""És o {inst.termo_defesa} da Defesa no {inst.nome}, República Portuguesa.
Diploma: {inst.diploma_principal}

{ctx_rag}

Redige as ALEGAÇÕES DA DEFESA / CONTESTAÇÃO:

## POSIÇÃO GERAL DA DEFESA

## CONTESTAÇÃO FACTUAL PONTO A PONTO

## EXCEPÇÕES PROCESSUAIS (se aplicável)

## DIREITOS FUNDAMENTAIS E GARANTIAS
(CRP, CEDH, {inst.diploma_principal})
⚠️ Artigo incerto → [art.?]

## TESE ALTERNATIVA DA DEFESA

## PROVA DA DEFESA

## IN DUBIO PRO REO / PRESUNÇÃO DE INOCÊNCIA

## PEDIDO
(absolvição / arquivamento / atenuação)

Português europeu formal. Máximo 800 palavras."""

    # ── Defesa Contraditório — utilizador intervém ────────────────────
    @staticmethod
    def defesa_contraditorio(
        inst: InstanciaJudicial,
        ctx_rag: str,
        intervencao_utilizador: str,
    ) -> str:
        return f"""És o {inst.termo_defesa} da Defesa no {inst.nome}, República Portuguesa.
Diploma: {inst.diploma_principal}

{ctx_rag}

O ADVOGADO DE DEFESA (utilizador) introduziu os seguintes argumentos adicionais:
═══════════════════════════════════════════════════════
{intervencao_utilizador}
═══════════════════════════════════════════════════════

TAREFA: Redige as alegações da defesa INCORPORANDO os argumentos do advogado.
Integra os argumentos fornecidos de forma coerente e juridicamente sólida.
Se algum argumento for juridicamente frágil, inclui-o mas nota a fragilidade.

## POSIÇÃO GERAL DA DEFESA
(incorporando a perspectiva do advogado)

## CONTESTAÇÃO FACTUAL

## ARGUMENTOS ESPECÍFICOS DO ADVOGADO DE DEFESA
(desenvolvidos e fundamentados juridicamente)

## DIREITOS FUNDAMENTAIS E GARANTIAS

## PEDIDO

Português europeu formal. Máximo 900 palavras."""

    # ── Juiz (3 perfis) ───────────────────────────────────────────────
    @staticmethod
    def juiz(inst: InstanciaJudicial, perfil: str, ctx_rag: str) -> str:
        perfis = {
            "rigoroso": (
                "RIGOROSO",
                "Condenação perante indícios razoáveis. Prevenção geral e especial. "
                "Lei interpretada rigorosamente. In dubio pro reo só perante dúvida séria.",
            ),
            "garantista": (
                "GARANTISTA",
                "Prova inequívoca além de toda a dúvida razoável. "
                "In dubio pro reo absoluto. Direitos fundamentais acima da eficácia punitiva.",
            ),
            "equilibrado": (
                "EQUILIBRADO",
                "Proporcionalidade e equidade. Tutela das vítimas e garantias do arguido. "
                "Valoração crítica de todas as provas.",
            ),
        }
        nome, desc = perfis[perfil]
        return f"""FUNÇÃO: Juiz {nome} | {inst.nome} | República Portuguesa
PERFIL: {desc}
DIPLOMA: {inst.diploma_principal}

{ctx_rag}

Redige o {inst.termo_decisao.upper()} com EXACTAMENTE 8 secções.
Terceira pessoa. Não escrevas fora das secções.

== 1. RELATÓRIO ==
[Partes, tribunal, objecto — 4-6 frases]

== 2. FACTOS PROVADOS ==
[Lista numerada com fundamento da prova]

== 3. FACTOS NÃO PROVADOS ==
[Com razão da não prova]

== 4. MOTIVAÇÃO DA DECISÃO DE FACTO ==
[Análise crítica das provas, credibilidade, valoração]

== 5. FUNDAMENTAÇÃO JURÍDICA ==
[Subsunção ao {inst.diploma_principal}]
[⚠️ Artigo incerto → [art.?] — NUNCA inventar]

== 6. DISPOSITIVO ==
[OBRIGATÓRIO: "O Tribunal DECIDE:" + CONDENA/ABSOLVE/JULGA]
[Sanção concreta, prazo, montante]

== 7. CUSTAS E TAXA DE JUSTIÇA ==

== 8. NOTA PARA O CIDADÃO ==
[3-4 frases em linguagem acessível]

Máximo 1000 palavras."""

    # ── Consistência e Incerteza ─────────────────────────────────────
    @staticmethod
    def consistencia(
        inst: InstanciaJudicial,
        s_rigorosa: str,
        s_garantista: str,
        s_equilibrada: str,
    ) -> str:
        return f"""És um analista jurídico especialista em {inst.nome}, República Portuguesa.

Três {inst.termo_decisao}s do mesmo caso, por juízes com perfis distintos:

=== SENTENÇA RIGOROSA ===
{s_rigorosa[:800]}

=== SENTENÇA GARANTISTA ===
{s_garantista[:800]}

=== SENTENÇA EQUILIBRADA ===
{s_equilibrada[:800]}

Produz RELATÓRIO DE CONSISTÊNCIA E INCERTEZA:

## CONVERGÊNCIAS
(factos e conclusões em que as 3 sentenças concordam — alta certeza)

## DIVERGÊNCIAS SUBSTANTIVAS
(onde diferem e porquê — revela discricionariedade legítima)

## PONTOS FACTUAIS MAIS FRÁGEIS
(factos questionados em pelo menos 1 sentença)

## ARTIGOS JURÍDICOS CONTESTADOS
(normas interpretadas diferentemente)

## GRAU DE INCERTEZA GLOBAL
(Baixo | Médio | Alto | Muito Alto + justificação de 2-3 linhas)

## RECOMENDAÇÃO AO CIDADÃO
(linguagem simples — o que este grau de incerteza significa na prática)

Rigoroso, neutro, analítico. Máximo 600 palavras."""

    # ── TEDH — Comparação europeia ────────────────────────────────────
    @staticmethod
    def analise_tedh(
        inst: InstanciaJudicial,
        caso_pt: str,
        ctx_tedh: str,
        lingua: str = "pt",
    ) -> str:
        if lingua == "en":
            return f"""You are a European human rights law expert specialising in ECtHR jurisprudence.

Portuguese case summary:
{caso_pt[:600]}

Relevant ECtHR case law:
{ctx_tedh[:1500]}

Analyse this Portuguese case in light of ECtHR jurisprudence:

## APPLICABLE CONVENTION ARTICLES
(ECHR articles potentially engaged)

## RELEVANT ECtHR PRECEDENTS
(key cases and their holdings)

## COMPLIANCE ASSESSMENT
(would the Portuguese proceedings likely comply with ECHR standards?)

## RISK OF STRASBOURG CHALLENGE
(Low | Medium | High | Very High + reasoning)

## RECOMMENDED SAFEGUARDS
(to align with ECtHR standards)

Be precise and cite specific ECtHR cases where possible. Max 600 words."""

        return f"""És um especialista em direito europeu dos direitos humanos e jurisprudência do TEDH.

Resumo do caso português:
{caso_pt[:600]}

Jurisprudência TEDH relevante:
{ctx_tedh[:1500]}

Analisa este caso português à luz da jurisprudência do TEDH:

## ARTIGOS DA CONVENÇÃO APLICÁVEIS
(artigos da CEDH potencialmente em causa)

## PRECEDENTES DO TEDH RELEVANTES
(casos-chave e respectivas decisões)

## AVALIAÇÃO DE CONFORMIDADE
(o processo português cumpriria os padrões CEDH?)

## RISCO DE QUEIXA A ESTRASBURGO
(Baixo | Médio | Alto | Muito Alto + fundamentação)

## SALVAGUARDAS RECOMENDADAS
(para alinhar com os padrões do TEDH)

Sê preciso e cita casos TEDH concretos quando possível. Máximo 600 palavras."""

    # ── Contraditório — feedback ao utilizador ────────────────────────
    @staticmethod
    def contraditorio_feedback(
        inst: InstanciaJudicial,
        argumento: str,
        acusacao: str,
        detetive: str,
    ) -> str:
        return f"""És o Juiz Presidente do {inst.nome}, República Portuguesa.

O advogado de defesa apresentou o seguinte argumento em sede de contraditório:
"{argumento}"

CONTEXTO DO PROCESSO:
Instrução: {detetive[:400]}
Acusação: {acusacao[:400]}

TAREFA: Avalia juridicamente o argumento do advogado de defesa.
Responde como um juiz imparcial que aprecia o argumento:

## ADMISSIBILIDADE DO ARGUMENTO
(admitido | parcialmente admitido | inadmissível + razão)

## FORÇA JURÍDICA
(forte | moderada | fraca + justificação)

## IMPACTO NA INSTRUÇÃO
(como este argumento altera ou não a análise dos factos)

## QUESTÕES DE DIREITO LEVANTADAS
(artigos relevantes do {inst.diploma_principal})

## NOTA AO ADVOGADO
(orientação sobre como desenvolver ou reforçar o argumento)

Linguagem jurídica formal mas clara. Máximo 400 palavras."""

    # ── Extracção de PDF ─────────────────────────────────────────────
    @staticmethod
    def pdf_extraction(conteudo: str, tipo_doc: str) -> str:
        return f"""És um especialista jurídico português.
Documento: {tipo_doc}

Extrai e estrutura as informações relevantes para o processo:

## TIPO DE DOCUMENTO
## PARTES IDENTIFICADAS
## DATAS RELEVANTES
## FACTOS PRINCIPAIS
## VALORES / MONTANTES
## OBSERVAÇÕES PARA O PROCESSO

Documento:
{conteudo[:3000]}

Conciso e preciso. Terminologia jurídica portuguesa."""
