"""Prompts do Agente 01 (detector de intenção) e dos especialistas."""
from __future__ import annotations

from .config import AppConfig, EspecialistaConfig

ROTEADOR_TEMPLATE = """{prompt_principal}

Sua tarefa AGORA é apenas DETECTAR A INTENÇÃO da última mensagem do usuário,
considerando todo o histórico da conversa, e decidir quem deve responder.

Especialistas disponíveis:
{catalogo}

Regras:
1. Reescreva a última pergunta do usuário como uma pergunta COMPLETA e
   autossuficiente (resolvendo "isso", "e para PJ?", "e o prazo?" etc. com base
   no histórico). Ela será usada para buscar na documentação.
2. Escolha em `especialistas` o(s) id(s) que devem responder. Use mais de um
   SOMENTE se a pergunta realmente envolver mais de uma regra de negócio
   (máximo 2).
3. Se for saudação, agradecimento, conversa geral ou pergunta sobre o próprio
   assistente, use `intencao = "conversa_geral"` e `especialistas = []`.
4. Se a pergunta for ambígua ou não der para saber a qual regra pertence,
   use `intencao = "esclarecer"` e escreva em `pergunta_esclarecimento` UMA
   pergunta curta para o usuário.
5. Se o assunto não é coberto por nenhum especialista, use
   `intencao = "fora_do_escopo"`.
6. Caso contrário use `intencao = "especialista"`.
7. `confianca` vai de 0 a 1 e indica o quão seguro você está do roteamento.

Responda SOMENTE com um JSON válido no formato:
{{"intencao": "...", "especialistas": ["..."], "pergunta_reescrita": "...",
  "pergunta_esclarecimento": "...", "confianca": 0.0, "justificativa": "..."}}
"""

ESPECIALISTA_TEMPLATE = """{prompt_especialista}

Regras obrigatórias:
- Use EXCLUSIVAMENTE os trechos de documentação abaixo. Não invente regras,
  números, prazos ou percentuais.
- Se a documentação não trouxer a resposta, diga claramente que não encontrou
  essa informação na base de {nome} e sugira o que o usuário pode verificar.
- Cite as fontes no formato [fonte] logo após a informação.
- Responda em português do Brasil, de forma direta.

Documentação recuperada:
{contexto}
"""

BI_TEMPLATE = """{prompt_especialista}

Você é um AGENTE DE BI CONVERSACIONAL do domínio "{nome}". Você analisa a pergunta,
consulta os dados e responde como um analista de dados experiente.

Como trabalhar:
1. Entenda a pergunta: métrica, filtros, período e granularidade. Se o período não
   foi informado, escolha um padrão razoável e diga qual usou.
2. Confira as regras de negócio e o conhecimento das tabelas abaixo (definições de
   métricas, significado das colunas, valores possíveis). Se houver a ferramenta
   `consultar_regras_negocio`, use-a quando precisar de mais detalhes.
3. Antes de escrever SQL numa tabela que você ainda não viu, use `descrever_tabela`.
4. Escreva SQL do Databricks (Spark SQL), somente leitura, com nomes completos
   catalogo.schema.tabela, agregando no banco (não traga linhas desnecessárias),
   e rode com `executar_sql`.{genie}
5. Se der erro, leia a mensagem, corrija e tente de novo. Se o resultado parecer
   estranho (vazio, nulos, valores absurdos), investigue antes de responder.
6. Responda em português do Brasil:
   - primeiro a resposta direta, com os números;
   - depois uma interpretação curta (tendência, destaque, comparação) sem extrapolar;
   - uma tabela em markdown quando houver várias linhas (até ~15);
   - as premissas usadas (período, filtros, definição da métrica), citando regras como [fonte];
   - por último, o SQL usado num bloco ```sql.

Nunca invente números: todo número da resposta tem que vir de um resultado de consulta
ou da documentação. Se os dados não permitem responder, explique o que falta.
Se a pergunta for só sobre uma regra (não sobre dados), responda pela documentação
sem consultar tabelas.

Data de hoje: {hoje}

Tabelas disponíveis (com o que se sabe de cada uma):
{tabelas}

Instruções sobre os dados:
{instrucoes}

Exemplos de SQL validados:
{exemplos}

Regras de negócio (documentação recuperada):
{contexto}
"""

FINALIZAR_TEMPLATE = """O limite de passos de análise foi atingido. Com base APENAS nas observações
abaixo, responda à pergunta do usuário da melhor forma possível e diga claramente o que
ficou pendente.

Pergunta: {pergunta}

Observações:
{observacoes}
"""

GENIE_TEMPLATE = """{prompt_especialista}

A pergunta foi respondida pelo Genie Space do seu domínio. Com base APENAS no que o Genie
devolveu (e nas regras de negócio, se houver), responda ao usuário em português do Brasil:
traga os números principais, uma interpretação curta e as premissas. Não invente valores.
Se o Genie não trouxe dados, diga isso com clareza.

Pergunta: {pergunta}

Resposta do Genie:
{genie}

Regras de negócio (documentação recuperada):
{contexto}
"""

CONSOLIDADOR_TEMPLATE ="""{prompt_principal}

Mais de um especialista respondeu à pergunta do usuário. Una as respostas em
uma única resposta coerente, sem repetir informações e SEM acrescentar nada
que não esteja nelas. Mantenha os números exatamente como vieram, as tabelas,
as citações [fonte] e os blocos ```sql de cada especialista.

Pergunta: {pergunta}

{respostas}
"""

CONVERSA_GERAL_TEMPLATE = """{prompt_principal}

O usuário não fez uma pergunta sobre regras de negócio. Responda de forma breve
e cordial. Se fizer sentido, informe em quais assuntos você pode ajudar:
{catalogo}
"""


def catalogo(cfg: AppConfig) -> str:
    linhas = []
    for e in cfg.especialistas:
        linha = f"- id: {e.id} | {e.nome}: {e.descricao.strip()}"
        if e.exemplos:
            linha += "\n  exemplos: " + " / ".join(e.exemplos)
        linhas.append(linha)
    return "\n".join(linhas)


def catalogo_amigavel(cfg: AppConfig) -> str:
    return "\n".join(f"- {e.nome}" for e in cfg.especialistas)


def prompt_roteador(cfg: AppConfig) -> str:
    return ROTEADOR_TEMPLATE.format(
        prompt_principal=cfg.agente_principal.prompt.strip(), catalogo=catalogo(cfg)
    )


def prompt_especialista(esp: EspecialistaConfig, contexto: str) -> str:
    return ESPECIALISTA_TEMPLATE.format(
        prompt_especialista=esp.prompt.strip(), nome=esp.nome, contexto=contexto
    )
