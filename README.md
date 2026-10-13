# Chat Multi-Especialistas · LangGraph + Databricks

Um único chat em que o **Agente 01** conversa com o usuário, entende o contexto, **detecta a intenção**
e aciona o **agente especialista** da regra de negócio correspondente. Cada especialista é um
**agente de BI conversacional**: lê as regras do seu domínio (RAG no Vector Search), explora as
tabelas do domínio no Unity Catalog, escreve e executa SQL num SQL Warehouse, corrige a consulta
se der erro e responde com os números, a interpretação e o SQL usado.

```
                 ┌──────────────────────────────┐
  usuário ─────► │ Agente 01 (agente_principal)  │  entende o histórico, reescreve a pergunta
                 └──────────────┬───────────────┘
                                │
                      ◆ detector de intenção ◆
        ┌───────────────┬───────┴────────┬───────────────┬────────────────┐
        ▼               ▼                ▼               ▼                ▼
 especialista_01  especialista_02  especialista_n+1  esclarecer   conversa_geral / fora_do_escopo
 (agente BI)      (agente BI)      (...)             (pergunta     (saudação, assunto
        └───────────┬───┘                             ambígua)       não coberto)
                    ▼
              consolidar  (une respostas se 2 especialistas foram acionados)

 Dentro de cada especialista (ciclo do agente):
   pergunta ─► pensa ─► ferramenta ─► observa ─► ... ─► resposta com números + SQL
                          ├─ consultar_regras_negocio  (Vector Search do domínio)
                          ├─ descrever_tabela          (colunas e comentários do Unity Catalog)
                          ├─ executar_sql              (SELECT somente leitura no SQL Warehouse)
                          └─ perguntar_genie           (opcional: Genie Space do domínio)
```

## Como começar: app Streamlit no Databricks

A configuração é feita numa **app Streamlit** (Databricks Apps) com 5 abas:

| Aba | O que acontece |
|---|---|
| **1 · Modelos e Genies** | Lista **todas as IAs** do workspace (endpoints de chat e de embeddings), os **Genie Spaces** e os **SQL Warehouses** que você pode usar. Você escolhe a IA do Agente 01, a dos especialistas, os Genies liberados, o warehouse e o preço dos modelos (LLMOps). |
| **2 · Agentes** | Para cada agente: nome, **prompt de negócio**, **tabelas** escolhidas navegando por catálogo → schema → tabela, Genie opcional e documentação opcional. Um botão cria um agente por Genie. |
| **3 · Estudar tabelas** | O sistema **estuda** as tabelas de cada agente no SQL Warehouse (esquema, comentários, nulos, distintos, mínimo/máximo, valores mais comuns, amostra) e, com o seu prompt, escreve o conhecimento do agente. Cada SQL de exemplo é **executado**; só os que rodam entram. Você revisa e edita tudo. |
| **4 · Agente geral** | Gera o `agents.yaml` (no Volume do projeto) e dispara um **job** (`notebooks/04_publicar_job`) que indexa documentos, avalia o roteamento, registra no Unity Catalog e publica o endpoint + Review App. A aba mostra o progresso, a acurácia do roteamento e o endpoint. |
| **5 · Conversar** | Testa o agente publicado: resposta, agentes acionados, SQL, tabela de resultado, tokens e US$. |

### Criar a app a partir do Git folder

1. **Compute → Apps → Create app → Create a custom app**. Dê um nome (ex.: `chat-especialistas`) e crie.
2. Na página da app, em **Configure → User authorization**, adicione os escopos:
   `sql`, `model-serving`, `genie`, `files`, `catalog.catalogs:read`, `catalog.schemas:read`, `catalog.tables:read`.
   Assim a app age **como quem está logado** e mostra os modelos, tabelas e Genies que essa pessoa enxerga.
   (Se a opção não aparecer, um admin precisa habilitar *user authorization* para Apps no workspace.)
3. Clique em **Deploy** e escolha como **source code path a pasta raiz do Git folder** (a que contém `app.py`
   e `app.yaml`, ex.: `/Workspace/Users/voce@empresa.com/chat-especialistas`). Não escolha a pasta `notebooks`.
4. Aguarde o status **Running** e abra a URL da app.
5. Depois de um `git pull` no Git folder, clique em **Deploy** de novo para a app pegar o código novo.

A app instala o `requirements.txt` da raiz (leve: Streamlit + SDK). O agente usa `requirements-agente.txt`.

**Conta de serviço da app** (aba *Authorization* da app): só é usada se a *user authorization* estiver
desligada, ou como plano B para disparar o job de publicação. Nesse caso, dê a ela `SELECT` nas tabelas,
`CAN USE` no warehouse, `CAN QUERY` nos endpoints e acesso aos Genies.

**O job de publicação** roda em computação serverless. Se o workspace não tiver serverless para jobs,
defina `CLUSTER_ID_PUBLICACAO` no `app.yaml`. Se a app não descobrir a pasta do repositório sozinha,
defina `CAMINHO_REPO`.

Prefere notebook? `notebooks/00_COMECE_AQUI` tem o mesmo assistente com ipywidgets.
Sem dados ainda? Rode `notebooks/exemplo_dados_ficticios` para criar tabelas fictícias de crédito e cobrança.

## Estrutura

```
app.py · app.yaml · requirements.txt  # ⭐ app Streamlit (Databricks Apps): configurador + chat de teste
notebooks/00_COMECE_AQUI.py           # mesmo assistente em notebook (ipywidgets)
notebooks/04_publicar_job.py          # job disparado pela app: indexa, avalia, registra e publica
notebooks/01_ingestao_documentos.py   # reindexar documentos quando mudarem
notebooks/02_registrar_e_publicar.py  # conversar com o agente, avaliar e republicar
notebooks/03_llmops_painel.py         # tabela Delta de LLMOps, views, alerta de orçamento, job
notebooks/exemplo_dados_ficticios.py  # (opcional) tabelas fictícias para testar
config/agents.yaml                    # exemplo; a app grava a configuração real no Volume do projeto
requirements-agente.txt               # dependências do agente (notebooks e endpoint)
src/chat_especialistas/
    configurador/descoberta.py        # lista IAs, Genies, warehouses, catálogos/schemas/tabelas
    configurador/estudo.py            # perfil das tabelas + conhecimento gerado + validação dos SQLs
    configurador/gerador.py           # escolhas do assistente → agents.yaml
    configurador/interface.py         # versão notebook (ipywidgets)
    configurador/llm_sdk.py           # LLM via SDK com a identidade de quem usa a app
    app_servicos.py                   # tudo o que a app faz no Databricks (descoberta, Volume, job, chat)
    graph.py                          # grafo LangGraph: Agente 01 → detector de intenção → especialistas
    bi_agent.py                       # especialista como agente de BI (pensar → ferramenta → observar)
    dados.py                          # Unity Catalog, SQL Warehouse e Genie em produção
    sql_guard.py                      # SQL somente leitura e só nas tabelas do agente
    retriever.py / ingestao.py        # documentação no Vector Search
    llmops.py                         # tokens, custo em US$ e registro de cada interação
    publicacao.py                     # avaliação do roteamento, registro no MLflow/UC e deploy
agent.py                              # wrapper MLflow ResponsesAgent (o que vai para o Model Serving)
docs_exemplo/                         # documentos fictícios para testar a documentação
tests/                                # testes locais sem Databricks (LLM simulado, SQL real em SQLite)
```

## Permissões necessárias

- **Quem configura** (usa a app ou o notebook): `SELECT` nas tabelas escolhidas, `CAN USE` no warehouse e
  nos endpoints de modelo, acesso aos Genie Spaces, e `CREATE SCHEMA/TABLE/VOLUME/MODEL` no
  catálogo/schema do projeto.
- **O endpoint publicado** recebe automaticamente credenciais para os recursos declarados (LLMs,
  warehouse, tabelas, Genies, índice). Com *"Permissão de cada usuário"* na aba 1, as consultas
  usam a permissão de quem pergunta (on-behalf-of-user, precisa estar habilitado no workspace).

O endpoint segue o formato OpenAI Responses e funciona no AI Playground, em apps Databricks
e via REST. A resposta traz em `custom_outputs` a intenção detectada, o(s) especialista(s)
acionado(s), a confiança, as fontes usadas, **o SQL executado** (`consultas_sql`) e o
**último resultado de cada especialista** (`dados`: colunas e linhas, pronto para virar tabela
ou gráfico num front-end).

## Agentes de BI conversacional

O assistente preenche isto automaticamente; o formato no `agents.yaml`, por especialista, é:

```yaml
dados:
  tabelas: [main.credito.propostas, main.credito.clientes]   # só estas podem ser consultadas
  instrucoes: |                                              # definições de métricas, joins, armadilhas
    - "Taxa de aprovação" = APROVADA / total
  exemplos_sql:                                              # consultas validadas (few-shot)
    - pergunta: Taxa de aprovação por mês
      sql: SELECT ...
genie_space_id: "01ef..."                                    # opcional
```

- **Segurança em duas camadas:** o `sql_guard` só deixa passar um único `SELECT`/`WITH`, com nomes
  completos e apenas nas tabelas do domínio, e aplica `LIMIT`. A camada principal continua sendo o
  Unity Catalog: dê `SELECT` só nas tabelas certas. Com `bi.autenticacao: usuario` (on-behalf-of-user),
  cada pessoa só enxerga o que tem permissão — inclusive filtros de linha e máscaras de coluna.
- **Qualidade do SQL:** comentários de tabela/coluna no Unity Catalog, `instrucoes` e `exemplos_sql`
  são o que mais melhora as respostas. Comece pelas perguntas que o negócio mais faz.
- **Sem `dados`** o especialista responde só pela documentação (uma chamada ao LLM).
- **Genie:** se o domínio já tem um Genie Space bem curado, informe `genie_space_id`; o agente pode
  delegar a consulta a ele e conferir o SQL gerado.
- **Limites:** `bi.max_iteracoes` (passos do agente), `bi.max_linhas` (LIMIT), `bi.linhas_para_o_llm`
  (quanto do resultado o agente lê) e `bi.timeout_segundos`.
- O modelo de LLM precisa suportar **tool calling** (o Llama 3.3 70B do Databricks suporta).

## Onde o MLflow é usado

| Parte | Uso do MLflow |
|---|---|
| `agent.py` | O chat é um `mlflow.pyfunc.ResponsesAgent` (models-from-code) |
| Rastreamento | `mlflow.langchain.autolog()` + `@mlflow.trace`: cada pergunta vira um trace com o Agente 01, cada especialista, cada chamada de LLM e **cada ferramenta (SQL, Genie, regras)** como spans |
| LLMOps | Tags `llmops.*` (tokens, US$) e o span `llmops_registro` em cada trace; o notebook 03 lê daqui |
| Notebook 02 | `log_model` com recursos (LLMs, índice, warehouse, tabelas, Genie), `register_model` no Unity Catalog e `agents.deploy` (Model Serving + Review App) |
| Feedback | A Review App grava as avaliações dos usuários de negócio junto aos traces |

## Adicionar o "especialista n+1"

No `00_COMECE_AQUI`: aba 2 → **Novo agente** (nome, prompt, tabelas) → aba 3 → **Estudar agentes
pendentes** → aba 4 → **Gerar agente geral**. Nenhuma linha de código muda.

A **descrição e os exemplos** gerados no estudo são o que o detector de intenção usa para decidir;
revise-os na aba 3 se dois agentes tiverem assuntos parecidos.

## Comportamentos importantes

| Situação | O que acontece |
|---|---|
| Pergunta clara de uma regra | 1 especialista responde com citações `[fonte]` |
| Pergunta que envolve 2 regras | 2 especialistas rodam em paralelo e o consolidador une |
| Confiança < `limiar_confianca` ou ambígua | Agente 01 pede esclarecimento |
| "Oi", "obrigado", "o que você faz?" | Agente 01 responde direto |
| Assunto não coberto | Informa os assuntos atendidos |
| Documentação não traz a resposta | Especialista diz que não encontrou (não inventa) |
| "E para PJ?" depois de outra pergunta | Agente 01 reescreve com o contexto antes de buscar |
| Pergunta analítica ("taxa de aprovação PJ por mês") | Agente de BI descreve as tabelas, roda SQL e responde com números, tabela e o SQL |
| SQL com erro | O agente lê o erro, corrige e roda de novo |
| SQL que altera dados ou lê tabela de outro domínio | Bloqueado antes de chegar ao warehouse |
| Análise que não fecha em `max_iteracoes` passos | Responde com o que apurou e diz o que ficou pendente |

## LLMOps: perguntas, respostas, tokens e US$

Cada interação gera **um registro** com: pergunta, resposta, pergunta reescrita, intenção,
especialistas acionados, confiança, fontes, usuário/conversa, latência, erro e, para **cada
chamada de LLM** (Agente 01, cada especialista, consolidador, embedding da busca), o endpoint,
tokens de entrada/saída e **custo em US$**.

```
agent.py ──► RastreadorUso (callback LangChain, por requisição)
         └─► registro ──┬─► custom_outputs.llmops ──► inference table (Delta, automático)
                        ├─► trace do MLflow (tags llmops.* + span llmops_registro)
                        ├─► log do Model Serving (linhas "LLMOPS {...}")
                        └─► JSONL num Volume (testes no notebook)
                                    │
notebooks/03_llmops_painel ─────────┘──► tabela Delta llmops_interacoes + views vw_llmops_*
                                         ──► AI/BI Dashboard + SQL Alert de orçamento
```

- **Preços** ficam em `llmops.precos_usd_por_milhao` no `agents.yaml` (US$ por 1M tokens, por endpoint).
  Os valores vêm de tabela pública; confirme com o seu contrato Databricks. Endpoint sem preço usa
  `preco_padrao_usd_por_milhao`.
- **Tokens** vêm do `usage` devolvido pelo endpoint. Se algum endpoint não devolver, o valor é
  estimado (~4 caracteres/token) e o registro sai com `tokens_estimados = true`. O embedding da
  busca vetorial é sempre estimado.
- **Agentes de BI:** cada registro traz também as consultas executadas (`consultas_sql`, com SQL,
  linhas, tempo e erro). O SQL Warehouse é cobrado em DBU, separado dos tokens: o painel mostra
  quantidade e tempo das consultas, e o custo do warehouse fica em `system.billing.usage`.
- **LGPD:** `registrar_textos: false` deixa pergunta/resposta fora do registro (tokens e custo continuam).
- **Usuário/conversa:** envie `custom_inputs: {"user_id": ..., "conversation_id": ...}` (ou `context`)
  na chamada ao endpoint para ter custo por usuário.
- Rode o `03_llmops_painel` como **Job de hora em hora** (o próprio notebook cria o job com o widget `criar_job = sim`).
- O custo calculado é uma **estimativa por preço de tabela**. Para conciliar com a fatura real, compare
  com `system.billing.usage` (filtre por `usage_metadata.endpoint_name`).

## Rodar os testes localmente

```bash
pip install -r requirements-agente.txt -r requirements.txt "mlflow-skinny>=3.1" sqlalchemy alembic
pytest -q
```

## Por que estudar e não "treinar"?

O "treino" de cada especialista é o **estudo**: o conhecimento (significado das tabelas, métricas,
SQLs validados) fica no `agents.yaml` e a documentação fica indexada no Vector Search. Mudou uma
regra ou uma tabela? É só estudar de novo, sem retreinar modelo. Fine-tuning só vale a pena depois,
se o estilo das respostas precisar de ajuste.
