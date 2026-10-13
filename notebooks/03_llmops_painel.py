# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · LLMOps — perguntas, respostas, tokens e US$ gastos
# MAGIC
# MAGIC Consolida **um registro por interação** do chat numa tabela Delta e cria as views para o painel.
# MAGIC
# MAGIC | Fonte | Quando aparece |
# MAGIC |---|---|
# MAGIC | **Traces do MLflow** (span `llmops_registro`) | toda chamada ao endpoint publicado e aos testes do notebook 02 — inclusive Playground e Review App (streaming) |
# MAGIC | **JSONL no Volume** | testes do notebook 02 (destino `jsonl`) |
# MAGIC | **Inference table** do endpoint | conferência: `custom_outputs.llmops` de cada resposta |
# MAGIC
# MAGIC Agende este notebook como **Job a cada hora** (última célula faz isso). Ele é incremental e idempotente
# MAGIC (MERGE por `request_id`), pode rodar quantas vezes quiser.
# MAGIC
# MAGIC Tabelas/views criadas em `<catalogo>.<schema>`:
# MAGIC `llmops_interacoes`, `llmops_chamadas`, `vw_llmops_custo_diario`, `vw_llmops_custo_por_etapa`,
# MAGIC `vw_llmops_custo_por_usuario`, `vw_llmops_custo_por_modelo`, `vw_llmops_top_perguntas_caras`,
# MAGIC `vw_llmops_consultas_sql` (SQL gerado pelos agentes de BI: erros, bloqueios, tempo).

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-agente.txt
# MAGIC %restart_python

# COMMAND ----------

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import mlflow
from pyspark.sql import functions as F

sys.path.insert(0, os.path.abspath("../src"))
from chat_especialistas import carregar_config

cfg = carregar_config("../config/agents.yaml")
catalogo, schema = cfg.projeto.catalogo, cfg.projeto.schema_
usuario = spark.sql("SELECT current_user()").first()[0]
modelo_uc = cfg.projeto.nome_modelo

dbutils.widgets.text("experimento", f"/Users/{usuario}/chat_especialistas", "Experimento MLflow (traces)")
dbutils.widgets.text("arquivo_jsonl", f"/Volumes/{catalogo}/{schema}/llmops/interacoes_dev.jsonl", "JSONL dev")
dbutils.widgets.text("inference_table", f"{modelo_uc}_payload", "Inference table do endpoint")
dbutils.widgets.text("orcamento_diario_usd", "10", "Orçamento diário (US$)")
dbutils.widgets.dropdown("criar_job", "nao", ["nao", "sim"], "Criar job de hora em hora?")

TABELA = f"{catalogo}.{schema}.llmops_interacoes"
EXPERIMENTO = dbutils.widgets.get("experimento")
ARQUIVO_JSONL = dbutils.widgets.get("arquivo_jsonl")
INFERENCE_TABLE = dbutils.widgets.get("inference_table")
ORCAMENTO = float(dbutils.widgets.get("orcamento_diario_usd"))

# COMMAND ----------

# MAGIC %md ## 1. Tabela Delta de interações

# COMMAND ----------

SCHEMA_REGISTRO = """
    request_id STRING, conversation_id STRING, user_id STRING, `timestamp` STRING,
    versao_agente STRING, pergunta STRING, resposta STRING, pergunta_reescrita STRING,
    intencao STRING, especialistas ARRAY<STRING>, confianca DOUBLE, fontes ARRAY<STRING>,
    qtd_chamadas_llm INT, tokens_input BIGINT, tokens_output BIGINT, tokens_total BIGINT,
    custo_usd DOUBLE, custo_por_etapa MAP<STRING, STRUCT<tokens: BIGINT, custo_usd: DOUBLE>>,
    tokens_estimados BOOLEAN, latencia_ms BIGINT, erro STRING,
    chamadas ARRAY<STRUCT<etapa: STRING, modelo: STRING, tokens_input: BIGINT, tokens_output: BIGINT,
                          custo_usd: DOUBLE, latencia_ms: BIGINT, estimado: BOOLEAN, tipo: STRING>>,
    qtd_consultas_sql INT, tempo_sql_ms BIGINT,
    consultas_sql ARRAY<STRUCT<especialista: STRING, via: STRING, objetivo: STRING, sql: STRING,
                               linhas: BIGINT, duracao_ms: BIGINT, erro: STRING>>,
    trace_id STRING, origem STRING
"""

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {TABELA} (
    request_id STRING NOT NULL, conversation_id STRING, user_id STRING, ts TIMESTAMP, data DATE,
    versao_agente STRING, pergunta STRING, resposta STRING, pergunta_reescrita STRING,
    intencao STRING, especialistas ARRAY<STRING>, confianca DOUBLE, fontes ARRAY<STRING>,
    qtd_chamadas_llm INT, tokens_input BIGINT, tokens_output BIGINT, tokens_total BIGINT,
    custo_usd DOUBLE, custo_por_etapa MAP<STRING, STRUCT<tokens: BIGINT, custo_usd: DOUBLE>>,
    tokens_estimados BOOLEAN, latencia_ms BIGINT, erro STRING,
    chamadas ARRAY<STRUCT<etapa: STRING, modelo: STRING, tokens_input: BIGINT, tokens_output: BIGINT,
                          custo_usd: DOUBLE, latencia_ms: BIGINT, estimado: BOOLEAN, tipo: STRING>>,
    qtd_consultas_sql INT, tempo_sql_ms BIGINT,
    consultas_sql ARRAY<STRUCT<especialista: STRING, via: STRING, objetivo: STRING, sql: STRING,
                               linhas: BIGINT, duracao_ms: BIGINT, erro: STRING>>,
    trace_id STRING, origem STRING, ingerido_em TIMESTAMP
)
CLUSTER BY (data)
COMMENT 'LLMOps do chat multi-especialistas: 1 linha por interação (pergunta, resposta, tokens, US$)'
""")

# COMMAND ----------

# MAGIC %md ## 2. Coletar registros novos (traces + JSONL)

# COMMAND ----------

registros = []

# --- 2a. Traces do MLflow (endpoint publicado + testes)
ultimo = spark.sql(f"SELECT max(ts) FROM {TABELA} WHERE origem = 'trace'").first()[0]
desde = (ultimo.replace(tzinfo=timezone.utc) - timedelta(hours=1)) if ultimo else datetime(2020, 1, 1, tzinfo=timezone.utc)
exp = mlflow.get_experiment_by_name(EXPERIMENTO)

if exp is None:
    print(f"⚠️  Experimento {EXPERIMENTO} não encontrado — rode o notebook 02 primeiro.")
else:
    traces = mlflow.search_traces(
        experiment_ids=[exp.experiment_id],
        filter_string=f"trace.timestamp_ms > {int(desde.timestamp() * 1000)}",
        max_results=50_000,
        return_type="list",
    )
    for t in traces:
        for span in t.search_spans(name="llmops_registro"):
            if isinstance(span.outputs, dict):
                registros.append({**span.outputs, "trace_id": t.info.trace_id, "origem": "trace"})
    print(f"Traces lidos desde {desde:%Y-%m-%d %H:%M} UTC: {len(traces)} → registros: {len(registros)}")

# --- 2b. JSONL de desenvolvimento
if os.path.exists(ARQUIVO_JSONL):
    with open(ARQUIVO_JSONL, encoding="utf-8") as f:
        dev = [json.loads(l) for l in f if l.strip()]
    registros += [{**r, "trace_id": None, "origem": "jsonl_dev"} for r in dev]
    print(f"Registros do JSONL: {len(dev)}")

# COMMAND ----------

if registros:
    df = (
        spark.createDataFrame([(json.dumps(r, default=str),) for r in registros], "j STRING")
        .select(F.from_json("j", SCHEMA_REGISTRO).alias("r"))
        .select("r.*")
        .withColumn("ts", F.to_timestamp("timestamp"))
        .withColumn("data", F.to_date("ts"))
        .withColumn("ingerido_em", F.current_timestamp())
        .drop("timestamp")
        .dropDuplicates(["request_id"])
    )
    df.createOrReplaceTempView("novos_registros")
    resultado = spark.sql(f"""
        MERGE WITH SCHEMA EVOLUTION INTO {TABELA} t
        USING novos_registros s ON t.request_id = s.request_id
        WHEN NOT MATCHED THEN INSERT *
    """)
    display(resultado)
else:
    print("Nada novo para ingerir.")

# COMMAND ----------

# MAGIC %md ## 3. Views para o painel

# COMMAND ----------

base = f"{catalogo}.{schema}"
spark.sql(f"""
CREATE OR REPLACE VIEW {base}.llmops_chamadas AS
SELECT i.request_id, i.ts, i.data, i.user_id, i.conversation_id, i.versao_agente,
       c.etapa,
       CASE WHEN c.etapa = 'agente_principal' THEN 'Agente 01 (detector de intenção)'
            WHEN c.etapa LIKE 'especialista_%' THEN 'Especialista ' || substr(c.etapa, 14)
            WHEN c.etapa LIKE 'busca_%' THEN 'Busca vetorial ' || substr(c.etapa, 7)
            ELSE initcap(replace(c.etapa, '_', ' ')) END AS etapa_descricao,
       c.modelo, c.tipo, c.tokens_input, c.tokens_output,
       c.tokens_input + c.tokens_output AS tokens_total, c.custo_usd, c.latencia_ms, c.estimado
FROM {TABELA} i LATERAL VIEW explode(i.chamadas) AS c
""")

spark.sql(f"""
CREATE OR REPLACE VIEW {base}.vw_llmops_custo_diario AS
SELECT data,
       count(*)                                   AS interacoes,
       count(DISTINCT user_id)                    AS usuarios,
       count(DISTINCT conversation_id)            AS conversas,
       sum(tokens_input)                          AS tokens_input,
       sum(tokens_output)                         AS tokens_output,
       sum(tokens_total)                          AS tokens_total,
       round(sum(custo_usd), 6)                   AS custo_usd,
       round(avg(custo_usd), 6)                   AS custo_medio_por_pergunta_usd,
       percentile(latencia_ms, 0.95)              AS latencia_p95_ms,
       sum(CASE WHEN erro IS NOT NULL THEN 1 ELSE 0 END) AS erros,
       round(avg(CASE WHEN intencao = 'esclarecer' THEN 1 ELSE 0 END), 4) AS taxa_esclarecimento,
       sum(qtd_consultas_sql)                     AS consultas_sql,
       round(avg(tempo_sql_ms), 0)                AS tempo_sql_medio_ms
FROM {TABELA}
GROUP BY data
""")

spark.sql(f"""
CREATE OR REPLACE VIEW {base}.vw_llmops_custo_por_etapa AS
SELECT data, etapa_descricao, count(*) AS chamadas, sum(tokens_total) AS tokens,
       round(sum(custo_usd), 6) AS custo_usd, avg(latencia_ms) AS latencia_media_ms
FROM {base}.llmops_chamadas GROUP BY data, etapa_descricao
""")

spark.sql(f"""
CREATE OR REPLACE VIEW {base}.vw_llmops_custo_por_usuario AS
SELECT coalesce(user_id, '(sem usuário)') AS user_id, count(*) AS interacoes,
       sum(tokens_total) AS tokens, round(sum(custo_usd), 6) AS custo_usd,
       min(ts) AS primeira_interacao, max(ts) AS ultima_interacao
FROM {TABELA} GROUP BY 1
""")

spark.sql(f"""
CREATE OR REPLACE VIEW {base}.vw_llmops_custo_por_modelo AS
SELECT data, modelo, tipo, sum(tokens_input) AS tokens_input, sum(tokens_output) AS tokens_output,
       round(sum(custo_usd), 6) AS custo_usd, max(CAST(estimado AS INT)) = 1 AS tem_estimativa
FROM {base}.llmops_chamadas GROUP BY data, modelo, tipo
""")

spark.sql(f"""
CREATE OR REPLACE VIEW {base}.vw_llmops_consultas_sql AS
SELECT i.request_id, i.ts, i.data, i.user_id, i.pergunta,
       q.especialista, q.via, q.objetivo, q.sql, q.linhas, q.duracao_ms, q.erro,
       q.erro LIKE 'bloqueado%' AS bloqueada
FROM {TABELA} i LATERAL VIEW explode(i.consultas_sql) AS q
""")

spark.sql(f"""
CREATE OR REPLACE VIEW {base}.vw_llmops_top_perguntas_caras AS
SELECT ts, user_id, pergunta, intencao, especialistas, tokens_total, custo_usd, latencia_ms, resposta
FROM {TABELA} ORDER BY custo_usd DESC LIMIT 200
""")
print("Views criadas.")

# COMMAND ----------

# MAGIC %md ## 4. Visão rápida

# COMMAND ----------

display(spark.sql(f"""
SELECT count(*) AS interacoes, sum(tokens_total) AS tokens,
       round(sum(custo_usd), 4) AS custo_total_usd,
       round(avg(custo_usd), 6) AS custo_medio_usd,
       round(sum(custo_usd) / nullif(count(DISTINCT data), 0), 4) AS custo_medio_dia_usd
FROM {TABELA}
"""))

# COMMAND ----------

display(spark.table(f"{base}.vw_llmops_custo_diario").orderBy(F.desc("data")))

# COMMAND ----------

display(spark.sql(f"""
SELECT etapa_descricao, sum(chamadas) AS chamadas, sum(tokens) AS tokens, round(sum(custo_usd), 4) AS custo_usd
FROM {base}.vw_llmops_custo_por_etapa GROUP BY 1 ORDER BY custo_usd DESC
"""))

# COMMAND ----------

display(spark.sql(f"""
SELECT ts, user_id, pergunta, especialistas, tokens_input, tokens_output, custo_usd, latencia_ms
FROM {TABELA} ORDER BY ts DESC LIMIT 100
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Conferência com a inference table do endpoint
# MAGIC O Model Serving grava request/response de cada chamada numa tabela Delta (AI Gateway).
# MAGIC Confira o nome em **Serving → endpoint → AI Gateway → Inference tables** e ajuste o widget.
# MAGIC Em respostas sem streaming, `custom_outputs.llmops` já vem pronto na coluna `response`.

# COMMAND ----------

if spark.catalog.tableExists(INFERENCE_TABLE):
    spark.sql(f"""
    CREATE OR REPLACE VIEW {base}.vw_llmops_inference_table AS
    SELECT request_time, databricks_request_id, requester, status_code, execution_duration_ms,
           response:custom_outputs.llmops.request_id::string      AS request_id,
           response:custom_outputs.llmops.intencao::string        AS intencao,
           response:custom_outputs.llmops.tokens_input::bigint    AS tokens_input,
           response:custom_outputs.llmops.tokens_output::bigint   AS tokens_output,
           response:custom_outputs.llmops.custo_usd::double       AS custo_usd,
           response:output[0].content[0].text::string             AS resposta,
           request                                                AS request_bruto
    FROM {INFERENCE_TABLE}
    """)
    display(spark.sql(f"""
        SELECT date(request_time) AS data, count(*) AS chamadas, sum(custo_usd) AS custo_usd,
               count(*) - count(custo_usd) AS sem_custom_outputs_streaming
        FROM {base}.vw_llmops_inference_table GROUP BY 1 ORDER BY 1 DESC
    """))
else:
    print(f"Inference table {INFERENCE_TABLE} ainda não existe (ela aparece após as primeiras chamadas ao endpoint).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Alerta de orçamento
# MAGIC Crie um **SQL Alert** (SQL → Alerts) com a consulta abaixo, condição `estourou = true`,
# MAGIC verificando a cada hora e notificando por e-mail/Slack.

# COMMAND ----------

consulta_alerta = f"""
SELECT round(coalesce(sum(custo_usd), 0), 4) AS custo_hoje_usd,
       {ORCAMENTO} AS orcamento_usd,
       coalesce(sum(custo_usd), 0) > {ORCAMENTO} AS estourou
FROM {TABELA}
WHERE data = current_date()
"""
print(consulta_alerta)
display(spark.sql(consulta_alerta))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Painel (AI/BI Dashboard)
# MAGIC Em **Dashboards → Create dashboard**, adicione como datasets as views `vw_llmops_*` e monte:
# MAGIC - **Contadores:** custo hoje, custo no mês, interações, custo médio por pergunta (`vw_llmops_custo_diario`)
# MAGIC - **Linha:** `custo_usd` e `interacoes` por `data`
# MAGIC - **Barras empilhadas:** `custo_usd` por `data` e `etapa_descricao` (`vw_llmops_custo_por_etapa`)
# MAGIC - **Tabela:** `vw_llmops_top_perguntas_caras` e `vw_llmops_custo_por_usuario`
# MAGIC - **Linha:** `latencia_p95_ms` e `taxa_esclarecimento` (saúde do detector de intenção)
# MAGIC - **Agentes de BI:** % de consultas com erro, consultas bloqueadas e tempo médio por especialista
# MAGIC   (`vw_llmops_consultas_sql`). O custo do SQL Warehouse é cobrado em DBU, à parte dos tokens:
# MAGIC   veja `system.billing.usage` filtrando pelo warehouse.

# COMMAND ----------

# MAGIC %md ## 8. (Opcional) Agendar este notebook a cada hora

# COMMAND ----------

if dbutils.widgets.get("criar_job") == "sim":
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service import jobs

    w = WorkspaceClient()
    caminho = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    job = w.jobs.create(
        name="chat_especialistas_llmops",
        tasks=[
            jobs.Task(
                task_key="llmops",
                notebook_task=jobs.NotebookTask(
                    notebook_path=caminho,
                    base_parameters={"experimento": EXPERIMENTO, "criar_job": "nao"},
                ),
            )
        ],
        schedule=jobs.CronSchedule(quartz_cron_expression="0 5 * * * ?", timezone_id="America/Manaus"),
    )
    print("Job criado:", job.job_id)
