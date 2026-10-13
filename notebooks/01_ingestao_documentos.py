# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingestão da documentação dos agentes
# MAGIC
# MAGIC > O assistente **00_COMECE_AQUI** já roda esta etapa. Use este notebook para reindexar quando
# MAGIC > os documentos mudarem (ou agende-o como Job).
# MAGIC
# MAGIC Lê `/Volumes/<catalogo>/<schema>/documentos/<id_do_agente>/` (PDF, DOCX, MD, TXT), quebra em trechos,
# MAGIC grava numa tabela Delta e sincroniza o índice do **Vector Search**. Só agentes com `documentos: true`.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
from chat_especialistas import carregar_config
from chat_especialistas.ingestao import ingerir_documentos, pasta_documentos

cfg = carregar_config("../config/agents.yaml")
dbutils.widgets.text("volume", pasta_documentos(cfg), "Volume com os documentos")
dbutils.widgets.text("chunk_size", "1000")
dbutils.widgets.text("chunk_overlap", "150")

resultado = ingerir_documentos(
    spark,
    cfg,
    volume=dbutils.widgets.get("volume"),
    chunk_size=int(dbutils.widgets.get("chunk_size")),
    chunk_overlap=int(dbutils.widgets.get("chunk_overlap")),
)

# COMMAND ----------

# MAGIC %md ## Teste rápido: cada agente só enxerga a própria documentação

# COMMAND ----------

if resultado:
    from databricks.vector_search.client import VectorSearchClient

    indice = VectorSearchClient(disable_notice=True).get_index(cfg.vector_search.endpoint, cfg.vector_search.indice_padrao)
    for esp in [e for e in cfg.especialistas if e.documentos]:
        pergunta = esp.exemplos[0] if esp.exemplos else esp.descricao
        r = indice.similarity_search(
            query_text=pergunta, columns=["especialista", "fonte"], filters={"especialista": esp.id}, num_results=3
        )
        print(f"\n[{esp.id}] {pergunta}")
        for linha in r.get("result", {}).get("data_array", []):
            print("   ", linha)
