# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Publicar o agente geral (executado pela app)
# MAGIC
# MAGIC A app Streamlit dispara este notebook como um job quando você clica em **Gerar agente geral**.
# MAGIC Ele lê o `agents.yaml` gerado pela app (no Volume do projeto), indexa os documentos, avalia o
# MAGIC roteamento, registra o modelo no Unity Catalog e publica o endpoint. Também pode ser rodado à mão.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-agente.txt
# MAGIC %restart_python

# COMMAND ----------

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

RAIZ = Path(os.path.abspath(".."))
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))

dbutils.widgets.text("config_path", "/Volumes/main/chat_especialistas/config/agents.yaml", "agents.yaml no Volume")
dbutils.widgets.dropdown("indexar", "true", ["true", "false"], "Indexar documentos")
dbutils.widgets.dropdown("avaliar", "true", ["true", "false"], "Avaliar roteamento")
dbutils.widgets.dropdown("publicar", "true", ["true", "false"], "Registrar e publicar")

CONFIG_VOLUME = dbutils.widgets.get("config_path")
# Cópia local: é ela que vai junto do modelo (code_paths) e é lida pelo agente no endpoint.
CONFIG_LOCAL = Path(tempfile.mkdtemp()) / "agents.yaml"
shutil.copy(CONFIG_VOLUME, CONFIG_LOCAL)
os.environ["CHAT_ESPECIALISTAS_CONFIG"] = str(CONFIG_LOCAL)

import mlflow
from chat_especialistas import carregar_config

cfg = carregar_config(CONFIG_LOCAL)
usuario = spark.sql("SELECT current_user()").first()[0]
EXPERIMENTO = f"/Users/{usuario}/chat_especialistas" if "@" in usuario else "/Shared/chat_especialistas"
mlflow.set_experiment(EXPERIMENTO)
print("Agentes:", [e.id for e in cfg.especialistas], "| Modelo:", cfg.projeto.nome_modelo, "| Executado por:", usuario)
retorno = {"config": CONFIG_VOLUME, "executado_por": usuario, "experimento": EXPERIMENTO}

# COMMAND ----------

from chat_especialistas.ingestao import ingerir_documentos, preparar_volumes

preparar_volumes(spark, cfg)
if dbutils.widgets.get("indexar") == "true" and cfg.usa_documentos:
    retorno["documentos"] = ingerir_documentos(spark, cfg)

# COMMAND ----------

from chat_especialistas.graph import criar_llms_databricks
from chat_especialistas.publicacao import avaliar_roteamento

if dbutils.widgets.get("avaliar") == "true":
    av = avaliar_roteamento(cfg, criar_llms_databricks(cfg)[0])
    retorno["avaliacao"] = {"acuracia": av["acuracia"], "acertos": av["acertos"], "total": av["total"],
                            "erros": [c for c in av["casos"] if not c["acertou"]]}

# COMMAND ----------

from chat_especialistas.publicacao import registrar_e_publicar

if dbutils.widgets.get("publicar") == "true":
    retorno["publicacao"] = registrar_e_publicar(cfg, RAIZ, caminho_config=CONFIG_LOCAL, experimento=EXPERIMENTO)

# COMMAND ----------

dbutils.notebook.exit(json.dumps(retorno, default=str, ensure_ascii=False))
