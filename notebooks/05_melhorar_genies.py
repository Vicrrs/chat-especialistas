# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Melhorar os Genie Spaces dos especialistas (job agendado pela app)
# MAGIC
# MAGIC Para cada especialista com Genie Space criado pela app:
# MAGIC 1. lê as mensagens novas do Genie com 👎 ou erro;
# MAGIC 2. roda o benchmark do espaço e guarda a nota no histórico;
# MAGIC 3. verifica se as tabelas mudaram desde o último estudo;
# MAGIC 4. gera propostas (instrução e/ou SQL de exemplo já executado) para alguém aprovar na app.
# MAGIC
# MAGIC **Nada é aplicado ao Genie aqui.** As propostas aparecem na aba *6 · Melhorar Genies* da app.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-agente.txt
# MAGIC %restart_python

# COMMAND ----------

import json
import os
import sys
from pathlib import Path

RAIZ = Path(os.path.abspath(".."))
sys.path.insert(0, str(RAIZ / "src"))

dbutils.widgets.text("estado_path", "/Volumes/main/chat_especialistas/config/assistente_estado.json", "Estado da app")
dbutils.widgets.text("melhoria_path", "/Volumes/main/chat_especialistas/config/melhoria.json", "Arquivo de melhoria")
dbutils.widgets.text("llm", "databricks-meta-llama-3-3-70b-instruct", "LLM das propostas")
dbutils.widgets.text("warehouse_id", "", "SQL Warehouse")
dbutils.widgets.dropdown("benchmark", "true", ["true", "false"], "Rodar benchmark")

from databricks.sdk import WorkspaceClient

from chat_especialistas.configurador.estudo import WarehouseFonteEstudo
from chat_especialistas.configurador.gerador import estado_de_json
from chat_especialistas.configurador.llm_sdk import ChatServing
from chat_especialistas.configurador.melhoria import EstadoMelhoria, ciclo_melhoria

w = WorkspaceClient()
_, agentes, _ = estado_de_json(Path(dbutils.widgets.get("estado_path")).read_text(encoding="utf-8"))
caminho_melhoria = Path(dbutils.widgets.get("melhoria_path"))
warehouse = dbutils.widgets.get("warehouse_id")
print("Especialistas com Genie da app:", [a.id for a in agentes if a.genie_gerado])

# COMMAND ----------

# Lê o arquivo de novo logo antes de gravar: a app pode ter aprovado propostas enquanto o job rodava.
estado = EstadoMelhoria.from_json(caminho_melhoria.read_text(encoding="utf-8") if caminho_melhoria.exists() else None)
resumo = ciclo_melhoria(
    w, ChatServing(w, dbutils.widgets.get("llm")), WarehouseFonteEstudo(w, warehouse), agentes, estado, warehouse,
    benchmark=dbutils.widgets.get("benchmark") == "true",
)

atual = EstadoMelhoria.from_json(caminho_melhoria.read_text(encoding="utf-8") if caminho_melhoria.exists() else None)
decididas = {p["id"]: p for p in atual.propostas if p["status"] != "pendente"}
estado.propostas = [decididas.get(p["id"], p) for p in estado.propostas]
estado.casos += [c for c in atual.casos if not estado.ja_conhecido(c["agente_id"], c["pergunta"])]
caminho_melhoria.write_text(estado.to_json(), encoding="utf-8")
print(json.dumps(resumo, ensure_ascii=False, indent=2, default=str))

# COMMAND ----------

dbutils.notebook.exit(json.dumps(resumo, default=str, ensure_ascii=False))
