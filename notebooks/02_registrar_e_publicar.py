# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Testar, avaliar, registrar e publicar o agente geral
# MAGIC
# MAGIC > O assistente **00_COMECE_AQUI** já faz avaliação, registro e deploy. Use este notebook para
# MAGIC > **conversar com o agente** antes de publicar, ou para republicar depois de editar o `agents.yaml`.
# MAGIC
# MAGIC 1. Conversa de teste (direto do código, com tokens e US$ de cada resposta)
# MAGIC 2. Avaliação do detector de intenção com as perguntas geradas no estudo
# MAGIC 3. Registro no MLflow / Unity Catalog e deploy (Model Serving + Review App)

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-agente.txt
# MAGIC %restart_python

# COMMAND ----------

import os
import sys
from pathlib import Path

RAIZ = Path(os.path.abspath(".."))
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))
os.environ["CHAT_ESPECIALISTAS_CONFIG"] = str(RAIZ / "config" / "agents.yaml")

import mlflow
from chat_especialistas import carregar_config
from chat_especialistas.ingestao import preparar_volumes

cfg = carregar_config()
usuario = spark.sql("SELECT current_user()").first()[0]
# Experimento fixo: é onde ficam os traces (inclusive os do endpoint) que o notebook 03 lê.
EXPERIMENTO = f"/Users/{usuario}/chat_especialistas"
mlflow.set_experiment(EXPERIMENTO)
preparar_volumes(spark, cfg)
print("Agentes:", [e.id for e in cfg.especialistas], "| Modelo:", cfg.projeto.nome_modelo)

# COMMAND ----------

# MAGIC %md ## 1. Conversa de teste

# COMMAND ----------

from agent import AGENT

# Só nesta sessão: também grava os registros LLMOps em JSONL no Volume.
if "jsonl" not in AGENT.cfg.llmops.destinos:
    AGENT.cfg.llmops.destinos.append("jsonl")

historico = []
gasto_sessao = {"tokens": 0, "usd": 0.0}


def perguntar(texto: str):
    historico.append({"role": "user", "content": texto})
    resp = AGENT.predict({"input": historico, "custom_inputs": {"conversation_id": "teste-notebook", "user_id": usuario}})
    resposta = resp.output[0].content[0]["text"]
    historico.append({"role": "assistant", "content": resposta})
    ops = resp.custom_outputs["llmops"]
    gasto_sessao["tokens"] += ops["tokens_total"]
    gasto_sessao["usd"] += ops["custo_usd"]
    print(f"👤 {texto}\n🤖 {resposta}")
    print(
        f"   ↳ intenção={ops['intencao']} agentes={ops['especialistas']} "
        f"tokens={ops['tokens_input']}+{ops['tokens_output']} custo=US$ {ops['custo_usd']:.6f} "
        f"SQLs={ops['qtd_consultas_sql']} latência={ops['latencia_ms']} ms"
    )
    for esp_id, dados in resp.custom_outputs.get("dados", {}).items():
        if dados["linhas"]:
            display(spark.createDataFrame([[None if v is None else str(v) for v in l] for l in dados["linhas"]],
                                          ", ".join(f"`{c}` string" for c in dados["colunas"])))
    print()


perguntar("Oi, tudo bem?")
for esp in cfg.especialistas:
    if esp.exemplos:
        perguntar(esp.exemplos[0])
print(f"Total da sessão: {gasto_sessao['tokens']} tokens · US$ {gasto_sessao['usd']:.6f}")

# COMMAND ----------

# Faça suas próprias perguntas (o histórico é mantido):
# perguntar("...")

# COMMAND ----------

# MAGIC %md ## 2. Avaliação do detector de intenção

# COMMAND ----------

from chat_especialistas.graph import criar_llms_databricks
from chat_especialistas.publicacao import avaliar_roteamento

avaliacao = avaliar_roteamento(cfg, criar_llms_databricks(cfg)[0])
display(spark.createDataFrame(
    [(c["pergunta"], c["esperado"], c["obtido"], c["intencao"], float(c["confianca"]), c["acertou"]) for c in avaliacao["casos"]],
    "pergunta string, esperado array<string>, obtido array<string>, intencao string, confianca double, acertou boolean",
))

# COMMAND ----------

# MAGIC %md ## 3. Registrar e publicar

# COMMAND ----------

from chat_especialistas.publicacao import registrar_e_publicar

resultado = registrar_e_publicar(cfg, RAIZ, experimento=EXPERIMENTO, publicar=True)
resultado
