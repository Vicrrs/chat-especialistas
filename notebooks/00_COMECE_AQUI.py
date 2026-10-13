# Databricks notebook source
# MAGIC %md
# MAGIC # 🤖 Comece aqui · Chat Multi-Especialistas
# MAGIC
# MAGIC > **Recomendado:** use a **app Streamlit** (`app.py`); o passo a passo está no README, em
# MAGIC > "Criar a app a partir do Git folder". Este notebook é a mesma configuração para quem prefere notebook.
# MAGIC
# MAGIC Este assistente monta o chat inteiro sem você editar código:
# MAGIC
# MAGIC | Aba | O que você faz |
# MAGIC |---|---|
# MAGIC | **1 · Modelos e Genies** | Vê todas as IAs (endpoints de LLM e embeddings), Genie Spaces e SQL Warehouses do workspace e escolhe quais usar |
# MAGIC | **2 · Agentes** | Cria cada agente especialista: nome, **prompt de negócio**, **tabelas** (navegando no Unity Catalog), Genie e documentação |
# MAGIC | **3 · Estudar tabelas** | O sistema estuda as tabelas de cada agente (esquema, perfil dos dados, amostra) e escreve o conhecimento dele; você revisa |
# MAGIC | **4 · Agente geral** | Gera o Agente 01 (detector de intenção) com todos os especialistas, avalia o roteamento e publica o endpoint |
# MAGIC
# MAGIC **Como usar:** conecte o notebook a um cluster (DBR 15.4 LTS ou mais novo, ou Serverless) e clique em **Run all**.
# MAGIC O assistente aparece no fim da página. Tudo o que você escolhe fica salvo em `config/assistente_estado.json`,
# MAGIC então dá para fechar e continuar depois.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-agente.txt "ipywidgets>=8"
# MAGIC %restart_python

# COMMAND ----------

import os
import sys
from pathlib import Path

RAIZ = Path(os.path.abspath(".."))
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))

from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks

from chat_especialistas.configurador.estudo import SparkFonteEstudo
from chat_especialistas.configurador.interface import Assistente
from chat_especialistas.graph import criar_llms_databricks
from chat_especialistas.ingestao import contar_documentos, ingerir_documentos, preparar_volumes
from chat_especialistas.publicacao import avaliar_roteamento, registrar_e_publicar

w = WorkspaceClient()
usuario = spark.sql("SELECT current_user()").first()[0]
EXPERIMENTO = f"/Users/{usuario}/chat_especialistas"


def criar_llm(endpoint: str):
    # Usado no estudo das tabelas (precisa de respostas longas e estáveis).
    return ChatDatabricks(endpoint=endpoint, temperature=0.0, max_tokens=4000)


def ingerir(cfg):
    return ingerir_documentos(spark, cfg)


def avaliar(cfg):
    return avaliar_roteamento(cfg, criar_llms_databricks(cfg)[0])


def publicar(cfg):
    preparar_volumes(spark, cfg)
    return registrar_e_publicar(cfg, RAIZ, experimento=EXPERIMENTO, publicar=True)


assistente = Assistente(
    w,
    fonte_estudo=SparkFonteEstudo(spark, w),
    criar_llm=criar_llm,
    raiz=RAIZ,
    ingerir=ingerir,
    avaliar=avaliar,
    publicar=publicar,
    contar_documentos=contar_documentos,
)
assistente.mostrar()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Depois de publicar
# MAGIC - **Conversar e testar:** `02_registrar_e_publicar` (conversa com tokens e US$ por resposta) ou a **Review App**.
# MAGIC - **Custos e uso (LLMOps):** `03_llmops_painel`.
# MAGIC - **Documentos mudaram:** rode `01_ingestao_documentos`.
# MAGIC - **Mudar agentes:** volte aqui, ajuste, estude de novo o que mudou e clique em *Gerar agente geral*.
# MAGIC
# MAGIC Resultado da última geração (avaliação, endpoint, Review App):

# COMMAND ----------

assistente.ultimo_resultado
