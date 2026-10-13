# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Dados de exemplo para os agentes de BI (opcional)
# MAGIC
# MAGIC Cria tabelas **fictícias** de crédito e cobrança, com os nomes usados no `agents.yaml`,
# MAGIC para você testar o chat de ponta a ponta antes de apontar para as tabelas reais.
# MAGIC
# MAGIC Dica para as tabelas reais: **comentários de tabela e coluna** no Unity Catalog são lidos pelo
# MAGIC agente (`descrever_tabela`). Quanto melhores os comentários, melhor o SQL que ele escreve.

# COMMAND ----------

dbutils.widgets.text("catalogo", "main")
CAT = dbutils.widgets.get("catalogo")

for schema in ("credito", "cobranca"):
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CAT}.{schema}")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {CAT}.credito.clientes (
  id_cliente BIGINT COMMENT 'Identificador do cliente',
  segmento STRING COMMENT 'PF ou PJ',
  regiao STRING COMMENT 'Norte, Nordeste, Centro-Oeste, Sudeste, Sul',
  faturamento_anual DECIMAL(18,2) COMMENT 'Faturamento anual declarado em R$ (PJ) ou renda anual (PF)',
  data_cadastro DATE COMMENT 'Data de abertura do cadastro'
) COMMENT 'Clientes da carteira de crédito (dados fictícios)'
""")
spark.sql(f"""
INSERT INTO {CAT}.credito.clientes
SELECT id,
       CASE WHEN rand(1) < 0.4 THEN 'PJ' ELSE 'PF' END,
       element_at(array('Norte','Nordeste','Centro-Oeste','Sudeste','Sul'), CAST(floor(rand(2) * 5) AS INT) + 1),
       CAST(round(rand(3) * 5000000, 2) AS DECIMAL(18,2)),
       date_add('2023-01-01', CAST(rand(4) * 1200 AS INT))
FROM range(1, 2001) AS t(id)
""")

spark.sql(f"""
CREATE OR REPLACE TABLE {CAT}.credito.propostas (
  id_proposta BIGINT COMMENT 'Identificador da proposta',
  id_cliente BIGINT COMMENT 'FK para credito.clientes.id_cliente',
  segmento STRING COMMENT 'PF ou PJ (cópia do cliente na data da proposta)',
  produto STRING COMMENT 'Capital de giro, Cartão, Consignado, Financiamento',
  valor_solicitado DECIMAL(18,2) COMMENT 'Valor solicitado em R$',
  limite_concedido DECIMAL(18,2) COMMENT 'Limite aprovado em R$ (0 se reprovada)',
  score INT COMMENT 'Score de crédito no momento da proposta (0 a 1000)',
  status STRING COMMENT 'APROVADA ou REPROVADA',
  motivo_reprovacao STRING COMMENT 'SCORE_BAIXO, RESTRICAO, COMPROMETIMENTO_RENDA ou NULL',
  data_proposta DATE COMMENT 'Data da proposta'
) COMMENT 'Propostas de crédito (dados fictícios)'
""")
spark.sql(f"""
INSERT INTO {CAT}.credito.propostas
WITH base AS (
  SELECT id, CAST(1 + floor(rand(5) * 2000) AS BIGINT) AS id_cliente,
         CAST(300 + rand(6) * 650 AS INT) AS score,
         rand(7) AS r_restricao, rand(8) AS r_renda,
         CAST(round(5000 + rand(9) * 300000, 2) AS DECIMAL(18,2)) AS valor,
         date_add('2025-01-01', CAST(rand(10) * 640 AS INT)) AS dt,
         element_at(array('Capital de giro','Cartão','Consignado','Financiamento'), CAST(floor(rand(11) * 4) AS INT) + 1) AS produto
  FROM range(1, 15001) AS t(id)
)
SELECT b.id, b.id_cliente, c.segmento, b.produto, b.valor,
       CASE WHEN motivo IS NULL THEN b.valor ELSE 0 END,
       b.score,
       CASE WHEN motivo IS NULL THEN 'APROVADA' ELSE 'REPROVADA' END,
       motivo, b.dt
FROM (
  SELECT *, CASE WHEN score < 450 THEN 'SCORE_BAIXO'
                 WHEN r_restricao < 0.08 THEN 'RESTRICAO'
                 WHEN r_renda < 0.10 THEN 'COMPROMETIMENTO_RENDA' END AS motivo
  FROM base
) b JOIN {CAT}.credito.clientes c USING (id_cliente)
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {CAT}.cobranca.parcelas (
  id_parcela BIGINT COMMENT 'Identificador da parcela',
  id_cliente BIGINT COMMENT 'FK para credito.clientes.id_cliente',
  regiao STRING COMMENT 'Região do cliente',
  data_vencimento DATE COMMENT 'Vencimento da parcela',
  dias_atraso INT COMMENT 'Dias de atraso na data de referência (0 = em dia)',
  valor_parcela DECIMAL(18,2) COMMENT 'Valor original da parcela em R$',
  valor_aberto DECIMAL(18,2) COMMENT 'Valor ainda em aberto em R$',
  negativado BOOLEAN COMMENT 'Cliente negativado por esta parcela',
  data_referencia DATE COMMENT 'Data da foto (snapshot mensal)'
) COMMENT 'Parcelas em cobrança, foto mensal (dados fictícios)'
""")
spark.sql(f"""
INSERT INTO {CAT}.cobranca.parcelas
SELECT id,
       CAST(1 + floor(rand(12) * 2000) AS BIGINT),
       element_at(array('Norte','Nordeste','Centro-Oeste','Sudeste','Sul'), CAST(floor(rand(13) * 5) AS INT) + 1),
       date_add(ref, -CAST(rand(14) * 90 AS INT)),
       atraso,
       valor, CASE WHEN atraso > 0 THEN valor ELSE 0 END,
       atraso >= 30,
       ref
FROM (
  SELECT id, last_day(add_months('2026-09-01', -CAST(floor(rand(15) * 12) AS INT))) AS ref,
         CASE WHEN rand(16) < 0.78 THEN 0 ELSE CAST(rand(17) * 180 AS INT) END AS atraso,
         CAST(round(200 + rand(18) * 8000, 2) AS DECIMAL(18,2)) AS valor
  FROM range(1, 40001) AS t(id)
)
""")

spark.sql(f"""
CREATE OR REPLACE TABLE {CAT}.cobranca.acordos (
  id_acordo BIGINT COMMENT 'Identificador do acordo',
  id_cliente BIGINT COMMENT 'FK para credito.clientes.id_cliente',
  data_acordo DATE COMMENT 'Data do acordo',
  valor_divida DECIMAL(18,2) COMMENT 'Dívida total antes do desconto em R$',
  desconto_pct DECIMAL(5,2) COMMENT 'Desconto concedido sobre juros e multa (%)',
  parcelas INT COMMENT 'Quantidade de parcelas do acordo (1 = à vista)',
  status STRING COMMENT 'ATIVO, QUITADO ou ROMPIDO'
) COMMENT 'Acordos de renegociação (dados fictícios)'
""")
spark.sql(f"""
INSERT INTO {CAT}.cobranca.acordos
SELECT id, CAST(1 + floor(rand(19) * 2000) AS BIGINT),
       date_add('2025-06-01', CAST(rand(20) * 480 AS INT)),
       CAST(round(500 + rand(21) * 40000, 2) AS DECIMAL(18,2)),
       CAST(round(rand(22) * 40, 2) AS DECIMAL(5,2)),
       CAST(1 + floor(rand(23) * 12) AS INT),
       element_at(array('ATIVO','QUITADO','ROMPIDO'), CAST(floor(rand(24) * 3) AS INT) + 1)
FROM range(1, 3001) AS t(id)
""")

# COMMAND ----------

display(spark.sql(f"""
SELECT 'credito.propostas' AS tabela, count(*) AS linhas FROM {CAT}.credito.propostas
UNION ALL SELECT 'cobranca.parcelas', count(*) FROM {CAT}.cobranca.parcelas
UNION ALL SELECT 'cobranca.acordos', count(*) FROM {CAT}.cobranca.acordos
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC Lembre de dar acesso ao service principal do endpoint (ou aos usuários, no modo `usuario`):
# MAGIC ```sql
# MAGIC GRANT USE CATALOG ON CATALOG main TO `<principal>`;
# MAGIC GRANT USE SCHEMA, SELECT ON SCHEMA main.credito TO `<principal>`;
# MAGIC GRANT USE SCHEMA, SELECT ON SCHEMA main.cobranca TO `<principal>`;
# MAGIC ```
