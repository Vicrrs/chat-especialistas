"""Guarda de SQL: garante que o agente de BI só LÊ dados das tabelas do seu domínio.

Isto é uma camada extra. O controle de verdade continua sendo o Unity Catalog:
dê ao service principal do endpoint (ou aos usuários, no modo on-behalf-of)
apenas SELECT nas tabelas de cada domínio.
"""
from __future__ import annotations

from typing import Iterable

import sqlglot
from sqlglot import exp

PROIBIDOS = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop,
    exp.Alter, exp.Command, exp.TruncateTable, exp.Grant, exp.Use, exp.Set,
)

# Funções que saem do "só ler as tabelas do domínio": rede (exfiltração), segredos, chamadas a
# modelos, código Java, leitura de arquivos/filas e consultas a outros bancos (federação).
FUNCOES_PROIBIDAS = {
    "http_request", "secret", "try_secret", "reflect", "try_reflect", "java_method",
    "read_files", "read_kafka", "read_kinesis", "read_pubsub", "read_pulsar", "read_statestore",
    "remote_query", "vector_search", "cloud_files_state", "table_changes", "list_secrets",
}
PREFIXOS_PROIBIDOS = ("ai_",)  # ai_query, ai_gen, ai_extract...


class SQLNaoPermitido(ValueError):
    """Mensagem volta para o agente, que pode corrigir a consulta."""


def validar_sql(
    sql: str,
    tabelas_permitidas: Iterable[str],
    max_linhas: int,
    dialeto: str = "databricks",
) -> str:
    permitidas = {t.lower() for t in tabelas_permitidas}
    texto = (sql or "").strip().rstrip(";").strip()
    if not texto:
        raise SQLNaoPermitido("SQL vazio.")

    try:
        instrucoes = [i for i in sqlglot.parse(texto, read=dialeto) if i is not None]
    except sqlglot.errors.ParseError as e:
        raise SQLNaoPermitido(f"SQL inválido: {str(e).splitlines()[0]}") from e

    if len(instrucoes) != 1:
        raise SQLNaoPermitido("Envie apenas UMA instrução SELECT por vez.")
    consulta = instrucoes[0]

    if not isinstance(consulta, exp.Query):
        raise SQLNaoPermitido("Apenas consultas de leitura (SELECT / WITH ... SELECT) são permitidas.")
    for no in consulta.walk():
        if isinstance(no, PROIBIDOS):
            raise SQLNaoPermitido("Apenas leitura: comandos que alteram dados ou objetos são proibidos.")
        if isinstance(no, exp.Func):
            nome = (no.name if isinstance(no, exp.Anonymous) else no.sql_name()).lower()
            if nome in FUNCOES_PROIBIDAS or nome.startswith(PREFIXOS_PROIBIDOS):
                raise SQLNaoPermitido(f"A função {nome} não é permitida nas consultas dos agentes.")
        if isinstance(no, exp.Dot) and isinstance(no.expression, exp.Func):
            raise SQLNaoPermitido("Funções de catálogo (catalogo.schema.funcao) não são permitidas.")
        if isinstance(no, exp.Version):
            # Time travel lê versões antigas, inclusive dados já apagados (LGPD: direito de exclusão).
            raise SQLNaoPermitido("Consultas a versões antigas da tabela (VERSION/TIMESTAMP AS OF) não são permitidas.")
        if isinstance(no, exp.Stream):
            raise SQLNaoPermitido("Leitura em streaming não é permitida.")

    ctes = {c.alias_or_name.lower() for c in consulta.find_all(exp.CTE)}
    for tabela in consulta.find_all(exp.Table):
        if not tabela.name:
            raise SQLNaoPermitido("Funções de tabela (ex.: read_files) não são permitidas.")
        if not tabela.db and tabela.name.lower() in ctes:
            continue
        if not (tabela.catalog and tabela.db):
            raise SQLNaoPermitido(
                f"Use o nome completo catalogo.schema.tabela para '{tabela.name}'. "
                f"Tabelas permitidas: {sorted(permitidas)}"
            )
        nome = f"{tabela.catalog}.{tabela.db}.{tabela.name}".lower()
        if nome not in permitidas:
            raise SQLNaoPermitido(
                f"A tabela {nome} não pertence a este domínio. Tabelas permitidas: {sorted(permitidas)}"
            )

    # LIMIT só vale como número literal; expressões (ex.: 10*1000000) e LIMIT ALL viram o máximo.
    limite = consulta.args.get("limit")
    valor = None
    if isinstance(limite, exp.Limit) and isinstance(limite.expression, exp.Literal) and limite.expression.is_int:
        valor = int(limite.expression.name)
    if valor is None or valor > max_linhas:
        consulta.set("limit", None)
        consulta = consulta.limit(max_linhas)

    # Sem comentários: o que roda no warehouse é exatamente o que foi validado.
    return consulta.sql(dialect=dialeto, comments=False)
