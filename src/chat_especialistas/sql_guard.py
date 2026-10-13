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

    limite = consulta.args.get("limit")
    valor = None
    if limite is not None:
        try:
            valor = int(limite.expression.name)
        except (AttributeError, ValueError):
            valor = None
    if valor is None or valor > max_linhas:
        consulta = consulta.limit(max_linhas)

    return consulta.sql(dialect=dialeto)
