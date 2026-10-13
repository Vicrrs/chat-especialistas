"""Acesso a dados dos agentes de BI: metadados do Unity Catalog, SQL Warehouse e Genie."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from .config import AppConfig, EspecialistaConfig


@dataclass
class ResultadoSQL:
    colunas: List[str] = field(default_factory=list)
    linhas: List[List[Any]] = field(default_factory=list)
    truncado: bool = False
    duracao_ms: int = 0
    erro: Optional[str] = None


class FonteDados(Protocol):
    def descrever_tabela(self, esp: EspecialistaConfig, tabela: str) -> str: ...
    def executar_sql(self, esp: EspecialistaConfig, sql: str) -> ResultadoSQL: ...
    def perguntar_genie(self, esp: EspecialistaConfig, pergunta: str) -> Dict[str, Any]: ...


class DatabricksFonteDados:
    """Unity Catalog (metadados) + Statement Execution API (SQL Warehouse) + Genie."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._cliente_servico = None
        self._cache_descricao: Dict[str, str] = {}

    # ----------------------------------------------------------- cliente
    def _cliente(self):
        from databricks.sdk import WorkspaceClient

        if self.cfg.bi.autenticacao == "usuario":
            # On-behalf-of-user: criado por requisição, com as credenciais de quem pergunta.
            from databricks.sdk.credentials_provider import ModelServingUserCredentials

            return WorkspaceClient(credentials_strategy=ModelServingUserCredentials())
        if self._cliente_servico is None:
            self._cliente_servico = WorkspaceClient()
        return self._cliente_servico

    # --------------------------------------------------------- metadados
    def descrever_tabela(self, esp: EspecialistaConfig, tabela: str) -> str:
        tabela = tabela.strip().lower()
        if tabela in self._cache_descricao:
            return self._cache_descricao[tabela]
        info = self._cliente().tables.get(tabela)
        linhas = [f"Tabela {tabela}" + (f" — {info.comment}" if info.comment else "")]
        for c in info.columns or []:
            linha = f"  - {c.name} ({c.type_text})"
            if c.comment:
                linha += f": {c.comment}"
            linhas.append(linha)
        texto = "\n".join(linhas)
        self._cache_descricao[tabela] = texto
        return texto

    # --------------------------------------------------------------- SQL
    def executar_sql(self, esp: EspecialistaConfig, sql: str) -> ResultadoSQL:
        from databricks.sdk.service.sql import (
            Disposition,
            ExecuteStatementRequestOnWaitTimeout,
            Format,
            StatementState,
        )

        bi = self.cfg.bi
        warehouse = self.cfg.warehouse_de(esp)
        if not warehouse or warehouse.startswith("COLOQUE"):
            return ResultadoSQL(erro="warehouse_id não configurado em agents.yaml (bi.warehouse_id).")

        w = self._cliente()
        inicio = time.perf_counter()
        resp = w.statement_execution.execute_statement(
            statement=sql,
            warehouse_id=warehouse,
            wait_timeout="30s",
            on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE,
            row_limit=bi.max_linhas,
            format=Format.JSON_ARRAY,
            disposition=Disposition.INLINE,
        )
        limite = time.time() + bi.timeout_segundos
        while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
            if time.time() > limite:
                w.statement_execution.cancel_execution(resp.statement_id)
                return ResultadoSQL(
                    erro=f"Consulta passou de {bi.timeout_segundos}s e foi cancelada. Simplifique ou agregue mais.",
                    duracao_ms=int((time.perf_counter() - inicio) * 1000),
                )
            time.sleep(1)
            resp = w.statement_execution.get_statement(resp.statement_id)

        duracao = int((time.perf_counter() - inicio) * 1000)
        if resp.status.state != StatementState.SUCCEEDED:
            msg = resp.status.error.message if resp.status.error else str(resp.status.state)
            return ResultadoSQL(erro=msg, duracao_ms=duracao)

        colunas = [c.name for c in resp.manifest.schema.columns] if resp.manifest else []
        linhas = (resp.result.data_array or []) if resp.result else []
        truncado = bool(resp.manifest and resp.manifest.truncated)
        return ResultadoSQL(colunas=colunas, linhas=linhas, truncado=truncado, duracao_ms=duracao)

    # ------------------------------------------------------------- Genie
    def perguntar_genie(self, esp: EspecialistaConfig, pergunta: str) -> Dict[str, Any]:
        w = self._cliente()
        inicio = time.perf_counter()
        msg = w.genie.start_conversation_and_wait(esp.genie_space_id, pergunta)
        saida: Dict[str, Any] = {"texto": None, "sql": None, "colunas": [], "linhas": []}
        for anexo in msg.attachments or []:
            if anexo.text and anexo.text.content:
                saida["texto"] = anexo.text.content
            if anexo.query:
                saida["sql"] = anexo.query.query
                saida["descricao"] = anexo.query.description
                res = w.genie.get_message_attachment_query_result(
                    esp.genie_space_id, msg.conversation_id, msg.message_id or msg.id, anexo.attachment_id
                )
                st = res.statement_response
                if st and st.manifest:
                    saida["colunas"] = [c.name for c in st.manifest.schema.columns]
                    saida["linhas"] = (st.result.data_array or [])[: self.cfg.bi.linhas_para_o_llm] if st.result else []
        if msg.error:
            saida["erro"] = str(msg.error)
        saida["duracao_ms"] = int((time.perf_counter() - inicio) * 1000)
        return saida
