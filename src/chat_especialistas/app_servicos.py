"""Serviços da app Streamlit: tudo o que conversa com o Databricks fica aqui.

Identidade: se a app tiver *user authorization* habilitada, o Databricks envia o token de quem
está logado no header `x-forwarded-access-token` e a app age COMO ESSA PESSOA (vê os modelos,
tabelas e Genies que ela vê). Sem isso, a app usa o service principal dela.
"""
from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, List, Optional

from .configurador.descoberta import Recursos, descobrir, listar_catalogos, listar_schemas, listar_tabelas
from .configurador.estudo import WarehouseFonteEstudo
from .configurador.llm_sdk import ChatServing
from .dados import executar_sql_warehouse

NOTEBOOK_PUBLICACAO = "notebooks/04_publicar_job"


class Servicos:
    def __init__(self, w, identidade: str, w_app=None, usuario: Optional[str] = None):
        self.w = w                      # cliente usado nas ações (usuário, se disponível)
        self.w_app = w_app or w         # service principal da app
        self.identidade = identidade    # "usuario" | "app"
        self.usuario = usuario

    # --------------------------------------------------------------- criação
    @classmethod
    def do_contexto(cls, headers: Optional[Dict[str, str]] = None) -> "Servicos":
        from databricks.sdk import WorkspaceClient

        headers = {k.lower(): v for k, v in (headers or {}).items()}
        w_app = WorkspaceClient()
        token = headers.get("x-forwarded-access-token")
        usuario = headers.get("x-forwarded-email") or headers.get("x-forwarded-preferred-username")
        if token:
            w = WorkspaceClient(host=w_app.config.host, token=token, auth_type="pat")
            return cls(w, "usuario", w_app, usuario)
        return cls(w_app, "app", w_app, usuario)

    # ------------------------------------------------------------ descoberta
    def recursos(self) -> Recursos:
        return descobrir(self.w)

    def catalogos(self) -> List[str]:
        return listar_catalogos(self.w)

    def schemas(self, catalogo: str) -> List[str]:
        return listar_schemas(self.w, catalogo)

    def tabelas(self, catalogo: str, schema: str) -> List[str]:
        return listar_tabelas(self.w, catalogo, schema)

    # ---------------------------------------------------------------- estudo
    def fonte_estudo(self, warehouse_id: str) -> WarehouseFonteEstudo:
        return WarehouseFonteEstudo(self.w, warehouse_id)

    def llm(self, endpoint: str) -> ChatServing:
        return ChatServing(self.w, endpoint, temperature=0.0, max_tokens=4000)

    # --------------------------------------------------------- armazenamento
    @staticmethod
    def pasta_config(catalogo: str, schema: str) -> str:
        return f"/Volumes/{catalogo}/{schema}/config"

    def preparar_projeto(self, catalogo: str, schema: str, warehouse_id: str) -> List[str]:
        """Cria schema e volumes do projeto (config, documentos, llmops). Devolve erros, se houver."""
        erros = []
        comandos = [f"CREATE SCHEMA IF NOT EXISTS `{catalogo}`.`{schema}`"] + [
            f"CREATE VOLUME IF NOT EXISTS `{catalogo}`.`{schema}`.`{v}`" for v in ("config", "documentos", "llmops")
        ]
        for sql in comandos:
            r = executar_sql_warehouse(self.w, warehouse_id, sql, 10, 120)
            if r.erro:
                erros.append(f"{sql}: {r.erro}")
        return erros

    def ler(self, caminho: str) -> Optional[str]:
        try:
            return self.w.files.download(caminho).contents.read().decode("utf-8")
        except Exception:  # noqa: BLE001 - arquivo/volume ainda não existe
            return None

    def gravar(self, caminho: str, texto: str) -> None:
        self.w.files.upload(caminho, io.BytesIO(texto.encode("utf-8")), overwrite=True)

    def contar_documentos(self, catalogo: str, schema: str, agente_id: str) -> int:
        pasta = f"/Volumes/{catalogo}/{schema}/documentos/{agente_id}"
        try:
            return sum(1 for e in self.w.files.list_directory_contents(pasta) if not e.is_directory)
        except Exception:  # noqa: BLE001
            return 0

    # ------------------------------------------------------------- publicação
    def caminho_repo(self) -> Optional[str]:
        """Pasta do repositório no workspace (= código-fonte da app)."""
        if os.environ.get("CAMINHO_REPO"):
            return os.environ["CAMINHO_REPO"].rstrip("/")
        nome = os.environ.get("DATABRICKS_APP_NAME")
        if not nome:
            return None
        try:
            app = self.w_app.apps.get(nome)
            caminho = app.default_source_code_path or (app.active_deployment.source_code_path if app.active_deployment else None)
            return caminho.rstrip("/") if caminho else None
        except Exception:  # noqa: BLE001
            return None

    def iniciar_publicacao(self, parametros: Dict[str, str]) -> Dict[str, Any]:
        """Dispara o notebook 04_publicar_job como um job avulso (serverless ou cluster informado).

        Tenta primeiro como o usuário logado; se não houver permissão, usa o service principal da app.
        """
        from databricks.sdk.service import jobs

        repo = self.caminho_repo()
        if not repo:
            raise RuntimeError(
                "Não descobri a pasta do repositório. Defina a variável CAMINHO_REPO no app.yaml "
                "(ex.: /Workspace/Users/voce@empresa.com/chat-especialistas)."
            )
        tarefa = jobs.SubmitTask(
            task_key="publicar_agente",
            notebook_task=jobs.NotebookTask(notebook_path=f"{repo}/{NOTEBOOK_PUBLICACAO}", base_parameters=parametros),
        )
        if os.environ.get("CLUSTER_ID_PUBLICACAO"):
            tarefa.existing_cluster_id = os.environ["CLUSTER_ID_PUBLICACAO"]

        erros = []
        clientes = [(self.w, self.identidade)] + ([(self.w_app, "app")] if self.w_app is not self.w else [])
        for cliente, quem in clientes:
            try:
                espera = cliente.jobs.submit(run_name="chat_especialistas · publicar agente geral", tasks=[tarefa])
                run_id = espera.response.run_id
                run = cliente.jobs.get_run(run_id)
                return {"run_id": run_id, "url": run.run_page_url, "como": quem}
            except Exception as e:  # noqa: BLE001
                erros.append(f"{quem}: {e}")
        raise RuntimeError("Não foi possível iniciar o job de publicação. " + " | ".join(erros))

    def status_publicacao(self, run_id: int, como: str = "usuario") -> Dict[str, Any]:
        cliente = self.w if como == self.identidade else self.w_app
        run = cliente.jobs.get_run(run_id)
        estado = run.state
        vida = getattr(getattr(estado, "life_cycle_state", None), "value", str(getattr(estado, "life_cycle_state", "")))
        resultado = getattr(getattr(estado, "result_state", None), "value", None)
        saida: Dict[str, Any] = {
            "estado": vida, "resultado": resultado, "url": run.run_page_url,
            "terminado": vida in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"),
            "mensagem": getattr(estado, "state_message", "") or "",
        }
        if saida["terminado"] and run.tasks:
            try:
                out = cliente.jobs.get_run_output(run.tasks[0].run_id)
                if out.notebook_output and out.notebook_output.result:
                    saida["retorno"] = json.loads(out.notebook_output.result)
                if out.error:
                    saida["erro"] = out.error
            except Exception as e:  # noqa: BLE001
                saida["erro"] = str(e)
        saida["sucesso"] = saida["terminado"] and resultado == "SUCCESS"
        return saida

    # ------------------------------------------------------------------ chat
    def conversar(self, endpoint: str, mensagens: List[Dict[str, str]], conversation_id: str) -> Dict[str, Any]:
        corpo = {
            "input": mensagens,
            "custom_inputs": {"conversation_id": conversation_id, "user_id": self.usuario or self.identidade},
        }
        return self.w.api_client.do("POST", f"/serving-endpoints/{endpoint}/invocations", body=corpo)


def extrair_resposta(resp: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    """Texto + detalhes (intenção, agentes, SQL, dados, tokens, US$) da resposta do endpoint."""
    textos = []
    for item in resp.get("output") or []:
        for parte in item.get("content") or []:
            if isinstance(parte, dict) and parte.get("text"):
                textos.append(parte["text"])
    custom = resp.get("custom_outputs") or {}
    ops = custom.get("llmops") or {}
    detalhes = {
        "intencao": custom.get("intencao"),
        "especialistas": custom.get("especialistas") or [],
        "fontes": custom.get("fontes") or [],
        "consultas_sql": custom.get("consultas_sql") or [],
        "dados": custom.get("dados") or {},
        "tokens": int(ops.get("tokens_total") or 0),
        "custo_usd": float(ops.get("custo_usd") or 0.0),
    }
    return "\n\n".join(textos) or "(resposta vazia)", detalhes
