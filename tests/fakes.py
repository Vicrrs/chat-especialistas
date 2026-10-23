"""Dublês para testes: LLM que suporta ferramentas e fonte de dados em SQLite."""
from __future__ import annotations

import sqlite3
import time
from typing import Any, Dict, List

import sqlglot
from sqlglot import exp
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from chat_especialistas.dados import ResultadoSQL
from chat_especialistas.retriever import Trecho


class FakeToolModel(GenericFakeChatModel):
    """GenericFakeChatModel que aceita bind_tools e registra as mensagens recebidas."""

    ferramentas_vinculadas: List[str] = []

    def bind_tools(self, tools, **kwargs):
        self.ferramentas_vinculadas = [t.name for t in tools]
        return self


def fake(*respostas) -> FakeToolModel:
    msgs = [r if isinstance(r, AIMessage) else AIMessage(r) for r in respostas]
    return FakeToolModel(messages=iter(msgs))


def chamar(nome: str, id_: str = "c1", **args) -> AIMessage:
    """AIMessage pedindo uma ferramenta."""
    return AIMessage("", tool_calls=[{"name": nome, "args": args, "id": id_}])


class RetrieverFalso:
    def __init__(self):
        self.chamadas = []

    def buscar(self, pergunta, esp) -> List[Trecho]:
        self.chamadas.append((esp.id, pergunta))
        return [Trecho(conteudo=f"regra de {esp.id}", fonte=f"manual_{esp.id}.pdf")]


class FonteSQLite:
    """Executa SQL de verdade: traduz Databricks -> SQLite e mapeia catalogo.schema.tabela."""

    def __init__(self):
        self.con = sqlite3.connect(":memory:", check_same_thread=False)
        self.executadas: List[str] = []
        c = self.con
        c.execute("CREATE TABLE credito__propostas (id INT, segmento TEXT, status TEXT, valor REAL, data_proposta TEXT)")
        c.executemany(
            "INSERT INTO credito__propostas VALUES (?,?,?,?,?)",
            [
                (1, "PJ", "APROVADA", 100000, "2026-07-10"),
                (2, "PJ", "REPROVADA", 50000, "2026-08-02"),
                (3, "PJ", "APROVADA", 80000, "2026-09-15"),
                (4, "PF", "APROVADA", 10000, "2026-09-20"),
                (5, "PJ", "APROVADA", 120000, "2026-09-28"),
            ],
        )
        c.execute("CREATE TABLE cobranca__parcelas (id INT, regiao TEXT, dias_atraso INT, valor_aberto REAL)")
        c.executemany(
            "INSERT INTO cobranca__parcelas VALUES (?,?,?,?)",
            [(1, "Norte", 45, 1000.0), (2, "Norte", 10, 500.0), (3, "Sul", 60, 2000.0)],
        )

    def descrever_tabela(self, esp, tabela: str) -> str:
        nome = tabela.split(".", 1)[1].replace(".", "__")
        cols = self.con.execute(f"PRAGMA table_info({nome})").fetchall()
        return f"Tabela {tabela}\n" + "\n".join(f"  - {c[1]} ({c[2]})" for c in cols)

    def executar_sql(self, esp, sql: str) -> ResultadoSQL:
        arvore = sqlglot.parse_one(sql, read="databricks")
        for t in arvore.find_all(exp.Table):
            if t.catalog and t.db:
                t.replace(exp.to_table(f"{t.db}__{t.name}"))
        sql_lite = arvore.sql(dialect="sqlite")
        self.executadas.append(sql_lite)
        inicio = time.perf_counter()
        try:
            cur = self.con.execute(sql_lite)
        except sqlite3.Error as e:
            return ResultadoSQL(erro=str(e))
        linhas = [list(r) for r in cur.fetchall()]
        return ResultadoSQL(
            colunas=[d[0] for d in cur.description],
            linhas=linhas,
            duracao_ms=int((time.perf_counter() - inicio) * 1000),
        )

    def perguntar_genie(self, esp, pergunta: str) -> Dict[str, Any]:
        return {"texto": "resposta genie", "sql": "SELECT 1", "colunas": ["x"], "linhas": [[1]], "duracao_ms": 5}


# ---------------------------------------------------------- assistente / estudo
from types import SimpleNamespace as NS

from chat_especialistas.configurador.estudo import Coluna


def _adicionar_metodos_estudo():
    def colunas(self, tabela):
        nome = tabela.split(".", 1)[1].replace(".", "__")
        cols = self.con.execute(f"PRAGMA table_info({nome})").fetchall()
        if not cols:
            raise ValueError(f"Tabela {tabela} não existe")
        return [Coluna(c[1], c[2].lower(), "") for c in cols]

    def comentario_tabela(self, tabela):
        return ""

    def consultar(self, sql):
        return self.executar_sql(None, sql)

    def descrever_genie(self, space_id):
        return f"Genie Space 'Cobrança': inadimplência por região ({space_id})"

    FonteSQLite.colunas = colunas
    FonteSQLite.comentario_tabela = comentario_tabela
    FonteSQLite.consultar = consultar
    FonteSQLite.descrever_genie = descrever_genie


_adicionar_metodos_estudo()


class FakeWorkspace:
    """Imita o WorkspaceClient só no que o assistente usa para descobrir recursos."""

    def __init__(self):
        def ep(nome, task, ready="READY", fm=True):
            ent = NS(external_model=None, foundation_model=NS(display_name=nome) if fm else None, entity_name=None)
            return NS(name=nome, task=task, state=NS(ready=ready), config=NS(served_entities=[ent]))

        self.serving_endpoints = NS(list=lambda: [
            ep("databricks-meta-llama-3-3-70b-instruct", "llm/v1/chat"),
            ep("databricks-claude-sonnet", "llm/v1/chat"),
            ep("databricks-gte-large-en", "llm/v1/embeddings"),
            ep("modelo-churn", "", fm=False),
        ])
        self.config = NS(host="https://ws", client_id=None)
        self.genie = FakeGenie()
        self.warehouses = NS(list=lambda: [NS(id="wh1", name="Serverless Starter", state=NS(value="RUNNING"),
                                              enable_serverless_compute=True, cluster_size="Small")])
        self.catalogs = NS(list=lambda: [NS(name="main"), NS(name="system")])
        self.schemas = NS(list=lambda catalog_name: [NS(name="credito"), NS(name="cobranca"), NS(name="information_schema")])
        self.tables = NS(list=lambda catalog_name, schema_name, **k: [
            NS(full_name=f"{catalog_name}.{schema_name}.{n}") for n in {"credito": ["propostas"], "cobranca": ["parcelas"]}.get(schema_name, [])
        ])


class FakeGenie:
    """Imita w.genie: espaços (criar/ler/atualizar), conversas com feedback e a avaliação nativa."""

    def __init__(self):
        self.espacos: Dict[str, Dict[str, Any]] = {}
        self.conversas: Dict[str, Dict[str, list]] = {}  # space_id -> conversa_id -> mensagens
        self.feedbacks: list = []
        self.avaliacao = None  # [(pergunta, "GOOD"/"BAD", motivos, sql_esperado, sql_obtido)]; None = API indisponível

    def list_spaces(self, page_size=None, page_token=None):
        return NS(spaces=[NS(space_id="g1", title="Cobrança", description="Inadimplência e acordos", warehouse_id="wh1")],
                  next_page_token=None)

    def create_space(self, warehouse_id, serialized_space, *, title=None, description=None, parent_path=None):
        sid = f"genie_{len(self.espacos) + 1}"
        self.espacos[sid] = {"serialized_space": serialized_space, "title": title, "description": description,
                             "warehouse_id": warehouse_id, "etag": "1"}
        return NS(space_id=sid)

    def get_space(self, space_id, include_serialized_space=None):
        e = self.espacos[space_id]
        return NS(space_id=space_id, title=e["title"], description=e["description"],
                  serialized_space=e["serialized_space"], etag=e["etag"])

    def update_space(self, space_id, **k):
        e = self.espacos[space_id]
        assert k.get("etag") == e["etag"], "etag desatualizado"
        e.update({x: v for x, v in k.items() if v is not None and x != "etag"}, etag=str(int(e["etag"]) + 1))
        return NS(space_id=space_id)

    def send_message_feedback(self, space_id, conversation_id, message_id, rating, comment=None):
        self.feedbacks.append((space_id, conversation_id, message_id, getattr(rating, "value", rating), comment))

    def list_conversations(self, space_id, include_all=None, page_size=None, page_token=None):
        return NS(conversations=[NS(conversation_id=c) for c in self.conversas.get(space_id, {})], next_page_token=None)

    def list_conversation_messages(self, space_id, conversation_id, page_size=None, page_token=None):
        return NS(messages=self.conversas[space_id][conversation_id], next_page_token=None)

    def genie_create_eval_run(self, space_id, benchmark_question_ids=None):
        if self.avaliacao is None:
            raise Exception("Evaluation API is not enabled")
        return NS(eval_run_id="r1", eval_run_status="RUNNING")

    def genie_get_eval_run(self, space_id, eval_run_id):
        ok = sum(1 for x in self.avaliacao if x[1] == "GOOD")
        return NS(eval_run_id=eval_run_id, eval_run_status="DONE", num_correct=ok, num_questions=len(self.avaliacao))

    def genie_list_eval_results(self, space_id, eval_run_id, page_size=None, page_token=None):
        return NS(eval_results=[NS(result_id=str(i), question=x[0], benchmark_answer=x[3])
                                for i, x in enumerate(self.avaliacao)], next_page_token=None)

    def genie_get_eval_result_details(self, space_id, eval_run_id, result_id):
        _, aval, motivos, esperado, obtido = self.avaliacao[int(result_id)]
        return NS(assessment=aval, assessment_reasons=motivos,
                  expected_response=[NS(response=esperado)], actual_response=[NS(response=obtido)])


def mensagem_genie(conteudo, ts, rating=None, comentario=None, sql=None, erro=None):
    return NS(content=conteudo, created_timestamp=ts, last_updated_timestamp=ts,
              feedback=NS(rating=rating, comment=comentario) if rating else None,
              error=NS(error=erro) if erro else None,
              attachments=[NS(query=NS(query=sql))] if sql else [])
