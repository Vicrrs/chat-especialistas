"""Testa o wrapper MLflow (agent.py) com o grafo simulado."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

mlflow = pytest.importorskip("mlflow")

import chat_especialistas  # noqa: E402
from chat_especialistas import construir_grafo  # noqa: E402
from chat_especialistas.retriever import Trecho  # noqa: E402
from fakes import FonteSQLite, fake  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]


class RetrieverFalso:
    def buscar(self, pergunta, esp):
        return [Trecho(conteudo="x", fonte="politica_credito.pdf")]


def test_agent_responses(monkeypatch):
    rota = json.dumps(
        {"intencao": "especialista", "especialistas": ["credito"], "pergunta_reescrita": "Limite PJ?", "confianca": 0.95}
    )

    def grafo_falso(cfg, checkpointer=None):
        return construir_grafo(
            cfg,
            fake(rota),
            fake("Até R$ 500 mil [politica_credito.pdf]"),
            RetrieverFalso(),
            fonte_dados=FonteSQLite(),
        )

    monkeypatch.setattr(chat_especialistas, "construir_grafo_databricks", grafo_falso)
    monkeypatch.setattr(mlflow.models, "set_model", lambda m: None)
    monkeypatch.setattr(mlflow.langchain, "autolog", lambda *a, **k: None, raising=False)

    spec = importlib.util.spec_from_file_location("agent_mod", RAIZ / "agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from mlflow.types.responses import ResponsesAgentRequest

    req = ResponsesAgentRequest(
        input=[
            {"role": "user", "content": "oi"},
            {"role": "assistant", "content": "Olá!"},
            {"role": "user", "content": "qual o limite pj?"},
        ]
    )
    resp = mod.AGENT.predict(req)
    assert "R$ 500 mil" in resp.output[0].content[0]["text"]
    assert resp.custom_outputs["especialistas"] == ["credito"]
    assert resp.custom_outputs["fontes"] == ["politica_credito.pdf"]

    ops = resp.custom_outputs["llmops"]
    assert ops["qtd_chamadas_llm"] == 2
    assert ops["tokens_total"] > 0 and ops["custo_usd"] > 0
    assert any(c["etapa"] == "busca_credito" for c in ops["chamadas"])  # embedding
    assert "pergunta" not in ops  # texto fica só no request/response da inference table


def test_stream_event_aceita_custom_outputs():
    from mlflow.types.responses import ResponsesAgentStreamEvent

    ev = ResponsesAgentStreamEvent(
        type="response.output_item.done",
        item={"type": "message", "id": "1", "role": "assistant",
              "content": [{"type": "output_text", "text": "x", "annotations": []}]},
        custom_outputs={"llmops": {"custo_usd": 0.1}},
    )
    assert ev.custom_outputs["llmops"]["custo_usd"] == 0.1
