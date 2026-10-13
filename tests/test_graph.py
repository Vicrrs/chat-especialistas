"""Testes do fluxo do grafo sem Databricks (LLM e Vector Search simulados)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from chat_especialistas import carregar_config
from chat_especialistas import construir_grafo as _construir
from fakes import FonteSQLite, RetrieverFalso, fake

CONFIG = Path(__file__).resolve().parents[1] / "config" / "agents.yaml"


def construir_grafo(*args, **kwargs):
    kwargs.setdefault("fonte_dados", FonteSQLite())
    return _construir(*args, **kwargs)


def llm(respostas):
    return fake(*respostas)


def rota(intencao, especialistas=(), confianca=0.9, reescrita="pergunta", esclarecimento=""):
    return json.dumps(
        {
            "intencao": intencao,
            "especialistas": list(especialistas),
            "pergunta_reescrita": reescrita,
            "pergunta_esclarecimento": esclarecimento,
            "confianca": confianca,
        }
    )


@pytest.fixture
def cfg():
    return carregar_config(CONFIG)


def test_roteia_para_um_especialista(cfg):
    ret = RetrieverFalso()
    g = construir_grafo(
        cfg,
        llm([rota("especialista", ["cobranca"], reescrita="Com quantos dias negativa?")]),
        llm(["Negativa com 30 dias [manual_cobranca.pdf]"]),
        ret,
    )
    out = g.invoke({"messages": [HumanMessage("quando negativa?")]})
    assert out["intencao"] == "especialista"
    assert ret.chamadas == [("cobranca", "Com quantos dias negativa?")]
    assert "30 dias" in out["messages"][-1].content
    assert out["messages"][-1].additional_kwargs["roteamento"]["fontes"] == ["manual_cobranca.pdf"]


def test_dois_especialistas_em_paralelo_e_consolida(cfg):
    ret = RetrieverFalso()
    g = construir_grafo(
        cfg,
        llm([rota("especialista", ["credito", "cobranca"])]),
        llm(["resp A", "resp B", "resposta consolidada"]),
        ret,
    )
    out = g.invoke({"messages": [HumanMessage("limite e atraso?")]})
    assert sorted(c[0] for c in ret.chamadas) == ["cobranca", "credito"]
    assert out["messages"][-1].content == "resposta consolidada"


def test_confianca_baixa_pede_esclarecimento(cfg):
    ret = RetrieverFalso()
    g = construir_grafo(
        cfg,
        llm([rota("especialista", ["credito"], confianca=0.3, esclarecimento="É sobre limite ou cobrança?")]),
        llm([]),
        ret,
    )
    out = g.invoke({"messages": [HumanMessage("e o prazo?")]})
    assert out["intencao"] == "esclarecer"
    assert ret.chamadas == []
    assert out["messages"][-1].content == "É sobre limite ou cobrança?"


def test_especialista_inexistente_e_json_invalido_viram_esclarecimento(cfg):
    g = construir_grafo(cfg, llm([rota("especialista", ["rh"]), "não sou json"]), llm([]), RetrieverFalso())
    assert g.invoke({"messages": [HumanMessage("a")]})["intencao"] == "esclarecer"
    assert g.invoke({"messages": [HumanMessage("b")]})["intencao"] == "esclarecer"


def test_conversa_geral_e_fora_do_escopo(cfg):
    g = construir_grafo(
        cfg,
        llm([rota("conversa_geral"), rota("fora_do_escopo")]),
        llm(["Olá! Posso ajudar com crédito, cobrança e cadastro."]),
        RetrieverFalso(),
    )
    assert "Olá" in g.invoke({"messages": [HumanMessage("oi")]})["messages"][-1].content
    out = g.invoke({"messages": [HumanMessage("receita de bolo")]})
    assert "Especialista em Crédito" in out["messages"][-1].content


def test_memoria_entre_turnos_zera_respostas(cfg):
    ret = RetrieverFalso()
    g = construir_grafo(
        cfg,
        llm([rota("especialista", ["credito"]), rota("especialista", ["cadastro"])]),
        llm(["r1", "r2"]),
        ret,
        checkpointer=MemorySaver(),
    )
    conf = {"configurable": {"thread_id": "t1"}}
    g.invoke({"messages": [HumanMessage("limite PJ?")]}, conf)
    out = g.invoke({"messages": [HumanMessage("e os documentos?")]}, conf)
    assert len(out["messages"]) == 4
    assert [r["especialista"] for r in out["respostas"]] == ["cadastro"]
    assert out["messages"][-1].content == "r2"
