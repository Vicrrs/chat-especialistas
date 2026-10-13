"""Testes de tokens, custo e destinos do LLMOps."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from chat_especialistas import carregar_config
from chat_especialistas import construir_grafo as _construir
from fakes import FakeToolModel, FonteSQLite
from chat_especialistas.llmops import RastreadorUso, custo_usd, emitir, montar_registro
from chat_especialistas.retriever import Trecho

CONFIG = Path(__file__).resolve().parents[1] / "config" / "agents.yaml"
MODELO = "databricks-meta-llama-3-3-70b-instruct"


class RetrieverFalso:
    def buscar(self, pergunta, esp):
        return [Trecho(conteudo="x", fonte=f"{esp.id}.pdf")]


def construir_grafo(*args, **kwargs):
    kwargs.setdefault("fonte_dados", FonteSQLite())
    return _construir(*args, **kwargs)


def msg(texto, tin=None, tout=None):
    if tin is None:
        return AIMessage(texto)
    return AIMessage(texto, usage_metadata={"input_tokens": tin, "output_tokens": tout, "total_tokens": tin + tout})


def fake(*mensagens):
    return FakeToolModel(messages=iter(list(mensagens)))


@pytest.fixture
def cfg():
    return carregar_config(CONFIG)


def test_custo_pela_tabela_de_precos(cfg):
    # 1M input a 0.50 + 1M output a 1.50
    assert custo_usd(cfg.llmops, MODELO, 1_000_000, 1_000_000) == pytest.approx(2.0)
    # endpoint desconhecido usa preço padrão (1.00 / 3.00)
    assert custo_usd(cfg.llmops, "outro", 1_000_000, 0) == pytest.approx(1.0)


def test_tokens_e_custo_por_etapa_com_dois_especialistas(cfg):
    rota = json.dumps({"intencao": "especialista", "especialistas": ["credito", "cobranca"],
                       "pergunta_reescrita": "limite e atraso", "confianca": 0.9})
    g = construir_grafo(
        cfg,
        fake(msg(rota, 1000, 100)),
        fake(msg("a", 2000, 200), msg("b", 2000, 200), msg("final", 500, 300)),
        RetrieverFalso(),
        nome_modelo_roteador=MODELO,
        nome_modelo_especialista=MODELO,
    )
    r = RastreadorUso(cfg.llmops)
    estado = g.invoke({"messages": [HumanMessage("limite e atraso?")]}, config={"callbacks": [r]})

    etapas = sorted(c.etapa for c in r.chamadas)
    assert etapas == ["agente_principal", "consolidar", "especialista_cobranca", "especialista_credito"]
    assert all(c.modelo == MODELO and not c.estimado for c in r.chamadas)

    reg = montar_registro(cfg, rastreador=r, pergunta="limite e atraso?", resposta="final", estado=estado)
    assert reg["tokens_input"] == 5500
    assert reg["tokens_output"] == 800
    assert reg["custo_usd"] == pytest.approx((5500 * 0.50 + 800 * 1.50) / 1e6)
    assert reg["custo_por_etapa"]["agente_principal"]["tokens"] == 1100
    assert reg["especialistas"] == ["credito", "cobranca"]


def test_estima_tokens_quando_endpoint_nao_informa(cfg):
    rota = json.dumps({"intencao": "conversa_geral", "especialistas": [], "confianca": 1})
    g = construir_grafo(cfg, fake(msg(rota)), fake(msg("Olá!")), RetrieverFalso(),
                        nome_modelo_roteador=MODELO, nome_modelo_especialista=MODELO)
    r = RastreadorUso(cfg.llmops)
    g.invoke({"messages": [HumanMessage("oi")]}, config={"callbacks": [r]})
    reg = montar_registro(cfg, rastreador=r, pergunta="oi", resposta="Olá!")
    assert reg["tokens_estimados"] is True
    assert reg["tokens_input"] > 0 and reg["custo_usd"] > 0


def test_destino_jsonl_e_textos_ocultos(cfg, tmp_path):
    cfg.llmops.destinos = ["jsonl", "log"]
    cfg.llmops.arquivo_jsonl = str(tmp_path / "sub" / "log.jsonl")
    cfg.llmops.registrar_textos = False
    r = RastreadorUso(cfg.llmops)
    r.registrar_embedding("busca_credito", "databricks-gte-large-en", "a" * 400)
    emitir(cfg, montar_registro(cfg, rastreador=r, pergunta="cpf 123", resposta="ok"))

    linha = json.loads(Path(cfg.llmops.arquivo_jsonl).read_text().strip())
    assert linha["pergunta"] is None and linha["resposta"] is None
    assert linha["tokens_input"] == 100
    assert linha["custo_usd"] == pytest.approx(100 * 0.13 / 1e6)
