"""Agente de BI conversacional: analisa, consulta dados (SQLite simulando o warehouse) e responde."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from chat_especialistas import carregar_config, construir_grafo
from chat_especialistas.llmops import RastreadorUso, montar_registro
from fakes import FonteSQLite, RetrieverFalso, chamar, fake

CONFIG = Path(__file__).resolve().parents[1] / "config" / "agents.yaml"
MODELO = "databricks-meta-llama-3-3-70b-instruct"

SQL_TAXA = (
    "SELECT avg(CASE WHEN status = 'APROVADA' THEN 1.0 ELSE 0 END) AS taxa_aprovacao, count(*) AS propostas "
    "FROM main.credito.propostas WHERE segmento = 'PJ' "
    "AND data_proposta BETWEEN '2026-07-01' AND '2026-09-30'"
)


def rota(*especialistas, reescrita="Taxa de aprovação PJ no 3º trimestre de 2026"):
    return json.dumps({"intencao": "especialista", "especialistas": list(especialistas),
                       "pergunta_reescrita": reescrita, "confianca": 0.95})


@pytest.fixture
def cfg():
    return carregar_config(CONFIG)


def grafo(cfg, roteador, especialista, fonte=None, ret=None):
    return construir_grafo(
        cfg, roteador, especialista, ret or RetrieverFalso(),
        fonte_dados=fonte or FonteSQLite(),
        nome_modelo_roteador=MODELO, nome_modelo_especialista=MODELO,
    )


def test_agente_bi_analisa_consulta_e_responde(cfg):
    fonte = FonteSQLite()
    llm_esp = fake(
        chamar("descrever_tabela", "c1", tabela="main.credito.propostas"),
        chamar("executar_sql", "c2", sql=SQL_TAXA, objetivo="taxa de aprovação PJ no 3T"),
        "A taxa de aprovação PJ no 3º trimestre foi **75%** (3 de 4 propostas).\n```sql\n...\n```",
    )
    g = grafo(cfg, fake(rota("credito")), llm_esp, fonte)
    r = RastreadorUso(cfg.llmops)
    out = g.invoke({"messages": [HumanMessage("qual a taxa de aprovação PJ no último tri?")]},
                   config={"callbacks": [r]})

    assert set(llm_esp.ferramentas_vinculadas) == {"consultar_regras_negocio", "descrever_tabela", "executar_sql"}
    resp = out["respostas"][0]
    assert resp["dados"]["colunas"] == ["taxa_aprovacao", "propostas"]
    assert resp["dados"]["linhas"] == [[0.75, 4]]  # número veio do banco, não do LLM
    assert resp["consultas_sql"][0]["linhas"] == 1 and "LIMIT 200" in resp["consultas_sql"][0]["sql"]
    assert resp["iteracoes"] == 3
    assert "75%" in out["messages"][-1].content

    meta = out["messages"][-1].additional_kwargs["roteamento"]
    assert meta["consultas_sql"][0]["especialista"] == "credito"
    assert meta["dados"]["credito"]["linhas"] == [[0.75, 4]]

    # LLMOps enxerga as 3 chamadas do agente de BI dentro da etapa do especialista
    reg = montar_registro(cfg, rastreador=r, pergunta="q", resposta="a", estado=out)
    assert reg["custo_por_etapa"]["especialista_credito"]["tokens"] > 0
    assert sum(1 for c in r.chamadas if c.etapa == "especialista_credito") == 3
    assert reg["qtd_consultas_sql"] == 1


def test_agente_corrige_sql_depois_de_erro(cfg):
    fonte = FonteSQLite()
    llm_esp = fake(
        chamar("executar_sql", "c1", sql="SELECT coluna_que_nao_existe FROM main.credito.propostas"),
        chamar("executar_sql", "c2", sql="SELECT count(*) AS total FROM main.credito.propostas"),
        "Há 5 propostas no total.",
    )
    out = grafo(cfg, fake(rota("credito")), llm_esp, fonte).invoke({"messages": [HumanMessage("quantas propostas?")]})
    consultas = out["respostas"][0]["consultas_sql"]
    assert consultas[0]["erro"] and not consultas[1]["erro"]
    assert out["respostas"][0]["dados"]["linhas"] == [[5]]


def test_bloqueia_escrita_e_tabela_de_outro_dominio(cfg):
    fonte = FonteSQLite()
    llm_esp = fake(
        chamar("executar_sql", "c1", sql="DELETE FROM main.credito.propostas"),
        chamar("executar_sql", "c2", sql="SELECT * FROM main.cobranca.parcelas"),
        "Não posso fazer isso.",
    )
    out = grafo(cfg, fake(rota("credito")), llm_esp, fonte).invoke({"messages": [HumanMessage("apaga tudo")]})
    consultas = out["respostas"][0]["consultas_sql"]
    assert all(c["erro"].startswith("bloqueado") for c in consultas)
    assert fonte.executadas == []  # nada chegou ao banco


def test_limite_de_iteracoes_forca_resposta_final(cfg):
    cfg.bi.max_iteracoes = 2
    llm_esp = fake(
        chamar("descrever_tabela", "c1", tabela="main.credito.propostas"),
        chamar("descrever_tabela", "c2", tabela="main.credito.clientes"),
        "Resposta parcial com o que foi apurado.",
    )
    out = grafo(cfg, fake(rota("credito")), llm_esp).invoke({"messages": [HumanMessage("análise complexa")]})
    assert out["respostas"][0]["limite_atingido"] is True
    assert out["messages"][-1].content == "Resposta parcial com o que foi apurado."


def test_genie_como_ferramenta(cfg):
    cfg.especialista("cobranca").genie_space_id = "space123"
    llm_esp = fake(chamar("perguntar_genie", "c1", pergunta="valor em atraso por região"), "Norte lidera.")
    out = grafo(cfg, fake(rota("cobranca", reescrita="valor em atraso por região")), llm_esp).invoke(
        {"messages": [HumanMessage("atraso por região?")]}
    )
    assert "perguntar_genie" in llm_esp.ferramentas_vinculadas
    assert out["respostas"][0]["consultas_sql"][0]["via"] == "genie"


def test_especialista_so_documentacao_nao_usa_ferramentas(cfg):
    llm_esp = fake("Contrato social, cartão CNPJ... [manual_cadastro.pdf]")
    out = grafo(cfg, fake(rota("cadastro", reescrita="documentos conta PJ")), llm_esp).invoke(
        {"messages": [HumanMessage("documentos pj?")]}
    )
    assert llm_esp.ferramentas_vinculadas == []
    assert out["respostas"][0]["consultas_sql"] == []


def test_dois_agentes_bi_em_paralelo(cfg):
    llm_esp = fake(
        chamar("executar_sql", "a1", sql="SELECT count(*) AS n FROM main.credito.propostas"),
        chamar("executar_sql", "b1", sql="SELECT sum(valor_aberto) AS v FROM main.cobranca.parcelas WHERE dias_atraso > 30"),
        "resposta 1", "resposta 2", "consolidada",
    )
    out = grafo(cfg, fake(rota("credito", "cobranca")), llm_esp).invoke({"messages": [HumanMessage("x")]})
    assert len(out["respostas"]) == 2
    assert out["messages"][-1].content == "consolidada"
