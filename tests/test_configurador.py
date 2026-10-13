"""Assistente de configuração ponta a ponta (sem Databricks): escolhe IAs/Genie, cria agentes,
estuda as tabelas, revisa e gera o agents.yaml que o agente de produção consegue carregar."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("ipywidgets")

from langchain_core.messages import HumanMessage  # noqa: E402

from chat_especialistas import carregar_config, construir_grafo  # noqa: E402
from chat_especialistas.configurador.gerador import AgenteRascunho, Selecao, gerar_config, slug  # noqa: E402
from chat_especialistas.configurador.interface import Assistente  # noqa: E402
from chat_especialistas.publicacao import avaliar_roteamento, casos_de_teste  # noqa: E402
from fakes import FakeWorkspace, FonteSQLite, RetrieverFalso, chamar, fake  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
T_CRED = "main.credito.propostas"


def estudo_json(nome, tabela=None):
    d = {"descricao": f"Responde sobre {nome}.", "exemplos": [f"Pergunta típica de {nome}?"],
         "perguntas_teste": [f"Teste de {nome}?"], "instrucoes": f"Regras de {nome}."}
    if tabela:
        d["descricao_tabelas"] = {tabela: "1 linha = 1 proposta"}
        d["exemplos_sql"] = [{"pergunta": "Total", "sql": f"SELECT count(*) AS n FROM {tabela}"}]
    return json.dumps(d)


@pytest.fixture
def projeto(tmp_path):
    (tmp_path / "config").mkdir()
    shutil.copy(RAIZ / "config" / "agents.yaml", tmp_path / "config" / "agents.yaml")
    return tmp_path


def test_slug():
    assert slug("Especialista em Cobrança") == "cobranca"
    assert slug("Agente de Crédito PJ") == "credito_pj"
    assert slug("Vendas 2026!") == "vendas_2026"


def test_gerar_config_valida_escolhas():
    sel = Selecao(llm_roteador="m", llm_especialistas="m")
    with pytest.raises(ValueError, match="Warehouse"):
        gerar_config(sel, [AgenteRascunho(nome="A", tabelas=[T_CRED])])
    with pytest.raises(ValueError, match="embeddings"):
        gerar_config(sel, [AgenteRascunho(nome="A", documentos=True)])
    dados = gerar_config(sel, [AgenteRascunho(nome="Cobrança", genie_space_id="g1"), AgenteRascunho(nome="Cobrança", genie_space_id="g2")])
    assert [e["id"] for e in dados["especialistas"]] == ["cobranca", "cobranca_2"]


def test_assistente_ponta_a_ponta(projeto):
    publicados, avaliados = [], []
    llm_estudo = fake(estudo_json("crédito", T_CRED), estudo_json("cobrança"))
    a = Assistente(
        FakeWorkspace(), FonteSQLite(), criar_llm=lambda ep: llm_estudo, raiz=projeto,
        avaliar=lambda cfg: avaliados.append(cfg) or {"acuracia": 1.0},
        publicar=lambda cfg: publicados.append(cfg) or {"endpoint": "agents_main-chat"},
    )

    # ---- aba 1: IAs, warehouse, Genie
    assert a.dd_roteador.value == "databricks-meta-llama-3-3-70b-instruct"
    assert a.dd_warehouse.value == "wh1"
    a.sm_genies.value = ("g1",)
    a.cb_catalogo.value, a.txt_schema.value = "main", "chat_teste"
    a.fp_esp_in.value = 0.7

    # ---- aba 2: agente com tabelas (navegando no catálogo) e agente a partir do Genie
    a._novo_agente()
    a.txt_nome.value = "Especialista em Crédito"
    a.ta_prompt.value = "Propostas de crédito PJ e PF; taxa de aprovação = aprovadas / total."
    a.dd_cat.value = "main"
    a.dd_sch.value = "credito"
    assert a.sm_tab.options == (T_CRED,)
    a.sm_tab.value = (T_CRED,)
    a.btn_add_tab.click()
    a.btn_salvar_agente.click()
    assert "salvo" in a.msg_agente.value
    a.btn_agente_por_genie.click()
    assert [x.id for x in a.agentes] == ["credito", "cobranca"]
    assert a.agentes[1].genie_space_id == "g1"

    # ---- aba 3: estudar e revisar
    a.btn_estudar_todos.click()
    assert all(x.estudado for x in a.agentes)
    a.dd_revisar.value = "credito"
    assert "1 linha = 1 proposta" in a.ta_desc_tabelas.value
    assert "✅" in a.html_sql.value
    a.ta_instrucoes.value = "Taxa de aprovação = APROVADA / total. Sempre filtrar por data_proposta."
    a.ta_desc_tabelas.value = f"### {T_CRED}\nPropostas; status APROVADA/REPROVADA."
    a.btn_salvar_revisao.click()

    # ---- aba 4: gerar o agente geral
    a.txt_nome_assistente.value = "Assistente Financeiro"
    a.cb_indexar.value = False
    a.btn_gerar.click()
    assert len(publicados) == 1 and len(avaliados) == 1, a.out_gerar.outputs
    assert a.ultimo_resultado["publicacao"]["endpoint"] == "agents_main-chat"

    cfg = carregar_config(projeto / "config" / "agents.yaml")
    assert list((projeto / "config").glob("agents.yaml.*.bak"))  # backup do anterior
    assert cfg.agente_principal.nome == "Assistente Financeiro"
    assert cfg.projeto.nome_modelo == "main.chat_teste.chat_especialistas"
    assert cfg.llmops.preco("databricks-meta-llama-3-3-70b-instruct").input == 0.7
    cred, cob = cfg.especialista("credito"), cfg.especialista("cobranca")
    assert cred.dados.tabelas == [T_CRED] and cred.documentos is False
    assert cred.dados.instrucoes.startswith("Taxa de aprovação")
    assert cred.dados.descricao_tabelas[T_CRED] == "Propostas; status APROVADA/REPROVADA."
    assert cred.dados.exemplos_sql[0].sql.startswith("SELECT count(*)")
    assert "taxa de aprovação = aprovadas / total" in cred.prompt
    assert cob.genie_space_id == "g1" and cob.dados is None
    assert cfg.bi.warehouse_id == "wh1"
    assert casos_de_teste(cfg)[0] == ("Teste de crédito?", ["credito"])

    # ---- o estado é retomado ao reabrir o notebook
    b = Assistente(FakeWorkspace(), FonteSQLite(), criar_llm=lambda ep: None, raiz=projeto)
    assert [x.id for x in b.agentes] == ["credito", "cobranca"] and b.sm_genies.value == ("g1",)

    # ---- e o grafo de produção roda com essa configuração
    rota = json.dumps({"intencao": "especialista", "especialistas": ["credito"], "pergunta_reescrita": "Total de propostas", "confianca": 0.9})
    llm_esp = fake(chamar("executar_sql", "c1", sql=f"SELECT count(*) AS total FROM {T_CRED}"), "São 5 propostas.")
    g = construir_grafo(cfg, fake(rota), llm_esp, RetrieverFalso(), fonte_dados=FonteSQLite())
    out = g.invoke({"messages": [HumanMessage("quantas propostas?")]})
    assert out["respostas"][0]["dados"]["linhas"] == [[5]]
    assert "consultar_regras_negocio" not in llm_esp.ferramentas_vinculadas  # agente sem documentos


def test_avaliacao_do_roteamento(projeto):
    cfg = carregar_config(projeto / "config" / "agents.yaml")
    cfg.especialista("credito").perguntas_teste = ["taxa de aprovação?"]
    casos = [("taxa de aprovação?", ["credito"]), ("oi", [])]
    roteador = fake(
        json.dumps({"intencao": "especialista", "especialistas": ["credito"], "confianca": 0.9}),
        json.dumps({"intencao": "especialista", "especialistas": ["cobranca"], "confianca": 0.9}),
    )
    r = avaliar_roteamento(cfg, roteador, casos, log=lambda _: None)
    assert r["acertos"] == 1 and r["total"] == 2
