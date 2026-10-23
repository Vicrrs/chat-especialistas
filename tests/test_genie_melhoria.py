"""Especialistas como Genie Spaces: publicação, roteador acionando o Genie e melhoria contínua."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from chat_especialistas import carregar_config, construir_grafo
from chat_especialistas.configurador import melhoria
from chat_especialistas.configurador.estudo import EstudoAgente, perfilar_tabela
from chat_especialistas.configurador.genie_espaco import FIM_BLOCO, INICIO_BLOCO, montar_espaco, publicar_espaco
from chat_especialistas.configurador.gerador import AgenteRascunho, Selecao, conhecimento, gerar_config
from fakes import FakeGenie, FonteSQLite, RetrieverFalso, fake, mensagem_genie

T = "main.credito.propostas"
CONFIG = Path(__file__).resolve().parents[1] / "config" / "agents.yaml"
SQL_TOTAL = f"SELECT count(*) AS n FROM {T}"


def agente(**k) -> AgenteRascunho:
    e = EstudoAgente(
        nome="Crédito", descricao="Propostas de crédito.", exemplos=["Taxa de aprovação PJ?"],
        descricao_tabelas={T: "1 linha = 1 proposta"}, instrucoes="Aprovação = APROVADA / total.",
        exemplos_sql=[{"pergunta": "Total de propostas", "sql": SQL_TOTAL}],
    )
    e.perfis = [vars(perfilar_tabela(FonteSQLite(), T))]
    e.perfis[0]["colunas"] = [vars(c) for c in e.perfis[0]["colunas"]]
    base = dict(nome="Especialista em Crédito", prompt_negocio="Propostas PJ e PF.", tabelas=[T], estudo=e.to_dict())
    return AgenteRascunho(**{**base, **k})


class W:
    def __init__(self):
        self.genie = FakeGenie()


# ------------------------------------------------------------------ publicação
def test_espaco_tem_tabelas_instrucoes_exemplos_e_benchmark():
    a = agente(aprendizados=[{"tipo": "instrucao", "instrucao": "Ignore propostas CANCELADAS."},
                             {"tipo": "exemplo_sql", "pergunta": "Valor médio PJ", "sql": f"SELECT avg(valor) FROM {T}"}],
               benchmark=[{"pergunta": "Total de propostas", "sql": SQL_TOTAL}])
    s = montar_espaco(a)
    assert s["data_sources"]["tables"] == [{"identifier": T}]
    texto = "".join(s["instructions"]["text_instructions"][0]["content"])
    assert texto.startswith(INICIO_BLOCO) and FIM_BLOCO in texto
    assert "Propostas PJ e PF." in texto and "Ignore propostas CANCELADAS." in texto and "1 linha = 1 proposta" in texto
    perguntas = sorted(x["question"][0] for x in s["instructions"]["example_question_sqls"])
    assert perguntas == ["Total de propostas", "Valor médio PJ"]
    assert s["config"]["sample_questions"][0]["question"] == ["Taxa de aprovação PJ?"]
    b = s["benchmarks"]["questions"][0]
    assert b["question"] == ["Total de propostas"] and b["answer"][0]["format"] == "SQL"
    assert all(len(x["id"]) == 32 for x in s["instructions"]["example_question_sqls"])


def test_atualizar_mantem_o_que_foi_curado_no_genie():
    a = agente()
    atual = montar_espaco(a)
    curado = {"id": "f" * 32, "question": ["Pergunta curada no Genie"], "sql": ["SELECT 1"]}
    atual["instructions"]["example_question_sqls"].append(curado)
    texto = "".join(atual["instructions"]["text_instructions"][0]["content"])
    atual["instructions"]["text_instructions"][0]["content"] = [texto + "\n\nNota escrita no Genie."]
    atual["data_sources"]["tables"][0]["column_configs"] = [{"column_name": "status", "get_example_values": True}]
    id_exemplo = next(x["id"] for x in atual["instructions"]["example_question_sqls"] if x["question"] == ["Total de propostas"])

    a.aprendizados.append({"tipo": "instrucao", "instrucao": "Datas em data_proposta."})
    novo = montar_espaco(a, atual)
    texto_novo = "".join(novo["instructions"]["text_instructions"][0]["content"])
    assert "Datas em data_proposta." in texto_novo and "Nota escrita no Genie." in texto_novo
    assert texto_novo.count(INICIO_BLOCO) == 1
    exemplos = {x["question"][0]: x["id"] for x in novo["instructions"]["example_question_sqls"]}
    assert exemplos["Pergunta curada no Genie"] == "f" * 32
    assert exemplos["Total de propostas"] == id_exemplo  # mesmo item, não duplicado
    assert novo["data_sources"]["tables"][0]["column_configs"][0]["column_name"] == "status"


def test_publicar_cria_e_depois_atualiza_o_mesmo_espaco():
    w, a = W(), agente()
    r = publicar_espaco(w, a, "wh1")
    assert r == {"space_id": "genie_1", "criado": True} and a.genie_gerado
    assert w.genie.espacos["genie_1"]["title"] == "Especialista em Crédito"
    a.aprendizados.append({"tipo": "instrucao", "instrucao": "Nova regra."})
    assert publicar_espaco(w, a, "wh1") == {"space_id": "genie_1", "criado": False}
    assert "Nova regra." in w.genie.espacos["genie_1"]["serialized_space"]
    assert len(w.genie.espacos) == 1


def test_nao_altera_genie_escolhido_a_mao():
    with pytest.raises(ValueError, match="escolhido à mão"):
        publicar_espaco(W(), agente(genie_space_id="g1"), "wh1")


def test_config_manda_a_pergunta_direto_ao_genie_e_leva_os_aprendizados():
    a = agente(genie_space_id="genie_1", genie_gerado=True,
               aprendizados=[{"tipo": "instrucao", "instrucao": "Ignore CANCELADAS."}])
    sel = Selecao(llm_roteador="m", llm_especialistas="m", warehouse_id="wh1", modo_especialistas="genie")
    esp = gerar_config(sel, [a])["especialistas"][0]
    assert esp["genie_modo"] == "direto" and esp["genie_space_id"] == "genie_1"
    assert "Ignore CANCELADAS." in esp["dados"]["instrucoes"]
    sel.modo_especialistas = "bi"
    assert gerar_config(sel, [a])["especialistas"][0]["genie_modo"] == "ferramenta"


# ------------------------------------------------------- roteador -> Genie
def _rota():
    return json.dumps({"intencao": "especialista", "especialistas": ["credito"],
                       "pergunta_reescrita": "Total de propostas de crédito", "confianca": 0.9})


def _grafo(fonte, llm_esp):
    cfg = carregar_config(CONFIG)
    esp = cfg.especialista("credito")
    esp.genie_space_id, esp.genie_modo = "genie_1", "direto"
    return construir_grafo(cfg, fake(_rota()), llm_esp, RetrieverFalso(), fonte_dados=fonte,
                           nome_modelo_roteador="m", nome_modelo_especialista="m")


def test_agente_01_aciona_o_genie_do_especialista():
    fonte = FonteSQLite()
    fonte.perguntar_genie = lambda esp, p: {"texto": "São 5.", "sql": SQL_TOTAL, "colunas": ["n"], "linhas": [[5]],
                                            "conversa_id": "c1", "mensagem_id": "m1", "duracao_ms": 3}
    llm_esp = fake("Foram **5** propostas.")
    out = _grafo(fonte, llm_esp).invoke({"messages": [HumanMessage("quantas propostas?")]})
    r = out["respostas"][0]
    assert r["iteracoes"] == 1 and llm_esp.ferramentas_vinculadas == []  # sem ciclo de ferramentas
    c = r["consultas_sql"][0]
    assert c["via"] == "genie" and c["objetivo"] == "Total de propostas de crédito"
    assert (c["genie_space_id"], c["genie_conversa_id"], c["genie_mensagem_id"]) == ("genie_1", "c1", "m1")
    assert r["dados"]["linhas"] == [[5]]
    assert "5" in out["messages"][-1].content


def test_genie_falhou_o_agente_de_bi_assume():
    fonte = FonteSQLite()
    fonte.perguntar_genie = lambda esp, p: {"erro": "Genie indisponível"}
    llm_esp = fake("Respondi pelo agente de BI.")
    out = _grafo(fonte, llm_esp).invoke({"messages": [HumanMessage("quantas propostas?")]})
    assert "executar_sql" in llm_esp.ferramentas_vinculadas
    assert out["respostas"][0]["consultas_sql"][0]["erro"] == "Genie indisponível"


# ----------------------------------------------------------------- melhoria
def test_compara_resultados_sem_depender_de_nomes_e_ordem():
    assert melhoria.mesmos_resultados(["n"], [["5"]], ["total", "x"], [[5.0, "a"]])
    assert melhoria.mesmos_resultados(["s", "n"], [["A", 1], ["B", 2]], ["qtd", "seg"], [[2, "b"], [1, "a"]])
    assert not melhoria.mesmos_resultados(["n"], [[5]], ["n"], [[6]])
    assert not melhoria.mesmos_resultados(["n"], [[5]], ["n"], [[5], [5]])


def test_casos_do_genie_le_so_feedback_negativo_e_erros_novos():
    w = W()
    w.genie.conversas["genie_1"] = {"c1": [
        mensagem_genie("velha ruim", 100, rating="NEGATIVE"),
        mensagem_genie("taxa de aprovação PJ", 200, rating="NEGATIVE", comentario="contou canceladas", sql="SELECT 1"),
        mensagem_genie("boa", 210, rating="POSITIVE"),
        mensagem_genie("quebrou", 220, erro="column not found"),
    ]}
    casos, lido = melhoria.casos_do_genie(w, "genie_1", "credito", desde_ms=150)
    assert lido == 220
    assert [(c.origem, c.pergunta) for c in casos] == [("genie_feedback", "taxa de aprovação PJ"), ("genie_erro", "quebrou")]
    assert "contou canceladas" in casos[0].detalhe and casos[0].sql_obtido == "SELECT 1"


def test_benchmark_nativo_vira_historico_e_casos():
    w, a = W(), agente(genie_space_id="genie_1", genie_gerado=True)
    w.genie.avaliacao = [("Total de propostas", "GOOD", [], SQL_TOTAL, SQL_TOTAL),
                         ("Taxa PJ", "BAD", ["LLM_JUDGE_MISSING_OR_INCORRECT_FILTER"], "SELECT 0.75", "SELECT 0.8")]
    r = melhoria.rodar_benchmark(w, a, "wh1", log=lambda _: None)
    assert (r["via"], r["acertos"], r["total"], r["acuracia"]) == ("genie", 1, 2, 0.5)
    casos = melhoria.casos_do_benchmark(r)
    assert len(casos) == 1 and casos[0].sql_esperado == "SELECT 0.75" and "INCORRECT_FILTER" in casos[0].detalhe


def test_benchmark_local_quando_avaliacao_nativa_nao_existe(monkeypatch):
    from chat_especialistas.dados import ResultadoSQL

    w = W()
    a = agente(genie_space_id="genie_1", genie_gerado=True,
               benchmark=[{"pergunta": "Total", "sql": SQL_TOTAL}, {"pergunta": "PJ", "sql": "SELECT 4"}])
    respostas = {"Total": {"sql": "x", "colunas": ["n"], "linhas": [["5"]]}, "PJ": {"sql": "y", "colunas": ["n"], "linhas": [["3"]]}}
    monkeypatch.setattr(melhoria, "perguntar_genie_space", lambda w, sid, p, max_linhas: respostas[p])
    fonte = FonteSQLite()
    monkeypatch.setattr(melhoria, "executar_sql_warehouse",
                        lambda w, wh, sql, *a: ResultadoSQL(["n"], [[4]]) if sql == "SELECT 4" else fonte.consultar(sql))
    r = melhoria.rodar_benchmark(w, a, "wh1", log=lambda _: None)
    assert r["via"] == "local" and [i["ok"] for i in r["itens"]] == [True, False]


def test_proposta_valida_o_sql_e_aprovada_vira_aprendizado():
    a = agente(genie_space_id="genie_1", genie_gerado=True)
    llm = fake(json.dumps({"diagnostico": "Faltou filtrar PJ.", "instrucao": "PJ = segmento 'PJ'.",
                           "sql": f"SELECT count(*) AS n FROM {T} WHERE segmento = 'PJ'"}))
    caso = vars(melhoria.caso_do_chat("credito", "Quantas propostas PJ?", comentario="veio PF junto"))
    p = melhoria.propor_melhoria(llm, FonteSQLite(), a, caso)
    assert p.validado and "segmento = 'PJ'" in p.sql
    assert p.instrucao == "PJ = segmento 'PJ'."

    d = vars(p)
    melhoria.aprovar(a, d, instrucao=p.instrucao, sql=p.sql)
    assert d["status"] == "aprovada"
    e = conhecimento(a)
    assert "PJ = segmento 'PJ'." in e.instrucoes
    assert any(x["pergunta"] == "Quantas propostas PJ?" for x in e.exemplos_sql)
    assert a.benchmark[-1]["pergunta"] == "Quantas propostas PJ?"
    # um novo estudo troca o estudo, mas o aprendizado continua
    a.estudo = EstudoAgente(instrucoes="estudo novo").to_dict()
    assert "PJ = segmento 'PJ'." in conhecimento(a).instrucoes


def test_proposta_com_sql_quebrado_fica_marcada():
    a = agente()
    llm = fake(json.dumps({"diagnostico": "x", "instrucao": "", "sql": f"SELECT coluna_que_nao_existe FROM {T}"}),
               f"SELECT outra_que_nao_existe FROM {T}")
    p = melhoria.propor_melhoria(llm, FonteSQLite(), a, vars(melhoria.caso_do_chat("credito", "?")))
    assert not p.validado and p.erro


def test_detecta_mudancas_nas_tabelas():
    a = agente()
    fonte = FonteSQLite()
    assert melhoria.detectar_mudancas(fonte, a) == []
    fonte.con.execute("ALTER TABLE credito__propostas ADD COLUMN canal TEXT")
    assert melhoria.detectar_mudancas(fonte, a) == [f"{T}: colunas novas canal"]


def test_ciclo_completo_e_estado_sem_duplicar_casos():
    w = W()
    a = agente(genie_space_id="genie_1", genie_gerado=True)
    w.genie.conversas["genie_1"] = {"c1": [mensagem_genie("Quantas PJ?", 10, rating="NEGATIVE")]}
    est = melhoria.EstadoMelhoria()
    llm = fake(*[json.dumps({"diagnostico": "d", "instrucao": "i", "sql": SQL_TOTAL})] * 2)
    resumo = melhoria.ciclo_melhoria(w, llm, FonteSQLite(), [a, agente(nome="Sem Genie")], est, "wh1", log=lambda _: None)
    assert list(resumo) == [a.id] and resumo[a.id]["propostas"] == 1
    assert len(est.pendentes()) == 1 and est.casos == [] and est.lido_genie_ms == {a.id: 10}
    # a mesma pergunta não vira outro caso enquanto há proposta pendente
    assert est.adicionar_casos([melhoria.caso_do_chat(a.id, "quantas  pj?")]) == 0
    assert melhoria.EstadoMelhoria.from_json(est.to_json()).propostas == est.propostas
