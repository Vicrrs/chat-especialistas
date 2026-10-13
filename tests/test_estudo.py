"""Estudo das tabelas: perfil real (SQLite) + conhecimento gerado pelo LLM + validação dos SQLs."""
from __future__ import annotations

import json

from chat_especialistas.configurador.estudo import estudar_agente, perfil_texto, perfilar_tabela
from fakes import FonteSQLite, fake

T = "main.credito.propostas"


def test_perfil_le_estatisticas_reais():
    p = perfilar_tabela(FonteSQLite(), T)
    assert p.total_linhas == 5 and not p.erros
    col = {c.nome: c for c in p.colunas}
    assert col["status"].distintos == 2
    assert dict((v, n) for v, n in col["status"].top_valores) == {"APROVADA": 4, "REPROVADA": 1}
    assert col["valor"].minimo == 10000 and col["valor"].maximo == 120000
    assert col["data_proposta"].minimo == "2026-07-10"
    assert len(p.amostra) == 5
    texto = perfil_texto(p)
    assert "APROVADA (4)" in texto and "Linhas: 5" in texto


def test_tabela_inexistente_vira_aviso():
    p = perfilar_tabela(FonteSQLite(), "main.credito.nao_existe")
    assert p.erros and p.colunas == []


def test_estudo_gera_conhecimento_e_so_mantem_sql_que_roda():
    resposta = {
        "nome": "Crédito",
        "descricao": "Responde sobre propostas de crédito, aprovação e valores.",
        "exemplos": ["Qual a taxa de aprovação PJ?", "Quanto foi solicitado em setembro?"],
        "perguntas_teste": ["Quantas propostas reprovadas?", "Valor médio aprovado PF?"],
        "descricao_tabelas": {T: "1 linha = 1 proposta. status: APROVADA/REPROVADA."},
        "instrucoes": "Taxa de aprovação = APROVADA / total.",
        "exemplos_sql": [
            {"pergunta": "Total por status", "sql": f"SELECT status, count(*) AS n FROM {T} GROUP BY status"},
            {"pergunta": "Valor médio", "sql": f"SELECT avg(valor_errado) FROM {T}"},       # erra, é corrigido
            {"pergunta": "Apagar", "sql": f"DELETE FROM {T}"},                               # bloqueado 2x
        ],
    }
    llm = fake(
        "```json\n" + json.dumps(resposta) + "\n```",
        f"```sql\nSELECT avg(valor) AS media FROM {T}\n```",  # correção do 2º
        f"DELETE FROM {T}",                                     # "correção" do 3º continua inválida
    )
    fonte = FonteSQLite()
    e = estudar_agente(llm, fonte, nome="Especialista em Crédito", prompt_negocio="Propostas de crédito",
                       tabelas=[T], log=lambda _: None)

    assert e.nome == "Especialista em Crédito"
    assert e.descricao.startswith("Responde sobre propostas")
    assert e.descricao_tabelas[T].startswith("1 linha = 1 proposta")
    assert [x["pergunta"] for x in e.exemplos_sql] == ["Total por status", "Valor médio"]
    assert "avg(valor)" in e.exemplos_sql[1]["sql"]
    ok = {v["pergunta"]: v for v in e.validacao_sql}
    assert ok["Valor médio"]["corrigido"] and ok["Valor médio"]["ok"]
    assert not ok["Apagar"]["ok"] and ok["Apagar"]["erro"].startswith("bloqueado")
    assert not any("DELETE" in s.upper() for s in fonte.executadas)


def test_estudo_de_agente_so_genie():
    llm = fake(json.dumps({"descricao": "Inadimplência via Genie", "exemplos": ["Atraso por região?"],
                           "perguntas_teste": ["Inadimplência no Norte?"]}))
    e = estudar_agente(llm, FonteSQLite(), nome="Cobrança", prompt_negocio="", tabelas=[],
                       genie_space_id="g1", log=lambda _: None)
    assert e.descricao == "Inadimplência via Genie" and e.exemplos_sql == [] and not e.avisos


def test_json_invalido_nao_quebra():
    e = estudar_agente(fake("não sei", "ainda não sei"), FonteSQLite(), nome="X", prompt_negocio="Regras de X",
                       tabelas=[T], log=lambda _: None)
    assert e.descricao == "Regras de X"
    assert any("JSON" in a for a in e.avisos)
