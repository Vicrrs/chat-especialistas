"""App Streamlit ponta a ponta com serviços simulados (AppTest roda o app.py de verdade)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from chat_especialistas.app_servicos import Servicos, extrair_resposta  # noqa: E402
from chat_especialistas.config import AppConfig  # noqa: E402
from fakes import FakeWorkspace, FonteSQLite, fake  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")
T = "main.credito.propostas"


class FakeServicos(Servicos):
    def __init__(self, respostas_llm):
        super().__init__(FakeWorkspace(), "usuario", usuario="ana@empresa.com")
        self.arquivos, self.jobs = {}, []
        self._llm = fake(*respostas_llm)

    def fonte_estudo(self, warehouse_id):
        return FonteSQLite()

    def llm(self, endpoint):
        return self._llm

    def preparar_projeto(self, catalogo, schema, warehouse_id):
        self.preparado = (catalogo, schema, warehouse_id)
        return []

    def ler(self, caminho):
        return self.arquivos.get(caminho)

    def gravar(self, caminho, texto):
        self.arquivos[caminho] = texto

    def contar_documentos(self, *a):
        return 0

    def iniciar_publicacao(self, parametros):
        self.jobs.append(parametros)
        return {"run_id": 42, "url": "https://ws/run/42", "como": "usuario"}

    def status_publicacao(self, run_id, como="usuario"):
        return {"terminado": True, "sucesso": True, "estado": "TERMINATED", "resultado": "SUCCESS", "retorno": {
            "avaliacao": {"acuracia": 0.75, "acertos": 3, "total": 4,
                          "erros": [{"pergunta": "x?", "esperado": ["credito"], "obtido": [], "intencao": "esclarecer"}]},
            "publicacao": {"endpoint": "agents_main-chat", "nome_modelo": "main.chat_especialistas.chat_especialistas",
                           "versao": "1", "review_app": "https://ws/review"},
        }}

    def conversar(self, endpoint, mensagens, conversation_id):
        return {"output": [{"type": "message", "content": [{"type": "output_text", "text": "São **5** propostas."}]}],
                "custom_outputs": {"intencao": "especialista", "especialistas": ["credito"],
                                   "consultas_sql": [{"sql": f"SELECT count(*) FROM {T}"}],
                                   "dados": {"credito": {"colunas": ["n"], "linhas": [[5]]}},
                                   "llmops": {"tokens_total": 900, "custo_usd": 0.0012}}}


def estudo_json():
    return json.dumps({"descricao": "Propostas de crédito e aprovação.", "exemplos": ["Taxa de aprovação PJ?"],
                       "perguntas_teste": ["Quantas propostas reprovadas?"],
                       "descricao_tabelas": {T: "1 linha = 1 proposta"}, "instrucoes": "Aprovação = APROVADA/total",
                       "exemplos_sql": [{"pergunta": "Total", "sql": f"SELECT count(*) AS n FROM {T}"}]})


def botao(at, texto):
    return next(b for b in at.button if texto in b.label)


def test_app_fluxo_completo():
    sv = FakeServicos([estudo_json()])
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["servicos"] = sv
    at.run()
    assert not at.exception, at.exception
    assert "Conectado como **ana@empresa.com**" in at.sidebar.success[0].value
    assert "**2** LLMs de chat" in at.markdown[0].value

    # ---- aba 1: escolhas
    at.multiselect[0].set_value(["g1"])  # Genies liberados
    botao(at, "Salvar escolhas").click().run()
    assert not at.exception, at.exception
    assert sv.preparado == ("main", "chat_especialistas", "wh1")
    estado = json.loads(sv.arquivos["/Volumes/main/chat_especialistas/config/assistente_estado.json"])
    assert estado["selecao"]["llm_especialistas"] == "databricks-meta-llama-3-3-70b-instruct"
    assert estado["selecao"]["genies"] == ["g1"]

    # ---- aba 2: agente com tabela escolhida navegando no catálogo
    at.text_input(key="ed_nome").set_value("Especialista em Crédito")
    at.text_area(key="ed_prompt").set_value("Propostas de crédito; aprovação = APROVADA / total.")
    at.selectbox(key="pick_cat").set_value("main").run()
    at.selectbox(key="pick_sch").set_value("credito").run()
    at.multiselect(key="pick_tabs").set_value([T]).run()
    botao(at, "Adicionar ao agente").click().run()
    assert at.multiselect(key="ed_tabelas").value == [T]
    botao(at, "Salvar agente").click().run()
    assert not at.exception, at.exception
    assert any("salvo" in s.value for s in at.success)
    botao(at, "Criar um agente para cada Genie").click().run()
    assert [a.id for a in at.session_state["agentes"]] == ["credito", "cobranca"]

    # ---- aba 3: estudar (o agente do Genie recebe a 2ª resposta do LLM)
    sv._llm = fake(estudo_json(), json.dumps({"descricao": "Inadimplência via Genie", "perguntas_teste": ["Atraso no Norte?"]}))
    botao(at, "Estudar agentes pendentes").click().run()
    assert not at.exception, at.exception
    assert all(a.estudado for a in at.session_state["agentes"])
    assert any("SQLs de exemplo" in m.value for m in at.markdown)

    # ---- aba 4: gerar agente geral -> yaml no Volume + job de publicação
    botao(at, "Gerar agente geral").click().run()
    assert not at.exception, at.exception
    cfg = AppConfig.model_validate(yaml.safe_load(sv.arquivos["/Volumes/main/chat_especialistas/config/agents.yaml"]))
    assert [e.id for e in cfg.especialistas] == ["credito", "cobranca"]
    assert cfg.especialista("credito").dados.exemplos_sql[0].sql.startswith("SELECT count(*)")
    assert cfg.especialista("cobranca").genie_space_id == "g1"
    assert sv.jobs[0]["config_path"] == "/Volumes/main/chat_especialistas/config/agents.yaml"
    assert any("Agente geral pronto" in s.value for s in at.success)
    assert at.session_state["extra"]["endpoint"] == "agents_main-chat"

    # ---- aba 5: conversar com o endpoint publicado
    assert next(t for t in at.text_input if t.label == "Endpoint do agente").value == "agents_main-chat"
    at.chat_input[0].set_value("quantas propostas?").run()
    assert not at.exception, at.exception
    assert at.session_state["chat"][-1]["content"] == "São **5** propostas."
    assert at.session_state["gasto_chat"]["tokens"] == 900

    # ---- reabrir a app recupera tudo do Volume
    at2 = AppTest.from_file(APP, default_timeout=30)
    sv2 = FakeServicos([])
    sv2.arquivos = sv.arquivos
    at2.session_state["servicos"] = sv2
    at2.run()
    assert not at2.exception, at2.exception
    assert [a.id for a in at2.session_state["agentes"]] == ["credito", "cobranca"]


def test_app_avisa_quando_sem_user_authorization():
    sv = FakeServicos([])
    sv.identidade = "app"
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["servicos"] = sv
    at.run()
    assert "própria conta de serviço" in at.sidebar.warning[0].value


def test_extrair_resposta_sem_custom_outputs():
    texto, d = extrair_resposta({"output": [{"content": [{"text": "oi"}]}]})
    assert texto == "oi" and d["tokens"] == 0


def test_plano_b_lista_com_conta_de_servico_quando_token_falha():
    from types import SimpleNamespace as NS

    def falha(*a, **k):
        raise Exception("unable to parse response. Request log:```GET /api/2.0/x\n< 401 Unauthorized\n< <html>```")

    w_usuario = NS(serving_endpoints=NS(list=falha), genie=NS(list_spaces=falha), warehouses=NS(list=falha))
    sv = Servicos(w_usuario, "usuario", w_app=FakeWorkspace())
    r = sv.recursos()
    assert [m.nome for m in r.llms_chat] and r.genies and r.warehouses
    assert any("HTTP 401 Unauthorized" in a for a in r.avisos)
    assert any("conta de serviço da app" in a for a in r.avisos)
