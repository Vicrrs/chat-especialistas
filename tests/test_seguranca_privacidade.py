"""Privacidade (LGPD), auditoria e credenciais.

- Com `llmops.registrar_textos: false`, nenhum texto do usuário (pergunta, pergunta reescrita, SQL com
  literais) vai para logs, traces, arquivo JSONL ou para a inference table via custom_outputs.
- Toda interação fica auditável: quem perguntou, em qual conversa, quais especialistas e consultas.
- Tokens e senhas nunca aparecem em mensagens de erro mostradas na app.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml
from langchain_core.messages import HumanMessage

from chat_especialistas import carregar_config, construir_grafo
from chat_especialistas.configurador.descoberta import mensagem_erro
from chat_especialistas.llmops import RastreadorUso, emitir, montar_registro
from chat_especialistas.seguranca import ocultar_segredos
from fakes import FonteSQLite, RetrieverFalso, chamar, fake

RAIZ = Path(__file__).resolve().parents[1]
CONFIG = RAIZ / "config" / "agents.yaml"
CPF = "123.456.789-09"
SQL_CPF = f"SELECT count(*) AS n FROM main.credito.propostas WHERE segmento = '{CPF}'"


def _cfg(tmp_path, registrar_textos: bool):
    d = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    d["llmops"].update(registrar_textos=registrar_textos, destinos=["log", "jsonl"],
                       arquivo_jsonl=str(tmp_path / "interacoes.jsonl"))
    caminho = tmp_path / "agents.yaml"
    caminho.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    return caminho


def _rota():
    return json.dumps({"intencao": "especialista", "especialistas": ["credito"],
                       "pergunta_reescrita": f"Propostas do cliente CPF {CPF}", "confianca": 0.9})


def _rodar(cfg):
    g = construir_grafo(cfg, fake(_rota()),
                        fake(chamar("executar_sql", "c1", sql=SQL_CPF, objetivo=f"propostas do CPF {CPF}"), "Há 0 propostas."),
                        RetrieverFalso(), fonte_dados=FonteSQLite(), nome_modelo_roteador="m", nome_modelo_especialista="m")
    rastreador = RastreadorUso(cfg.llmops)
    estado = g.invoke({"messages": [HumanMessage(f"quantas propostas tem o CPF {CPF}?")]}, config={"callbacks": [rastreador]})
    return montar_registro(cfg, rastreador=rastreador, pergunta=f"CPF {CPF}", resposta=estado["messages"][-1].content,
                           estado=estado, conversation_id="conv-1", user_id="ana@banco.com.br")


def test_sem_textos_nada_do_usuario_vai_para_logs_e_arquivos(tmp_path, capsys):
    cfg = carregar_config(_cfg(tmp_path, registrar_textos=False))
    registro = _rodar(cfg)
    emitir(cfg, registro)
    gravado = (tmp_path / "interacoes.jsonl").read_text(encoding="utf-8") + capsys.readouterr().out
    assert gravado.strip()
    assert CPF not in gravado and CPF not in json.dumps(registro, ensure_ascii=False)
    assert registro["pergunta"] is None and registro["resposta"] is None and registro["pergunta_reescrita"] is None


def test_com_textos_o_registro_e_completo(tmp_path):
    registro = _rodar(carregar_config(_cfg(tmp_path, registrar_textos=True)))
    assert CPF in registro["pergunta"] and CPF in registro["consultas_sql"][0]["sql"]


def test_trilha_de_auditoria(tmp_path):
    r = _rodar(carregar_config(_cfg(tmp_path, registrar_textos=False)))
    assert r["user_id"] == "ana@banco.com.br" and r["conversation_id"] == "conv-1"
    assert r["request_id"] and r["timestamp"] and r["versao_agente"]
    assert r["especialistas"] == ["credito"] and r["intencao"] == "especialista"
    assert r["qtd_consultas_sql"] == 1 and r["consultas_sql"][0]["especialista"] == "credito"


def test_endpoint_nao_devolve_texto_para_a_inference_table_quando_desligado(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    import chat_especialistas

    caminho = _cfg(tmp_path, registrar_textos=False)
    monkeypatch.setenv("CHAT_ESPECIALISTAS_CONFIG", str(caminho))
    monkeypatch.setattr(chat_especialistas, "construir_grafo_databricks", lambda cfg, checkpointer=None: construir_grafo(
        cfg, fake(_rota()), fake("Há 0 propostas."), RetrieverFalso(), fonte_dados=FonteSQLite()))
    monkeypatch.setattr(mlflow.models, "set_model", lambda m: None)
    monkeypatch.setattr(mlflow.langchain, "autolog", lambda *a, **k: None, raising=False)
    spec = importlib.util.spec_from_file_location("agent_privacidade", RAIZ / "agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from mlflow.types.responses import ResponsesAgentRequest

    resp = mod.AGENT.predict(ResponsesAgentRequest(input=[{"role": "user", "content": f"propostas do CPF {CPF}?"}]))
    assert CPF not in json.dumps(resp.custom_outputs, ensure_ascii=False, default=str)


# ------------------------------------------------------------- credenciais
# Credenciais FALSAS montadas em tempo de execução: no código não fica nenhum texto com cara de
# token de verdade (o push protection do GitHub e outros scanners bloqueariam o repositório).
TOKEN_FALSO = "dap" + "i" + "0123456789abcdef" * 2


@pytest.mark.parametrize("vazamento", [
    TOKEN_FALSO,
    TOKEN_FALSO + "-2",
    "Authorization: Bearer abcDEF123456789xyz",
    "ey" + "JhbGciOiJSUzI1NiJ9.ey" + "JzdWIiOiJhbmFAYmFuY28ifQ.c2lnbmF0dXJhLWZha2U",
    "password=S3nh@Forte!",
    "client_secret" + ": 'abc123def456'",
    "AK" + "IA" + "ABCDEFGHIJKLMNOP",
])
def test_credenciais_ocultadas(vazamento):
    texto = ocultar_segredos(f"Falha ao chamar a API ({vazamento}) no host https://ws")
    segredo = vazamento.split()[-1].split("=")[-1].split(": ")[-1].strip("'")
    assert segredo not in texto and "***" in texto and "https://ws" in texto


def test_mensagem_de_erro_da_app_nao_mostra_token():
    erro = Exception(f"401 Unauthorized. Request: GET /api/2.0/x headers {{'Authorization': 'Bearer {TOKEN_FALSO}'}}")
    assert TOKEN_FALSO[:20] not in mensagem_erro(erro)
