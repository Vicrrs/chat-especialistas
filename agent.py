"""Ponto de entrada do agente para MLflow (models-from-code) e Databricks Model Serving.

Este arquivo é registrado no MLflow pelo notebook 02 e servido como um
ResponsesAgent, compatível com o Review App, AI Playground e Agent Evaluation.
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, List

import mlflow
from langchain_core.messages import AIMessage, HumanMessage
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

# Permite importar o pacote tanto no repo quanto empacotado pelo MLflow (code_paths).
_AQUI = Path(__file__).resolve().parent
for p in (_AQUI / "src", _AQUI):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from chat_especialistas import carregar_config, construir_grafo_databricks  # noqa: E402
from chat_especialistas.llmops import RastreadorUso, emitir, montar_registro  # noqa: E402

mlflow.langchain.autolog()


def _texto_conteudo(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            parte.get("text", "") for parte in content if isinstance(parte, dict)
        )
    return str(content or "")


def _para_mensagens_langchain(itens: List[Dict[str, Any]]):
    msgs = []
    for item in itens:
        role = item.get("role")
        if role == "user":
            msgs.append(HumanMessage(_texto_conteudo(item.get("content"))))
        elif role == "assistant":
            msgs.append(AIMessage(_texto_conteudo(item.get("content"))))
        # system/developer e itens de ferramenta são ignorados: o prompt do
        # Agente 01 vem da configuração.
    return msgs


class ChatEspecialistasAgent(ResponsesAgent):
    def __init__(self, caminho_config: str | None = None):
        self.cfg = carregar_config(caminho_config or os.environ.get("CHAT_ESPECIALISTAS_CONFIG"))
        self.grafo = construir_grafo_databricks(self.cfg)

    @staticmethod
    def _identificacao(request: ResponsesAgentRequest):
        """conversation_id / user_id vindos de `context` ou `custom_inputs`."""
        ctx = getattr(request, "context", None)
        extras = getattr(request, "custom_inputs", None) or {}
        conversa = getattr(ctx, "conversation_id", None) or extras.get("conversation_id")
        usuario = getattr(ctx, "user_id", None) or extras.get("user_id")
        return conversa, usuario

    @mlflow.trace(name="chat_especialistas", span_type="AGENT")
    def _responder(self, request: ResponsesAgentRequest):
        inicio = time.perf_counter()
        itens = [i.model_dump() if hasattr(i, "model_dump") else dict(i) for i in request.input]
        mensagens = _para_mensagens_langchain(itens)
        pergunta = next(
            (_texto_conteudo(m.content) for m in reversed(mensagens) if isinstance(m, HumanMessage)), ""
        )
        conversa, usuario = self._identificacao(request)
        rastreador = RastreadorUso(self.cfg.llmops)

        estado: Dict[str, Any] = {}
        texto_resposta, fontes, roteamento, erro = "", [], {}, None
        try:
            estado = self.grafo.invoke({"messages": mensagens}, config={"callbacks": [rastreador]})
            resposta = estado["messages"][-1]
            texto_resposta = _texto_conteudo(resposta.content)
            if isinstance(resposta, AIMessage):
                roteamento = resposta.additional_kwargs.get("roteamento", {})
                fontes = roteamento.get("fontes", [])
            # Cada especialista com documentação fez 1 busca no Vector Search (embedding da pergunta).
            for eid in estado.get("especialistas", []):
                if not self.cfg.especialista(eid).documentos:
                    continue
                rastreador.registrar_embedding(
                    f"busca_{eid}", self.cfg.vector_search.modelo_embedding,
                    estado.get("pergunta_reescrita", pergunta),
                )
        except Exception as exc:
            erro = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            registro = montar_registro(
                self.cfg,
                rastreador=rastreador,
                pergunta=pergunta,
                resposta=texto_resposta,
                estado=estado,
                fontes=fontes,
                latencia_ms=int((time.perf_counter() - inicio) * 1000),
                conversation_id=conversa,
                user_id=usuario,
                erro=erro,
            )
            emitir(self.cfg, registro)

        item = self.create_text_output_item(text=texto_resposta, id=str(uuid.uuid4()))
        # custom_outputs vai para a inference table do endpoint -> base do painel LLMOps.
        custom = {
            "intencao": estado.get("intencao"),
            "especialistas": estado.get("especialistas", []),
            # custom_outputs também vai para a inference table: respeita registrar_textos (LGPD).
            "pergunta_reescrita": estado.get("pergunta_reescrita") if self.cfg.llmops.registrar_textos else None,
            "confianca": estado.get("confianca"),
            "fontes": fontes,
            # BI: SQL executado e o último resultado de cada especialista (para tabela/gráfico no front-end)
            "consultas_sql": roteamento.get("consultas_sql", []),
            "dados": roteamento.get("dados", {}),
            "llmops": {k: v for k, v in registro.items() if k not in ("pergunta", "resposta", "consultas_sql")},
        }
        return item, custom

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        item, custom = self._responder(request)
        return ResponsesAgentResponse(output=[item], custom_outputs=custom)

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        item, custom = self._responder(request)
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done", item=item, custom_outputs=custom
        )


AGENT = ChatEspecialistasAgent()
mlflow.models.set_model(AGENT)
