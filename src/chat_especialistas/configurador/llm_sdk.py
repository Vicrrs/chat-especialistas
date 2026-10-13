"""LLM de chat via Databricks SDK (serving_endpoints.query), com a identidade do cliente informado.

Na app Streamlit o cliente é criado com o token do usuário logado, então o estudo usa as
permissões dele. Tem a mesma interface mínima que o estudo usa: `invoke(mensagens) -> AIMessage`.
"""
from __future__ import annotations

from typing import List

from langchain_core.messages import AIMessage, BaseMessage


class ChatServing:
    def __init__(self, w, endpoint: str, temperature: float = 0.0, max_tokens: int = 4000):
        self.w, self.endpoint, self.temperature, self.max_tokens = w, endpoint, temperature, max_tokens

    def invoke(self, mensagens: List[BaseMessage], **_) -> AIMessage:
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        papeis = {"system": ChatMessageRole.SYSTEM, "human": ChatMessageRole.USER, "ai": ChatMessageRole.ASSISTANT}
        msgs = [
            ChatMessage(role=papeis.get(m.type, ChatMessageRole.USER), content=m.content if isinstance(m.content, str) else str(m.content))
            for m in mensagens
        ]
        r = self.w.serving_endpoints.query(
            self.endpoint, messages=msgs, temperature=self.temperature, max_tokens=self.max_tokens
        )
        conteudo = r.choices[0].message.content if r.choices else ""
        if isinstance(conteudo, list):  # alguns modelos devolvem blocos
            conteudo = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in conteudo)
        return AIMessage(conteudo or "")
