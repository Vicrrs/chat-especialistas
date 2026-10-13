"""Grafo LangGraph: Agente 01 (contexto + detector de intenção) -> especialistas.

    START
      │
      ▼
  agente_principal  ── detecta intenção, reescreve a pergunta, escolhe especialista(s)
      │
      ├──► especialista_<id> (1 ou 2 em paralelo; agente de BI: regras + SQL) ──► consolidar ──► END
      ├──► esclarecer       (pergunta ambígua / confiança baixa)             ──► END
      ├──► conversa_geral   (saudação, agradecimento, "o que você faz?")     ──► END
      └──► fora_do_escopo   (assunto que nenhum especialista cobre)          ──► END
"""
from __future__ import annotations

import json
import logging
import re
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send

from . import prompts
from .config import AppConfig
from .bi_agent import executar_especialista
from .dados import FonteDados
from .retriever import Retriever

log = logging.getLogger(__name__)

INTENCOES = {"especialista", "esclarecer", "conversa_geral", "fora_do_escopo"}
MAX_ESPECIALISTAS = 2


# --------------------------------------------------------------------- estado
def _acumular(atual: Optional[list], novo: Optional[list]) -> list:
    """Reducer de `respostas`: None zera (início de cada turno), lista acumula."""
    if novo is None:
        return []
    return (atual or []) + novo


class EstadoChat(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]
    intencao: str
    especialistas: List[str]
    pergunta_reescrita: str
    pergunta_esclarecimento: str
    confianca: float
    justificativa: str
    respostas: Annotated[List[Dict[str, Any]], _acumular]


class EntradaEspecialista(TypedDict):
    especialista_id: str
    pergunta: str
    messages: List[AnyMessage]


# ------------------------------------------------------------------- helpers
def _janela(messages: List[AnyMessage], n: int) -> List[AnyMessage]:
    msgs = [m for m in messages if not isinstance(m, SystemMessage)]
    return msgs[-n:]


def _texto(msg: AnyMessage) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)


def _extrair_json(texto: str) -> Dict[str, Any]:
    texto = re.sub(r"```(?:json)?", "", texto)
    inicio, fim = texto.find("{"), texto.rfind("}")
    if inicio == -1 or fim == -1:
        raise ValueError("resposta do roteador sem JSON")
    return json.loads(texto[inicio : fim + 1])


def _ultima_pergunta(messages: List[AnyMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return _texto(m)
    return ""


# ------------------------------------------------------- detector de intenção
def detectar_intencao(
    cfg: AppConfig,
    llm_roteador: BaseChatModel,
    historico: List[AnyMessage],
    sys_roteador: Optional[str] = None,
) -> Dict[str, Any]:
    """Agente 01: lê o histórico e decide a intenção e o(s) especialista(s).

    Também é usado sozinho para AVALIAR o roteamento antes do deploy (barato: só o roteador).
    """
    ids_validos = {e.id for e in cfg.especialistas}
    pergunta_original = _ultima_pergunta(historico)
    resposta = llm_roteador.invoke([SystemMessage(sys_roteador or prompts.prompt_roteador(cfg)), *historico])

    try:
        d = _extrair_json(_texto(resposta))
    except Exception:  # noqa: BLE001
        log.warning("Roteador devolveu saída inválida: %s", _texto(resposta)[:500])
        d = {"intencao": "esclarecer", "confianca": 0.0}

    intencao = d.get("intencao") if d.get("intencao") in INTENCOES else "esclarecer"
    especialistas = [e for e in (d.get("especialistas") or []) if e in ids_validos]
    especialistas = list(dict.fromkeys(especialistas))[:MAX_ESPECIALISTAS]
    try:
        confianca = float(d.get("confianca", 0.0))
    except (TypeError, ValueError):
        confianca = 0.0

    if intencao == "especialista" and not especialistas:
        intencao = "esclarecer"
    if intencao == "especialista" and confianca < cfg.roteamento.limiar_confianca:
        intencao = "esclarecer"

    esclarecimento = (d.get("pergunta_esclarecimento") or "").strip()
    if intencao == "esclarecer" and not esclarecimento:
        esclarecimento = (
            "Para eu te direcionar ao especialista certo, pode detalhar um pouco mais "
            "sua dúvida? Posso ajudar com:\n" + prompts.catalogo_amigavel(cfg)
        )

    return {
        "intencao": intencao,
        "especialistas": especialistas if intencao == "especialista" else [],
        "pergunta_reescrita": (d.get("pergunta_reescrita") or pergunta_original).strip(),
        "pergunta_esclarecimento": esclarecimento,
        "confianca": confianca,
        "justificativa": d.get("justificativa", ""),
    }


# --------------------------------------------------------------------- grafo
def construir_grafo(
    cfg: AppConfig,
    llm_roteador: BaseChatModel,
    llm_especialista: BaseChatModel,
    retriever: Retriever,
    checkpointer=None,
    nome_modelo_roteador: Optional[str] = None,
    nome_modelo_especialista: Optional[str] = None,
    fonte_dados: Optional[FonteDados] = None,
):
    # Etiqueta cada LLM com o nome do endpoint para o LLMOps calcular o custo certo.
    from .llmops import CHAVE_MODELO

    if nome_modelo_roteador:
        llm_roteador = llm_roteador.with_config(metadata={CHAVE_MODELO: nome_modelo_roteador})
    meta_especialista = {CHAVE_MODELO: nome_modelo_especialista} if nome_modelo_especialista else None
    llm_especialista_base = llm_especialista  # agentes de BI fazem bind_tools sobre o modelo cru
    if meta_especialista:
        llm_especialista = llm_especialista.with_config(metadata=meta_especialista)

    for e in cfg.especialistas:
        if e.eh_bi and fonte_dados is None:
            raise ValueError(f"Especialista '{e.id}' tem dados/Genie mas nenhuma fonte_dados foi informada.")

    janela = cfg.roteamento.janela_historico
    sys_roteador = prompts.prompt_roteador(cfg)

    # ---- Agente 01: entende o contexto e detecta a intenção
    def agente_principal(state: EstadoChat) -> Dict[str, Any]:
        historico = _janela(state["messages"], janela)
        return {**detectar_intencao(cfg, llm_roteador, historico, sys_roteador), "respostas": None}

    # ---- Detector de intenção (aresta condicional)
    def detector_de_intencao(state: EstadoChat):
        intencao = state["intencao"]
        if intencao == "especialista":
            historico = _janela(state["messages"], janela)
            return [
                Send(
                    f"especialista_{eid}",
                    {
                        "especialista_id": eid,
                        "pergunta": state["pergunta_reescrita"],
                        "messages": historico,
                    },
                )
                for eid in state["especialistas"]
            ]
        return intencao

    # ---- Especialistas (um nó por regra de negócio)
    def criar_no_especialista(eid: str):
        esp = cfg.especialista(eid)

        def especialista(entrada: EntradaEspecialista) -> Dict[str, Any]:
            resultado = executar_especialista(
                cfg, esp, llm_especialista_base, retriever, fonte_dados,
                entrada["pergunta"], entrada["messages"], metadata=meta_especialista,
            )
            return {"respostas": [resultado]}

        especialista.__name__ = f"especialista_{eid}"
        return especialista

    def consolidar(state: EstadoChat) -> Dict[str, Any]:
        respostas = state.get("respostas") or []
        meta = {
            "especialistas": [r["especialista"] for r in respostas],
            "fontes": sorted({f for r in respostas for f in r["fontes"]}),
            "consultas_sql": [
                {"especialista": r["especialista"], **c} for r in respostas for c in r.get("consultas_sql", [])
            ],
            "dados": {r["especialista"]: r["dados"] for r in respostas if r.get("dados")},
        }
        if len(respostas) == 1:
            texto = respostas[0]["texto"]
        else:
            blocos = "\n\n".join(f"### {r['nome']}\n{r['texto']}" for r in respostas)
            sistema = prompts.CONSOLIDADOR_TEMPLATE.format(
                prompt_principal=cfg.agente_principal.prompt.strip(),
                pergunta=state["pergunta_reescrita"],
                respostas=blocos,
            )
            texto = _texto(llm_especialista.invoke([SystemMessage(sistema)]))
        return {"messages": [AIMessage(texto, additional_kwargs={"roteamento": meta})]}

    def esclarecer(state: EstadoChat) -> Dict[str, Any]:
        return {"messages": [AIMessage(state["pergunta_esclarecimento"])]}

    def conversa_geral(state: EstadoChat) -> Dict[str, Any]:
        sistema = prompts.CONVERSA_GERAL_TEMPLATE.format(
            prompt_principal=cfg.agente_principal.prompt.strip(),
            catalogo=prompts.catalogo_amigavel(cfg),
        )
        resposta = llm_especialista.invoke([SystemMessage(sistema), *_janela(state["messages"], janela)])
        return {"messages": [AIMessage(_texto(resposta))]}

    def fora_do_escopo(state: EstadoChat) -> Dict[str, Any]:
        texto = (
            "Esse assunto não está entre as regras de negócio que eu atendo. "
            "Posso ajudar com:\n" + prompts.catalogo_amigavel(cfg)
        )
        return {"messages": [AIMessage(texto)]}

    # ---- Montagem
    g = StateGraph(EstadoChat)
    g.add_node("agente_principal", agente_principal)
    destinos = ["esclarecer", "conversa_geral", "fora_do_escopo"]
    for e in cfg.especialistas:
        nome = f"especialista_{e.id}"
        g.add_node(nome, criar_no_especialista(e.id))
        g.add_edge(nome, "consolidar")
        destinos.append(nome)
    g.add_node("consolidar", consolidar)
    g.add_node("esclarecer", esclarecer)
    g.add_node("conversa_geral", conversa_geral)
    g.add_node("fora_do_escopo", fora_do_escopo)

    g.add_edge(START, "agente_principal")
    g.add_conditional_edges("agente_principal", detector_de_intencao, destinos)
    for n in ("consolidar", "esclarecer", "conversa_geral", "fora_do_escopo"):
        g.add_edge(n, END)

    return g.compile(checkpointer=checkpointer)


def criar_llms_databricks(cfg: AppConfig):
    """(roteador, especialistas) servidos pelo Databricks Model Serving."""
    from databricks_langchain import ChatDatabricks

    llm_roteador = ChatDatabricks(
        endpoint=cfg.llm.endpoint_roteador,
        temperature=cfg.llm.temperatura_roteador,
        max_tokens=600,
    )
    llm_especialista = ChatDatabricks(
        endpoint=cfg.llm.endpoint_especialistas,
        temperature=cfg.llm.temperatura_especialistas,
        max_tokens=cfg.llm.max_tokens,
    )
    return llm_roteador, llm_especialista


def construir_grafo_databricks(cfg: AppConfig, checkpointer=None):
    """Versão de produção: LLMs do Databricks Model Serving + Vector Search."""
    from .dados import DatabricksFonteDados
    from .retriever import DatabricksRetriever

    llm_roteador, llm_especialista = criar_llms_databricks(cfg)
    return construir_grafo(
        cfg,
        llm_roteador,
        llm_especialista,
        DatabricksRetriever(cfg),
        checkpointer,
        nome_modelo_roteador=cfg.llm.endpoint_roteador,
        nome_modelo_especialista=cfg.llm.endpoint_especialistas,
        fonte_dados=DatabricksFonteDados(cfg) if any(e.eh_bi for e in cfg.especialistas) else None,
    )
