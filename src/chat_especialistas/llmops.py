"""LLMOps: tokens, custo em US$ e registro de cada interação.

Como funciona
-------------
1. `RastreadorUso` é um callback do LangChain que captura, para CADA chamada de
   LLM do grafo, a etapa (nó do LangGraph), o endpoint, tokens de entrada/saída,
   latência e custo calculado pela tabela de preços do `agents.yaml`.
2. `montar_registro` junta isso com a pergunta, a resposta, a intenção detectada,
   os especialistas acionados e as fontes em UM registro por interação.
3. `emitir` envia o registro para os destinos configurados (log, trace do MLflow,
   arquivo JSONL). Além disso o agent.py coloca o registro em
   `custom_outputs.llmops`, que o Databricks grava na inference table (Delta).
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from .config import AppConfig, LLMOpsConfig

CHAVE_MODELO = "llmops_modelo"  # metadata colocada em cada LLM (ver graph.py)
_lock_arquivo = threading.Lock()


def estimar_tokens(texto: str) -> int:
    """Estimativa (~4 caracteres por token) quando o endpoint não devolve `usage`."""
    return max(1, math.ceil(len(texto or "") / 4)) if texto else 0


def custo_usd(cfg: LLMOpsConfig, modelo: str, tokens_input: int, tokens_output: int) -> float:
    p = cfg.preco(modelo)
    return round((tokens_input * p.input + tokens_output * p.output) / 1_000_000, 8)


@dataclass
class ChamadaLLM:
    etapa: str
    modelo: str
    tokens_input: int = 0
    tokens_output: int = 0
    custo_usd: float = 0.0
    latencia_ms: int = 0
    estimado: bool = False
    tipo: str = "llm"  # "llm" ou "embedding"


@dataclass
class _Pendente:
    etapa: str
    modelo: str
    inicio: float
    texto_entrada: str


class RastreadorUso(BaseCallbackHandler):
    """Callback thread-safe (especialistas rodam em paralelo). Crie um por requisição."""

    raise_error = False

    def __init__(self, cfg: LLMOpsConfig):
        self.cfg = cfg
        self.chamadas: List[ChamadaLLM] = []
        self._pendentes: Dict[UUID, _Pendente] = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------- callbacks
    def on_chat_model_start(self, serialized, messages, *, run_id: UUID, metadata=None, **kwargs):
        metadata = metadata or {}
        texto = "\n".join(
            str(m.content) for lote in messages for m in lote
        )
        with self._lock:
            self._pendentes[run_id] = _Pendente(
                etapa=metadata.get("langgraph_node", "desconhecida"),
                modelo=metadata.get(CHAVE_MODELO) or _nome_modelo(serialized, kwargs),
                inicio=time.perf_counter(),
                texto_entrada=texto,
            )

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs):
        with self._lock:
            pend = self._pendentes.pop(run_id, None)
        if pend is None:
            return
        tin, tout, texto_saida = _extrair_uso(response)
        estimado = tin is None or tout is None
        if tin is None:
            tin = estimar_tokens(pend.texto_entrada)
        if tout is None:
            tout = estimar_tokens(texto_saida)
        chamada = ChamadaLLM(
            etapa=pend.etapa,
            modelo=pend.modelo,
            tokens_input=int(tin),
            tokens_output=int(tout),
            custo_usd=custo_usd(self.cfg, pend.modelo, tin, tout),
            latencia_ms=int((time.perf_counter() - pend.inicio) * 1000),
            estimado=estimado,
        )
        with self._lock:
            self.chamadas.append(chamada)

    def on_llm_error(self, error, *, run_id: UUID, **kwargs):
        with self._lock:
            self._pendentes.pop(run_id, None)

    # ------------------------------------------------------------ extras
    def registrar_embedding(self, etapa: str, modelo: str, texto: str):
        """Custo (estimado) do embedding da pergunta feito pelo Vector Search."""
        tin = estimar_tokens(texto)
        with self._lock:
            self.chamadas.append(
                ChamadaLLM(etapa, modelo, tin, 0, custo_usd(self.cfg, modelo, tin, 0), 0, True, "embedding")
            )


def _nome_modelo(serialized: Optional[dict], kwargs: dict) -> str:
    params = kwargs.get("invocation_params") or {}
    for chave in ("endpoint", "model", "model_name"):
        if params.get(chave):
            return str(params[chave])
    return (serialized or {}).get("kwargs", {}).get("endpoint", "desconhecido")


def _extrair_uso(response: LLMResult):
    texto_saida = ""
    tin = tout = None
    try:
        geracao = response.generations[0][0]
        texto_saida = geracao.text or ""
        msg = getattr(geracao, "message", None)
        uso = getattr(msg, "usage_metadata", None)
        if uso:
            tin, tout = uso.get("input_tokens"), uso.get("output_tokens")
    except (IndexError, AttributeError):
        pass
    if tin is None and response.llm_output:
        uso = response.llm_output.get("token_usage") or response.llm_output.get("usage") or {}
        tin = uso.get("prompt_tokens", uso.get("input_tokens"))
        tout = uso.get("completion_tokens", uso.get("output_tokens"))
    return tin, tout, texto_saida


# ------------------------------------------------------------------ registro
def montar_registro(
    cfg: AppConfig,
    *,
    rastreador: RastreadorUso,
    pergunta: str,
    resposta: str,
    estado: Optional[Dict[str, Any]] = None,
    fontes: Optional[List[str]] = None,
    latencia_ms: int = 0,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    erro: Optional[str] = None,
) -> Dict[str, Any]:
    estado = estado or {}
    ops = cfg.llmops
    chamadas = [asdict(c) for c in rastreador.chamadas]

    por_etapa: Dict[str, Dict[str, float]] = {}
    for c in chamadas:
        e = por_etapa.setdefault(c["etapa"], {"tokens": 0, "custo_usd": 0.0})
        e["tokens"] += c["tokens_input"] + c["tokens_output"]
        e["custo_usd"] = round(e["custo_usd"] + c["custo_usd"], 8)

    consultas = [
        {
            "especialista": r.get("especialista"),
            "via": c.get("via"),
            "objetivo": c.get("objetivo") if ops.registrar_textos else None,  # é a pergunta reescrita
            "sql": c.get("sql") if ops.registrar_textos else None,
            "linhas": c.get("linhas"),
            "duracao_ms": c.get("duracao_ms"),
            "erro": c.get("erro"),
        }
        for r in (estado.get("respostas") or [])
        for c in r.get("consultas_sql", [])
    ]

    tin = sum(c["tokens_input"] for c in chamadas)
    tout = sum(c["tokens_output"] for c in chamadas)
    return {
        "request_id": request_id or str(uuid.uuid4()),
        "conversation_id": conversation_id,
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "versao_agente": ops.versao_agente,
        "pergunta": pergunta if ops.registrar_textos else None,
        "resposta": resposta if ops.registrar_textos else None,
        "pergunta_reescrita": estado.get("pergunta_reescrita") if ops.registrar_textos else None,
        "intencao": estado.get("intencao"),
        "especialistas": estado.get("especialistas", []),
        "confianca": estado.get("confianca"),
        "fontes": fontes or [],
        "qtd_chamadas_llm": sum(1 for c in chamadas if c["tipo"] == "llm"),
        "tokens_input": tin,
        "tokens_output": tout,
        "tokens_total": tin + tout,
        "custo_usd": round(sum(c["custo_usd"] for c in chamadas), 8),
        "custo_por_etapa": por_etapa,
        "tokens_estimados": any(c["estimado"] for c in chamadas),
        "latencia_ms": latencia_ms,
        "erro": erro,
        "chamadas": chamadas,
        # BI: consultas executadas no SQL Warehouse / Genie (custo do warehouse é cobrado à parte, em DBU)
        "qtd_consultas_sql": len(consultas),
        "tempo_sql_ms": sum(c["duracao_ms"] or 0 for c in consultas),
        "consultas_sql": consultas,
    }


# ------------------------------------------------------------------ destinos
def emitir(cfg: AppConfig, registro: Dict[str, Any]) -> None:
    ops = cfg.llmops
    if not ops.habilitado:
        return
    for destino in ops.destinos:
        try:
            if destino == "log":
                _para_log(registro)
            elif destino == "mlflow_trace":
                _para_mlflow_trace(registro)
            elif destino == "jsonl" and ops.arquivo_jsonl:
                _para_jsonl(ops.arquivo_jsonl, registro)
        except Exception as exc:  # noqa: BLE001 - LLMOps nunca derruba o chat
            print(f"[LLMOPS] falha no destino '{destino}': {exc}", flush=True)


def _para_log(registro: Dict[str, Any]) -> None:
    # Prefixo fixo para filtrar nos logs do Model Serving.
    print("LLMOPS " + json.dumps(registro, ensure_ascii=False, default=str), flush=True)


def _para_mlflow_trace(registro: Dict[str, Any]) -> None:
    import mlflow

    tags = {
        "llmops.custo_usd": f"{registro['custo_usd']:.8f}",
        "llmops.tokens_input": str(registro["tokens_input"]),
        "llmops.tokens_output": str(registro["tokens_output"]),
        "llmops.tokens_total": str(registro["tokens_total"]),
        "llmops.intencao": str(registro["intencao"]),
        "llmops.especialistas": ",".join(registro["especialistas"]),
        "llmops.versao_agente": str(registro["versao_agente"]),
    }
    if registro.get("conversation_id"):
        tags["llmops.conversation_id"] = str(registro["conversation_id"])
    if registro.get("user_id"):
        tags["llmops.user_id"] = str(registro["user_id"])
    mlflow.update_current_trace(tags=tags)
    # Registro completo num span próprio: o notebook 03 lê daqui para a tabela Delta.
    with mlflow.start_span(name="llmops_registro") as span:
        span.set_outputs(registro)


def _para_jsonl(caminho: str, registro: Dict[str, Any]) -> None:
    pasta = os.path.dirname(caminho)
    if pasta:
        os.makedirs(pasta, exist_ok=True)
    with _lock_arquivo, open(caminho, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False, default=str) + "\n")
