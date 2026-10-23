"""Publica cada agente especialista como um Genie Space (criar ou atualizar).

O conteúdo do espaço vem do que o agente sabe (estudo das tabelas + aprendizados aprovados):
  - tabelas do agente                        -> data_sources.tables
  - prompt de negócio, tabelas e métricas    -> instructions.text_instructions (um bloco gerenciado)
  - SQLs de exemplo validados                -> instructions.example_question_sqls
  - perguntas de exemplo                     -> config.sample_questions
  - benchmark (pergunta + SQL esperado)      -> benchmarks.questions

Ao atualizar, parte do `serialized_space` atual: o que foi curado direto no Genie (exemplos,
instruções fora do bloco gerenciado, configurações de colunas) é mantido.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from .estudo import EstudoAgente
from .gerador import AgenteRascunho, conhecimento

INICIO_BLOCO = "=== chat-especialistas: início (gerado pela app; edite pela app) ==="
FIM_BLOCO = "=== chat-especialistas: fim ==="


def _id() -> str:
    return uuid.uuid4().hex


def _linhas(texto: str) -> List[str]:
    """O Genie guarda textos como lista de linhas (com a quebra de linha no fim)."""
    return (texto or "").strip().splitlines(keepends=True)


def _texto(valor: Any) -> str:
    return "".join(valor) if isinstance(valor, list) else str(valor or "")


def _chave(pergunta: Any) -> str:
    return " ".join(_texto(pergunta).lower().split())


def instrucoes_genie(a: AgenteRascunho, e: EstudoAgente) -> str:
    partes = [f"Você é o {a.nome}."]
    if a.prompt_negocio.strip():
        partes.append("Contexto do negócio:\n" + a.prompt_negocio.strip())
    descricoes = [f"- {t}: {d.strip()}" for t, d in e.descricao_tabelas.items() if d and d.strip()]
    if descricoes:
        partes.append("Tabelas:\n" + "\n".join(descricoes))
    if e.instrucoes.strip():
        partes.append("Regras para as consultas e definições de métricas:\n" + e.instrucoes.strip())
    partes.append("Responda sempre em português do Brasil.")
    return "\n\n".join(partes)


def _bloco_gerenciado(atual: str, novo: str) -> str:
    """Troca só o bloco da app; o que alguém escreveu fora dele no Genie continua lá."""
    bloco = f"{INICIO_BLOCO}\n{novo.strip()}\n{FIM_BLOCO}"
    if INICIO_BLOCO in atual and FIM_BLOCO in atual:
        antes, resto = atual.split(INICIO_BLOCO, 1)
        depois = resto.split(FIM_BLOCO, 1)[1]
        return (antes + bloco + depois).strip()
    return f"{bloco}\n\n{atual.strip()}".strip()


def montar_espaco(a: AgenteRascunho, atual: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """`serialized_space` do agente, mesclado ao atual (se o espaço já existe)."""
    e = conhecimento(a)
    s: Dict[str, Any] = json.loads(json.dumps(atual)) if atual else {"version": 1}

    # ---- tabelas: mantém a configuração das que já estavam (ex.: colunas curadas no Genie)
    fontes = s.setdefault("data_sources", {})
    existentes = {t.get("identifier", "").lower(): t for t in fontes.get("tables") or []}
    fontes["tables"] = sorted(
        (existentes.get(t) or {"identifier": t} for t in a.tabelas), key=lambda t: t["identifier"]
    )

    inst = s.setdefault("instructions", {})
    # ---- instruções: o Genie aceita um único bloco de texto geral
    textos = inst.get("text_instructions") or []
    atual_txt = _texto(textos[0].get("content")) if textos else ""
    inst["text_instructions"] = [{
        "id": (textos[0].get("id") if textos else None) or _id(),
        "content": _linhas(_bloco_gerenciado(atual_txt, instrucoes_genie(a, e))),
    }]

    # ---- SQLs de exemplo: os da app substituem os de mesma pergunta; os curados no Genie ficam
    exemplos = {_chave(x.get("question")): x for x in inst.get("example_question_sqls") or []}
    for x in e.exemplos_sql:
        anterior = exemplos.get(_chave(x["pergunta"])) or {}
        exemplos[_chave(x["pergunta"])] = {
            "id": anterior.get("id") or _id(), "question": [x["pergunta"].strip()], "sql": _linhas(x["sql"]),
        }
    inst["example_question_sqls"] = sorted(exemplos.values(), key=lambda x: x["id"])

    # ---- perguntas sugeridas na tela do Genie
    if e.exemplos:
        config = s.setdefault("config", {})
        sugeridas = {_chave(q.get("question")): q for q in config.get("sample_questions") or []}
        for q in e.exemplos:
            sugeridas.setdefault(_chave(q), {"id": _id(), "question": [q.strip()]})
        config["sample_questions"] = sorted(sugeridas.values(), key=lambda q: q["id"])

    # ---- benchmark: perguntas com o SQL esperado, para avaliar o espaço com o tempo
    if a.benchmark:
        bench = s.setdefault("benchmarks", {})
        perguntas = {_chave(q.get("question")): q for q in bench.get("questions") or []}
        for b in a.benchmark:
            if not (b.get("pergunta") or "").strip():
                continue
            anterior = perguntas.get(_chave(b["pergunta"])) or {}
            item: Dict[str, Any] = {"id": anterior.get("id") or _id(), "question": [b["pergunta"].strip()]}
            if (b.get("sql") or "").strip():
                item["answer"] = [{"format": "SQL", "content": _linhas(b["sql"])}]
            perguntas[_chave(b["pergunta"])] = item
        bench["questions"] = sorted(perguntas.values(), key=lambda q: q["id"])
    return s


def descricao_espaco(a: AgenteRascunho) -> str:
    e = conhecimento(a)
    return (e.descricao or a.prompt_negocio or a.nome).strip()[:1000]


def publicar_espaco(w, a: AgenteRascunho, warehouse_id: str, pasta: Optional[str] = None) -> Dict[str, Any]:
    """Cria o Genie Space do agente (primeira vez) ou atualiza o que a app já criou.

    Devolve {"space_id", "criado": bool}. Um Genie escolhido à mão (não criado pela app) não é alterado.
    """
    if not a.tabelas:
        raise ValueError(f"'{a.nome}' não tem tabelas; o Genie Space precisa de ao menos uma.")
    if not warehouse_id:
        raise ValueError("Escolha um SQL Warehouse na aba 1.")
    titulo = a.nome.strip() or a.id
    if a.genie_space_id and a.genie_gerado:
        atual = w.genie.get_space(a.genie_space_id, include_serialized_space=True)
        serial = json.loads(atual.serialized_space) if atual.serialized_space else None
        w.genie.update_space(
            a.genie_space_id, title=titulo, description=descricao_espaco(a), warehouse_id=warehouse_id,
            serialized_space=json.dumps(montar_espaco(a, serial), ensure_ascii=False), etag=atual.etag,
        )
        return {"space_id": a.genie_space_id, "criado": False}
    if a.genie_space_id:
        raise ValueError(
            f"'{a.nome}' usa um Genie escolhido à mão; a app não altera esse espaço. "
            "Tire o Genie do agente na aba 2 para a app criar um próprio."
        )
    novo = w.genie.create_space(
        warehouse_id, json.dumps(montar_espaco(a), ensure_ascii=False),
        title=titulo, description=descricao_espaco(a), parent_path=pasta,
    )
    a.genie_space_id, a.genie_gerado = novo.space_id, True
    return {"space_id": novo.space_id, "criado": True}
