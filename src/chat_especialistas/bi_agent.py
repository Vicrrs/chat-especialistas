"""Especialista como AGENTE DE BI CONVERSACIONAL.

Ciclo do agente (ReAct):  pensar  ->  usar ferramenta  ->  observar  ->  ...  ->  responder

Ferramentas de cada especialista (só as do seu domínio):
- consultar_regras_negocio : busca na documentação do domínio (Vector Search)
- descrever_tabela         : colunas, tipos e comentários (Unity Catalog)
- executar_sql             : SELECT somente leitura nas tabelas permitidas (SQL Warehouse)
- perguntar_genie          : delega a pergunta a um Genie Space (opcional)
"""
from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from . import prompts
from .config import AppConfig, EspecialistaConfig
from .dados import FonteDados
from .retriever import Retriever, Trecho, formatar_contexto
from .sql_guard import SQLNaoPermitido, validar_sql

log = logging.getLogger(__name__)
MAX_CHARS_OBSERVACAO = 12_000


def _texto(msg: AnyMessage) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)


def _tabela_texto(colunas: List[str], linhas: List[List[Any]], limite: int) -> str:
    if not colunas:
        return "(sem colunas)"
    if not linhas:
        return " | ".join(colunas) + "\n(0 linhas)"
    corpo = "\n".join(" | ".join("NULL" if v is None else str(v) for v in l) for l in linhas[:limite])
    rodape = ""
    if len(linhas) > limite:
        rodape = f"\n(mostrando {limite} de {len(linhas)} linhas — agregue mais se precisar de tudo)"
    return " | ".join(colunas) + "\n" + corpo + rodape


class _Sessao:
    """Estado de UMA execução do especialista (thread-safe por ser criado a cada chamada)."""

    def __init__(self):
        self.fontes: set[str] = set()
        self.consultas: List[Dict[str, Any]] = []
        self.ultimo_resultado: Optional[Dict[str, Any]] = None
        self.observacoes: List[str] = []


def _ferramentas(
    cfg: AppConfig,
    esp: EspecialistaConfig,
    retriever: Retriever,
    fonte: Optional[FonteDados],
    sessao: _Sessao,
) -> List[StructuredTool]:
    bi = cfg.bi
    ferramentas: List[StructuredTool] = []

    def consultar_regras_negocio(consulta: str) -> str:
        """Busca regras de negócio, definições de métricas e critérios na documentação do domínio."""
        trechos = retriever.buscar(consulta, esp)
        sessao.fontes.update(t.fonte for t in trechos)
        return formatar_contexto(trechos)

    if esp.documentos:
        ferramentas.append(StructuredTool.from_function(consultar_regras_negocio))

    if esp.dados and fonte is not None:
        permitidas = esp.dados.tabelas

        def descrever_tabela(tabela: str) -> str:
            """Mostra colunas, tipos e comentários de uma tabela (nome completo catalogo.schema.tabela)."""
            if tabela.strip().lower() not in permitidas:
                return f"ERRO: tabela fora do domínio. Permitidas: {permitidas}"
            return fonte.descrever_tabela(esp, tabela)

        def executar_sql(sql: str, objetivo: str = "") -> str:
            """Executa UMA consulta SELECT somente leitura no Databricks SQL e devolve o resultado.
            `objetivo` descreve em poucas palavras o que a consulta quer descobrir."""
            registro: Dict[str, Any] = {"via": "sql", "objetivo": objetivo, "sql": sql}
            try:
                sql_final = validar_sql(sql, permitidas, bi.max_linhas)
            except SQLNaoPermitido as e:
                registro.update(erro=f"bloqueado: {e}", linhas=0, duracao_ms=0)
                sessao.consultas.append(registro)
                return f"ERRO (consulta não permitida): {e}"
            registro["sql"] = sql_final
            res = fonte.executar_sql(esp, sql_final)
            registro.update(linhas=len(res.linhas), duracao_ms=res.duracao_ms, erro=res.erro)
            sessao.consultas.append(registro)
            if res.erro:
                return f"ERRO ao executar: {res.erro}"
            sessao.ultimo_resultado = {
                "sql": sql_final,
                "colunas": res.colunas,
                "linhas": res.linhas[: bi.linhas_para_o_llm],
                "total_linhas": len(res.linhas),
                "truncado": res.truncado or len(res.linhas) >= bi.max_linhas,
            }
            aviso = " (atingiu o LIMIT — pode haver mais linhas)" if sessao.ultimo_resultado["truncado"] else ""
            return f"{len(res.linhas)} linha(s){aviso}:\n" + _tabela_texto(
                res.colunas, res.linhas, bi.linhas_para_o_llm
            )

        ferramentas += [
            StructuredTool.from_function(descrever_tabela),
            StructuredTool.from_function(executar_sql),
        ]

    if esp.genie_space_id and fonte is not None:

        def perguntar_genie(pergunta: str) -> str:
            """Faz a pergunta, em linguagem natural, ao Genie Space deste domínio e devolve o SQL gerado e o resultado."""
            return _consultar_genie(cfg, esp, fonte, pergunta, sessao)[1]

        ferramentas.append(StructuredTool.from_function(perguntar_genie))

    return ferramentas


def _consultar_genie(
    cfg: AppConfig, esp: EspecialistaConfig, fonte: FonteDados, pergunta: str, sessao: _Sessao
) -> tuple[Dict[str, Any], str]:
    """Pergunta ao Genie, registra a consulta na sessão e devolve (resultado, texto para o LLM)."""
    r = fonte.perguntar_genie(esp, pergunta)
    sessao.consultas.append(
        {"via": "genie", "objetivo": pergunta, "sql": r.get("sql"), "linhas": len(r.get("linhas", [])),
         "duracao_ms": r.get("duracao_ms", 0), "erro": r.get("erro"), "genie_space_id": esp.genie_space_id,
         "genie_conversa_id": r.get("conversa_id"), "genie_mensagem_id": r.get("mensagem_id")}
    )
    if r.get("colunas"):
        sessao.ultimo_resultado = {
            "sql": r.get("sql"), "colunas": r["colunas"], "linhas": r["linhas"],
            "total_linhas": len(r["linhas"]), "truncado": False,
        }
    partes = []
    if r.get("erro"):
        partes.append(f"ERRO do Genie: {r['erro']}")
    if r.get("texto"):
        partes.append(f"Resposta do Genie: {r['texto']}")
    if r.get("sql"):
        partes.append(f"SQL do Genie:\n{r['sql']}")
    if r.get("colunas"):
        partes.append(_tabela_texto(r["colunas"], r["linhas"], cfg.bi.linhas_para_o_llm))
    return r, "\n\n".join(partes) or "Genie não retornou resultado."


def _prompt_bi(cfg: AppConfig, esp: EspecialistaConfig, trechos: List[Trecho]) -> str:
    dados = esp.dados
    tabelas = "\n".join(f"- {t}" for t in dados.tabelas) if dados else "(nenhuma — use o Genie)"
    exemplos = "(nenhum)"
    if dados and dados.exemplos_sql:
        exemplos = "\n\n".join(f"Pergunta: {e.pergunta}\n```sql\n{e.sql.strip()}\n```" for e in dados.exemplos_sql)
    genie = ""
    if esp.genie_space_id:
        genie = (
            "\n   Você também pode usar `perguntar_genie` para perguntas analíticas em linguagem"
            " natural; confira o SQL que o Genie gerou antes de confiar no resultado."
        )
    if dados and dados.descricao_tabelas:
        tabelas = "\n\n".join(
            f"### {t}\n{dados.descricao_tabelas.get(t, '(sem descrição — use descrever_tabela)').strip()}"
            for t in dados.tabelas
        )
    return prompts.BI_TEMPLATE.format(
        prompt_especialista=esp.prompt.strip(),
        nome=esp.nome,
        genie=genie,
        hoje=date.today().isoformat(),
        tabelas=tabelas,
        instrucoes=(dados.instrucoes.strip() if dados and dados.instrucoes else "(nenhuma)"),
        exemplos=exemplos,
        contexto=formatar_contexto(trechos) if esp.documentos else "(este agente não usa documentação)",
    )


def executar_especialista(
    cfg: AppConfig,
    esp: EspecialistaConfig,
    llm: BaseChatModel,
    retriever: Retriever,
    fonte: Optional[FonteDados],
    pergunta: str,
    historico: List[AnyMessage],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """`llm` deve ser o modelo SEM with_config: bind_tools descartaria a metadata
    usada pelo LLMOps, então ela é aplicada aqui depois do bind."""
    sessao = _Sessao()
    llm_simples = llm.with_config(metadata=metadata) if metadata else llm
    # Regras do domínio entram no contexto quando o agente tem documentação.
    trechos = retriever.buscar(pergunta, esp) if esp.documentos else []
    sessao.fontes.update(t.fonte for t in trechos)

    historico = list(historico)
    if historico and isinstance(historico[-1], HumanMessage):
        historico[-1] = HumanMessage(pergunta)
    elif not historico:
        historico = [HumanMessage(pergunta)]

    # ---- Especialista só de documentação: uma chamada, como antes.
    if not esp.eh_bi:
        sistema = prompts.prompt_especialista(esp, formatar_contexto(trechos))
        resposta = llm_simples.invoke([SystemMessage(sistema), *historico])
        return _resultado(esp, _texto(resposta), sessao, iteracoes=1)

    # ---- Genie direto: o Genie Space do domínio responde; o LLM só redige a resposta.
    if esp.genie_space_id and esp.genie_modo == "direto" and fonte is not None:
        try:
            r, texto_genie = _consultar_genie(cfg, esp, fonte, pergunta, sessao)
        except Exception as exc:  # noqa: BLE001 - Genie fora do ar: cai para o agente de BI, se houver tabelas
            log.exception("Falha no Genie %s", esp.genie_space_id)
            r, texto_genie = {"erro": f"{type(exc).__name__}: {exc}"}, ""
        if not r.get("erro") or not esp.dados:
            sistema = prompts.GENIE_TEMPLATE.format(
                prompt_especialista=esp.prompt.strip(),
                pergunta=pergunta,
                genie=texto_genie or f"ERRO do Genie: {r.get('erro')}",
                contexto=formatar_contexto(trechos) if esp.documentos else "(este agente não usa documentação)",
            )
            resposta = llm_simples.invoke([SystemMessage(sistema), *historico])
            return _resultado(esp, _texto(resposta), sessao, iteracoes=1)

    # ---- Agente de BI
    ferramentas = _ferramentas(cfg, esp, retriever, fonte, sessao)
    por_nome = {f.name: f for f in ferramentas}
    llm_com_ferramentas = llm.bind_tools(ferramentas)
    if metadata:
        llm_com_ferramentas = llm_com_ferramentas.with_config(metadata=metadata)
    mensagens: List[AnyMessage] = [SystemMessage(_prompt_bi(cfg, esp, trechos)), *historico]
    max_iter = esp.max_iteracoes or cfg.bi.max_iteracoes

    for iteracao in range(1, max_iter + 1):
        ai = llm_com_ferramentas.invoke(mensagens)
        mensagens.append(ai)
        chamadas = getattr(ai, "tool_calls", None) or []
        if not chamadas:
            return _resultado(esp, _texto(ai), sessao, iteracoes=iteracao)

        for chamada in chamadas:
            nome, args = chamada["name"], chamada.get("args") or {}
            ferramenta = por_nome.get(nome)
            if ferramenta is None:
                saida = f"ERRO: ferramenta '{nome}' não existe. Use: {list(por_nome)}"
            else:
                try:
                    saida = str(ferramenta.invoke(args))
                except Exception as exc:  # noqa: BLE001 - o agente vê o erro e tenta de novo
                    log.exception("Falha na ferramenta %s", nome)
                    saida = f"ERRO na ferramenta {nome}: {type(exc).__name__}: {exc}"
            saida = saida[:MAX_CHARS_OBSERVACAO]
            sessao.observacoes.append(f"[{nome}] {json.dumps(args, ensure_ascii=False)}\n{saida}")
            mensagens.append(ToolMessage(content=saida, tool_call_id=chamada["id"], name=nome))

    # ---- Limite de passos: responde com o que já apurou (sem ferramentas).
    final = llm_simples.invoke(
        [
            SystemMessage(esp.prompt.strip()),
            HumanMessage(
                prompts.FINALIZAR_TEMPLATE.format(
                    pergunta=pergunta, observacoes="\n\n".join(sessao.observacoes)[-MAX_CHARS_OBSERVACAO:]
                )
            ),
        ]
    )
    return _resultado(esp, _texto(final), sessao, iteracoes=max_iter, limite_atingido=True)


def _resultado(esp, texto, sessao: _Sessao, iteracoes: int, limite_atingido: bool = False) -> Dict[str, Any]:
    return {
        "especialista": esp.id,
        "nome": esp.nome,
        "texto": texto,
        "fontes": sorted(sessao.fontes),
        "consultas_sql": sessao.consultas,
        "dados": sessao.ultimo_resultado,
        "iteracoes": iteracoes,
        "limite_atingido": limite_atingido,
    }
