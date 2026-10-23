"""Melhoria contínua dos Genie Spaces dos especialistas.

Ciclo (a app roda sob demanda; o notebook 05 roda agendado):
  1. junta CASOS a melhorar: 👎 no Genie, 👎 no chat da app, erros e perguntas que falharam no benchmark;
  2. para cada caso, o LLM propõe uma instrução nova e/ou um SQL de exemplo; o SQL é EXECUTADO
     (com uma tentativa de correção) e só é marcado como validado se rodar;
  3. uma pessoa aprova ou rejeita na app; o aprovado vira "aprendizado" do agente;
  4. a app atualiza o Genie Space e roda o benchmark de novo; a nota fica no histórico.
Também detecta mudanças nas tabelas (colunas novas/removidas) para sugerir um novo estudo.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ..dados import executar_sql_warehouse, perguntar_genie_space
from .estudo import EstudoAgente, FonteEstudo, PerfilColuna, PerfilTabela, _validar_exemplo, extrair_json, perfil_texto
from .gerador import AgenteRascunho, conhecimento

MAX_CASOS_POR_RODADA = 10
STATUS_FIM_AVALIACAO = ("DONE", "EVALUATION_CANCELLED", "EVALUATION_FAILED", "EVALUATION_TIMEOUT")


def _agora() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _chave(texto: str) -> str:
    return " ".join((texto or "").lower().split())


def _valor(enum_ou_str: Any) -> str:
    return getattr(enum_ou_str, "value", enum_ou_str) or ""


# ------------------------------------------------------------------ estado
@dataclass
class Caso:
    agente_id: str
    origem: str  # benchmark | genie_feedback | genie_erro | chat
    pergunta: str
    sql_obtido: str = ""
    sql_esperado: str = ""
    detalhe: str = ""
    quando: str = field(default_factory=_agora)


@dataclass
class Proposta:
    agente_id: str
    origem: str
    pergunta: str
    diagnostico: str = ""
    instrucao: str = ""
    sql: str = ""
    validado: bool = False
    erro: str = ""
    detalhe_caso: str = ""
    status: str = "pendente"  # pendente | aprovada | rejeitada
    id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S%f"))
    criada_em: str = field(default_factory=_agora)


@dataclass
class EstadoMelhoria:
    casos: List[Dict[str, Any]] = field(default_factory=list)       # aguardando proposta
    propostas: List[Dict[str, Any]] = field(default_factory=list)
    historico: List[Dict[str, Any]] = field(default_factory=list)   # notas do benchmark ao longo do tempo
    lido_genie_ms: Dict[str, int] = field(default_factory=dict)     # até onde o feedback do Genie foi lido
    mudancas: Dict[str, List[str]] = field(default_factory=dict)    # mudanças nas tabelas por agente

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, default=str)

    @classmethod
    def from_json(cls, texto: Optional[str]) -> "EstadoMelhoria":
        if not texto:
            return cls()
        d = json.loads(texto)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def ja_conhecido(self, agente_id: str, pergunta: str) -> bool:
        k = _chave(pergunta)
        return any(
            x["agente_id"] == agente_id and _chave(x["pergunta"]) == k
            for x in self.casos + [p for p in self.propostas if p["status"] == "pendente"]
        )

    def adicionar_casos(self, casos: List[Caso]) -> int:
        novos = 0
        for c in casos:
            if c.pergunta.strip() and not self.ja_conhecido(c.agente_id, c.pergunta):
                self.casos.append(asdict(c))
                novos += 1
        return novos

    def pendentes(self, agente_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return [p for p in self.propostas if p["status"] == "pendente" and agente_id in (None, p["agente_id"])]


# ------------------------------------------------------------------ casos
def _sql_da_mensagem(msg) -> str:
    for anexo in getattr(msg, "attachments", None) or []:
        if anexo.query and anexo.query.query:
            return anexo.query.query
    return ""


def casos_do_genie(w, space_id: str, agente_id: str, desde_ms: int = 0, max_conversas: int = 200) -> tuple[List[Caso], int]:
    """Mensagens do Genie com 👎 ou erro desde `desde_ms`. Devolve (casos, maior timestamp lido)."""
    casos: List[Caso] = []
    maior, token, lidas = desde_ms, None, 0
    while lidas < max_conversas:
        resp = w.genie.list_conversations(space_id, include_all=True, page_size=100, page_token=token)
        for conv in resp.conversations or []:
            lidas += 1
            msgs = w.genie.list_conversation_messages(space_id, conv.conversation_id, page_size=100).messages or []
            for m in msgs:
                ts = m.last_updated_timestamp or m.created_timestamp or 0
                if ts <= desde_ms:
                    continue
                maior = max(maior, ts)
                fb = m.feedback
                if fb and _valor(fb.rating) == "NEGATIVE":
                    casos.append(Caso(agente_id, "genie_feedback", m.content, _sql_da_mensagem(m),
                                      detalhe=f"👎 no Genie{': ' + fb.comment if fb.comment else ''}"))
                elif m.error and m.error.error:
                    casos.append(Caso(agente_id, "genie_erro", m.content, _sql_da_mensagem(m),
                                      detalhe=f"erro no Genie: {m.error.error}"))
        token = resp.next_page_token
        if not token:
            break
    return casos, maior


def caso_do_chat(agente_id: str, pergunta: str, sql: str = "", comentario: str = "") -> Caso:
    return Caso(agente_id, "chat", pergunta, sql, detalhe=f"👎 no chat{': ' + comentario if comentario else ''}")


# --------------------------------------------------------------- benchmark
def _normalizar(v: Any) -> Any:
    if v is None:
        return None
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return str(v).strip().lower()


def _coluna(linhas: List[List[Any]], i: int) -> List[Any]:
    return sorted((_normalizar(l[i]) for l in linhas), key=lambda x: (x is None, str(x)))


def mesmos_resultados(colunas_esp, linhas_esp, colunas_obt, linhas_obt) -> bool:
    """Mesmo nº de linhas e cada coluna esperada aparece (com os mesmos valores) no resultado obtido.
    Não depende de nomes nem da ordem das colunas/linhas; colunas extras no obtido são aceitas."""
    if len(linhas_esp) != len(linhas_obt):
        return False
    obtidas = [_coluna(linhas_obt, i) for i in range(len(colunas_obt))]
    return all(_coluna(linhas_esp, i) in obtidas for i in range(len(colunas_esp)))


def _benchmark_nativo(w, space_id: str, log, espera_max: int) -> Dict[str, Any]:
    run = w.genie.genie_create_eval_run(space_id)
    limite = time.time() + espera_max
    while _valor(run.eval_run_status) not in STATUS_FIM_AVALIACAO:
        if time.time() > limite:
            raise TimeoutError(f"avaliação do Genie passou de {espera_max}s")
        time.sleep(5)
        run = w.genie.genie_get_eval_run(space_id, run.eval_run_id)
    if _valor(run.eval_run_status) != "DONE":
        raise RuntimeError(f"avaliação do Genie terminou com {_valor(run.eval_run_status)}")
    itens, token = [], None
    while True:
        resp = w.genie.genie_list_eval_results(space_id, run.eval_run_id, page_size=100, page_token=token)
        for r in resp.eval_results or []:
            d = w.genie.genie_get_eval_result_details(space_id, run.eval_run_id, r.result_id)
            avaliacao = _valor(d.assessment)
            itens.append({
                "pergunta": r.question or "",
                "ok": avaliacao == "GOOD",
                "avaliacao": avaliacao,
                "motivos": [_valor(m) for m in d.assessment_reasons or []],
                "sql_esperado": next((x.response for x in d.expected_response or [] if x.response), r.benchmark_answer or ""),
                "sql_obtido": next((x.response for x in d.actual_response or [] if x.response), ""),
            })
        token = resp.next_page_token
        if not token:
            break
    log(f"   avaliação nativa do Genie: {run.num_correct or 0}/{run.num_questions or len(itens)} corretas")
    return {"via": "genie", "itens": itens}


def _benchmark_local(w, space_id: str, warehouse_id: str, benchmark: List[Dict[str, str]], log) -> Dict[str, Any]:
    itens = []
    for b in benchmark:
        if not (b.get("pergunta") and b.get("sql")):
            continue
        item: Dict[str, Any] = {"pergunta": b["pergunta"], "sql_esperado": b["sql"], "motivos": []}
        try:
            obtido = perguntar_genie_space(w, space_id, b["pergunta"], max_linhas=1000)
            item["sql_obtido"] = obtido.get("sql") or ""
            esperado = executar_sql_warehouse(w, warehouse_id, b["sql"], 1000, 120)
            if esperado.erro:
                item.update(ok=False, avaliacao="NEEDS_REVIEW", motivos=[f"SQL esperado falhou: {esperado.erro}"])
            elif obtido.get("erro") or not obtido.get("colunas"):
                item.update(ok=False, avaliacao="BAD", motivos=[obtido.get("erro") or "Genie não trouxe dados"])
            else:
                ok = mesmos_resultados(esperado.colunas, esperado.linhas, obtido["colunas"], obtido["linhas"])
                item.update(ok=ok, avaliacao="GOOD" if ok else "BAD", motivos=[] if ok else ["resultado diferente"])
        except Exception as e:  # noqa: BLE001
            item.update(ok=False, avaliacao="BAD", motivos=[f"{type(e).__name__}: {e}"])
        log(f"   {'✓' if item['ok'] else '✗'} {b['pergunta']}")
        itens.append(item)
    return {"via": "local", "itens": itens}


def rodar_benchmark(
    w, a: AgenteRascunho, warehouse_id: str, log: Callable[[str], None] = print, espera_max: int = 900
) -> Dict[str, Any]:
    """Avalia o Genie Space do agente. Usa a avaliação nativa do Genie (benchmarks do espaço);
    se ela não estiver disponível no workspace, compara o resultado do Genie com o do SQL esperado."""
    if not a.genie_space_id:
        raise ValueError(f"'{a.nome}' não tem Genie Space.")
    try:
        r = _benchmark_nativo(w, a.genie_space_id, log, espera_max)
    except Exception as e:  # noqa: BLE001 - API de avaliação ausente/desabilitada: faz a comparação local
        log(f"   avaliação nativa indisponível ({type(e).__name__}: {str(e)[:120]}); comparando resultados aqui")
        r = _benchmark_local(w, a.genie_space_id, warehouse_id, a.benchmark, log)
    acertos = sum(1 for i in r["itens"] if i["ok"])
    r.update(agente_id=a.id, quando=_agora(), acertos=acertos, total=len(r["itens"]),
             acuracia=(acertos / len(r["itens"])) if r["itens"] else None)
    return r


def casos_do_benchmark(resultado: Dict[str, Any]) -> List[Caso]:
    return [
        Caso(resultado["agente_id"], "benchmark", i["pergunta"], i.get("sql_obtido") or "", i.get("sql_esperado") or "",
             detalhe="falhou no benchmark" + (f": {', '.join(i['motivos'])}" if i.get("motivos") else ""))
        for i in resultado["itens"] if not i["ok"]
    ]


def resumo_benchmark(resultado: Dict[str, Any]) -> Dict[str, Any]:
    return {k: resultado.get(k) for k in ("agente_id", "quando", "via", "acertos", "total", "acuracia")}


# ---------------------------------------------------------------- propostas
PROPOR_SISTEMA = """Você melhora um Genie Space (assistente de dados em linguagem natural do Databricks).
Recebe o conhecimento atual do espaço, o perfil das tabelas e um CASO em que a resposta foi ruim.
Diagnostique a causa e proponha a menor correção que faria o Genie acertar perguntas como essa:
- `instrucao`: uma regra curta e geral (definição de métrica, filtro padrão, join correto, significado
  de um valor). Deixe vazio se o problema não for falta de regra.
- `sql`: a consulta correta para a pergunta do caso (Databricks SQL, somente leitura, nomes completos
  catalogo.schema.tabela, só colunas que existem no perfil). Ela vira um exemplo do Genie.
Responda SOMENTE com JSON: {"diagnostico": "...", "instrucao": "...", "sql": "..."}"""

PROPOR_USUARIO = """Conhecimento atual do espaço:
{conhecimento}

Perfil das tabelas:
{perfis}

CASO ({origem}): {detalhe}
Pergunta: {pergunta}
SQL que o Genie usou: {sql_obtido}
SQL esperado (se houver): {sql_esperado}"""


def perfis_texto(e: EstudoAgente) -> str:
    perfis = []
    for p in e.perfis:
        d = dict(p)
        d["colunas"] = [PerfilColuna(**c) for c in d.get("colunas") or []]
        perfis.append(perfil_texto(PerfilTabela(**d)))
    return "\n\n".join(perfis) or "(sem perfil — estude as tabelas)"


def _conhecimento_txt(a: AgenteRascunho, e: EstudoAgente) -> str:
    exemplos = "\n".join(f"- {x['pergunta']}\n  {x['sql'].strip()}" for x in e.exemplos_sql) or "(nenhum)"
    return f"Negócio: {a.prompt_negocio.strip() or '(não informado)'}\n\nInstruções:\n{e.instrucoes or '(nenhuma)'}\n\nExemplos SQL:\n{exemplos}"


def propor_melhoria(llm, fonte: FonteEstudo, a: AgenteRascunho, caso: Dict[str, Any], max_linhas: int = 200) -> Proposta:
    e = conhecimento(a)
    perfis = perfis_texto(e)
    p = Proposta(agente_id=a.id, origem=caso["origem"], pergunta=caso["pergunta"], detalhe_caso=caso.get("detalhe", ""))
    try:
        resp = llm.invoke([
            SystemMessage(PROPOR_SISTEMA),
            HumanMessage(PROPOR_USUARIO.format(
                conhecimento=_conhecimento_txt(a, e), perfis=perfis, origem=caso["origem"], detalhe=caso.get("detalhe", ""),
                pergunta=caso["pergunta"], sql_obtido=caso.get("sql_obtido") or "(nenhum)",
                sql_esperado=caso.get("sql_esperado") or "(não informado)",
            )),
        ])
        dados = extrair_json(resp.content if isinstance(resp.content, str) else str(resp.content))
    except Exception as ex:  # noqa: BLE001
        p.erro = f"o LLM não devolveu uma proposta válida: {ex}"
        return p
    p.diagnostico = str(dados.get("diagnostico") or "").strip()
    p.instrucao = str(dados.get("instrucao") or "").strip()
    # O SQL esperado do benchmark já é a resposta certa; senão, vale o do LLM.
    sql = (caso.get("sql_esperado") or str(dados.get("sql") or "")).strip().rstrip(";")
    if sql:
        v = _validar_exemplo(llm, fonte, p.pergunta, sql, a.tabelas, max_linhas, perfis)
        p.sql, p.validado, p.erro = v["sql"], v["ok"], v["erro"] or ""
    return p


def aprovar(a: AgenteRascunho, proposta: Dict[str, Any], *, instrucao: str, sql: str, no_benchmark: bool = True) -> None:
    """Grava a proposta (com as edições de quem aprovou) como aprendizado do agente."""
    quando = _agora()
    if instrucao.strip():
        a.aprendizados.append({"tipo": "instrucao", "instrucao": instrucao.strip(), "origem": proposta["origem"], "quando": quando})
    if sql.strip():
        a.aprendizados.append({"tipo": "exemplo_sql", "pergunta": proposta["pergunta"], "sql": sql.strip(),
                               "origem": proposta["origem"], "quando": quando})
        if no_benchmark and not any(_chave(b["pergunta"]) == _chave(proposta["pergunta"]) for b in a.benchmark):
            a.benchmark.append({"pergunta": proposta["pergunta"], "sql": sql.strip()})
    proposta.update(status="aprovada", instrucao=instrucao.strip(), sql=sql.strip())


# ---------------------------------------------------------------- mudanças
def detectar_mudancas(fonte: FonteEstudo, a: AgenteRascunho) -> List[str]:
    """Compara as colunas atuais de cada tabela com as do último estudo."""
    e = EstudoAgente.from_dict(a.estudo) if a.estudo else EstudoAgente()
    antes = {p["tabela"]: {c["nome"]: c["tipo"] for c in p.get("colunas") or []} for p in e.perfis}
    mudancas = []
    for t in a.tabelas:
        try:
            agora = {c.nome: c.tipo for c in fonte.colunas(t)}
        except Exception as ex:  # noqa: BLE001
            mudancas.append(f"{t}: não foi possível ler ({ex})")
            continue
        if t not in antes:
            mudancas.append(f"{t}: tabela ainda não estudada")
            continue
        novas = sorted(set(agora) - set(antes[t]))
        removidas = sorted(set(antes[t]) - set(agora))
        tipos = sorted(c for c in set(agora) & set(antes[t]) if agora[c] != antes[t][c])
        if novas:
            mudancas.append(f"{t}: colunas novas {', '.join(novas)}")
        if removidas:
            mudancas.append(f"{t}: colunas removidas {', '.join(removidas)}")
        if tipos:
            mudancas.append(f"{t}: tipo mudou em {', '.join(tipos)}")
    return mudancas


# ------------------------------------------------------------------- ciclo
def ciclo_melhoria(
    w,
    llm,
    fonte: FonteEstudo,
    agentes: List[AgenteRascunho],
    estado: EstadoMelhoria,
    warehouse_id: str,
    *,
    benchmark: bool = True,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Uma rodada completa para os agentes com Genie Space criado pela app. Não altera nenhum Genie:
    só junta casos, avalia e deixa propostas para aprovação."""
    resumo: Dict[str, Any] = {}
    for a in [a for a in agentes if a.genie_space_id and a.genie_gerado]:
        log(f"🔁 {a.nome}")
        r: Dict[str, Any] = {"casos_novos": 0, "propostas": 0}
        try:
            casos, lido = casos_do_genie(w, a.genie_space_id, a.id, estado.lido_genie_ms.get(a.id, 0))
            estado.lido_genie_ms[a.id] = lido
            r["casos_novos"] += estado.adicionar_casos(casos)
        except Exception as e:  # noqa: BLE001
            log(f"   não consegui ler as conversas do Genie: {e}")
        if benchmark and a.benchmark:
            try:
                b = rodar_benchmark(w, a, warehouse_id, log)
                estado.historico.append(resumo_benchmark(b))
                r["benchmark"] = resumo_benchmark(b)
                r["casos_novos"] += estado.adicionar_casos(casos_do_benchmark(b))
            except Exception as e:  # noqa: BLE001
                log(f"   benchmark falhou: {e}")
        estado.mudancas[a.id] = detectar_mudancas(fonte, a)
        r["mudancas"] = estado.mudancas[a.id]
        r["propostas"] = gerar_propostas(llm, fonte, a, estado, log)
        resumo[a.id] = r
    return resumo


def gerar_propostas(llm, fonte: FonteEstudo, a: AgenteRascunho, estado: EstadoMelhoria,
                    log: Callable[[str], None] = print, limite: int = MAX_CASOS_POR_RODADA) -> int:
    """Transforma os casos pendentes do agente em propostas (até `limite` por rodada)."""
    meus = [c for c in estado.casos if c["agente_id"] == a.id][:limite]
    for c in meus:
        p = propor_melhoria(llm, fonte, a, c)
        estado.propostas.append(asdict(p))
        estado.casos.remove(c)
        log(f"   {'✓' if p.validado else '•'} proposta: {c['pergunta']}")
    return len(meus)
