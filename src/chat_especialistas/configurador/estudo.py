"""Estudo das tabelas de cada agente.

Para cada agente, a partir do prompt de negócio escrito por quem configura:
  1. lê os metadados (colunas, tipos, comentários) de cada tabela;
  2. levanta um perfil dos dados (nulos, distintos, mínimo/máximo, valores mais frequentes) e uma amostra;
  3. pede ao LLM o "conhecimento" do agente: descrição para o roteador, perguntas de exemplo,
     significado das tabelas/colunas, instruções de métricas e SQLs de exemplo;
  4. EXECUTA cada SQL de exemplo; o que falhar volta ao LLM uma vez para correção; o que
     continuar falhando é descartado. Só entram no agente consultas que rodaram de verdade.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..dados import ResultadoSQL
from ..sql_guard import SQLNaoPermitido, validar_sql

TIPOS_COMPLEXOS = ("array", "map", "struct", "binary", "variant", "geography", "geometry")
TIPOS_CATEGORICOS = ("string", "varchar", "char", "boolean", "text")
LINHAS_PERFIL = 100_000
MAX_COLUNAS = 80
MAX_COLUNAS_TOP = 25
MAX_CHARS_VALOR = 60


# ------------------------------------------------------------------ fontes
@dataclass
class Coluna:
    nome: str
    tipo: str
    comentario: str = ""


class FonteEstudo(Protocol):
    def colunas(self, tabela: str) -> List[Coluna]: ...
    def comentario_tabela(self, tabela: str) -> str: ...
    def consultar(self, sql: str) -> ResultadoSQL: ...
    def descrever_genie(self, space_id: str) -> str: ...


class SparkFonteEstudo:
    """Usa a sessão Spark do notebook (permissões de quem está configurando)."""

    def __init__(self, spark, w=None, max_linhas: int = 1000):
        self.spark = spark
        self.w = w
        self.max_linhas = max_linhas

    def colunas(self, tabela: str) -> List[Coluna]:
        return [
            Coluna(f.name, f.dataType.simpleString(), (f.metadata or {}).get("comment", "") or "")
            for f in self.spark.table(tabela).schema.fields
        ]

    def comentario_tabela(self, tabela: str) -> str:
        try:
            for linha in self.spark.sql(f"DESCRIBE TABLE EXTENDED {tabela}").collect():
                if (linha[0] or "").strip() == "Comment":
                    return linha[1] or ""
        except Exception:  # noqa: BLE001
            pass
        return ""

    def consultar(self, sql: str) -> ResultadoSQL:
        import time

        inicio = time.perf_counter()
        try:
            df = self.spark.sql(sql)
            linhas = [list(r) for r in df.limit(self.max_linhas).collect()]
            return ResultadoSQL(
                colunas=df.columns, linhas=linhas, duracao_ms=int((time.perf_counter() - inicio) * 1000)
            )
        except Exception as e:  # noqa: BLE001
            msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            return ResultadoSQL(erro=msg[:500], duracao_ms=int((time.perf_counter() - inicio) * 1000))

    def descrever_genie(self, space_id: str) -> str:
        if self.w is None:
            return ""
        g = self.w.genie.get_space(space_id)
        return f"Genie Space '{g.title}': {g.description or '(sem descrição)'}"


class WarehouseFonteEstudo:
    """Usa um SQL Warehouse + Unity Catalog (é o que a app Streamlit usa; não precisa de Spark)."""

    def __init__(self, w, warehouse_id: str, max_linhas: int = 1000, timeout_segundos: int = 180):
        self.w = w
        self.warehouse_id = warehouse_id
        self.max_linhas = max_linhas
        self.timeout = timeout_segundos
        self._info: Dict[str, Any] = {}

    def _tabela(self, tabela: str):
        if tabela not in self._info:
            self._info[tabela] = self.w.tables.get(tabela)
        return self._info[tabela]

    def colunas(self, tabela: str) -> List[Coluna]:
        return [Coluna(c.name, (c.type_text or "").lower(), c.comment or "") for c in self._tabela(tabela).columns or []]

    def comentario_tabela(self, tabela: str) -> str:
        return self._tabela(tabela).comment or ""

    def consultar(self, sql: str) -> ResultadoSQL:
        from ..dados import executar_sql_warehouse

        return executar_sql_warehouse(self.w, self.warehouse_id, sql, self.max_linhas, self.timeout)

    def descrever_genie(self, space_id: str) -> str:
        g = self.w.genie.get_space(space_id)
        return f"Genie Space '{g.title}': {g.description or '(sem descrição)'}"


# ------------------------------------------------------------------ perfil
@dataclass
class PerfilColuna:
    nome: str
    tipo: str
    comentario: str = ""
    nulos_pct: Optional[float] = None
    distintos: Optional[int] = None
    minimo: Any = None
    maximo: Any = None
    top_valores: List[List[Any]] = field(default_factory=list)


@dataclass
class PerfilTabela:
    tabela: str
    comentario: str = ""
    total_linhas: Optional[int] = None
    colunas: List[PerfilColuna] = field(default_factory=list)
    amostra_colunas: List[str] = field(default_factory=list)
    amostra: List[List[Any]] = field(default_factory=list)
    erros: List[str] = field(default_factory=list)


def _q(nome: str) -> str:
    return "`" + nome.replace("`", "``") + "`"


def _curto(v: Any) -> Any:
    if v is None or isinstance(v, (int, float, bool)):
        return v
    s = str(v)
    return s if len(s) <= MAX_CHARS_VALOR else s[: MAX_CHARS_VALOR - 1] + "…"


def _num(v: Any) -> Any:
    """O SQL Warehouse devolve tudo como texto; converte contagens para número."""
    if v is None or isinstance(v, (int, float)):
        return v
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return v


def _eh(tipo: str, prefixos) -> bool:
    return tipo.lower().startswith(prefixos)


def perfilar_tabela(fonte: FonteEstudo, tabela: str, log: Callable[[str], None] = lambda _: None) -> PerfilTabela:
    p = PerfilTabela(tabela=tabela)
    try:
        cols = fonte.colunas(tabela)
    except Exception as e:  # noqa: BLE001
        p.erros.append(f"não foi possível ler as colunas: {e}")
        return p
    p.comentario = fonte.comentario_tabela(tabela)
    p.colunas = [PerfilColuna(c.nome, c.tipo, c.comentario) for c in cols[:MAX_COLUNAS]]
    if len(cols) > MAX_COLUNAS:
        p.erros.append(f"tabela com {len(cols)} colunas; perfil feito nas {MAX_COLUNAS} primeiras")

    r = fonte.consultar(f"SELECT count(*) AS n FROM {tabela}")
    if r.erro:
        p.erros.append(f"contagem: {r.erro}")
    elif r.linhas:
        p.total_linhas = int(_num(r.linhas[0][0]))

    base = f"(SELECT * FROM {tabela} LIMIT {LINHAS_PERFIL}) AS base"
    simples = [c for c in p.colunas if not _eh(c.tipo, TIPOS_COMPLEXOS)]
    if simples:
        partes = ["count(*) AS total"]
        for i, c in enumerate(simples):
            partes += [
                f"count({_q(c.nome)}) AS nn_{i}",
                f"count(DISTINCT {_q(c.nome)}) AS d_{i}",
                f"min({_q(c.nome)}) AS mn_{i}",
                f"max({_q(c.nome)}) AS mx_{i}",
            ]
        r = fonte.consultar(f"SELECT {', '.join(partes)} FROM {base}")
        if r.erro:
            p.erros.append(f"estatísticas: {r.erro}")
        elif r.linhas:
            v = dict(zip(r.colunas, r.linhas[0]))
            total = _num(v.get("total")) or 0
            for i, c in enumerate(simples):
                nn = _num(v.get(f"nn_{i}")) or 0
                c.nulos_pct = round(100 * (1 - nn / total), 1) if total else None
                c.distintos = _num(v.get(f"d_{i}"))
                c.minimo, c.maximo = _curto(v.get(f"mn_{i}")), _curto(v.get(f"mx_{i}"))

    categoricas = [
        c for c in simples
        if _eh(c.tipo, TIPOS_CATEGORICOS) and c.distintos is not None and 0 < c.distintos <= 30
    ][:MAX_COLUNAS_TOP]
    for c in categoricas:
        r = fonte.consultar(
            f"SELECT {_q(c.nome)} AS valor, count(*) AS n FROM {base} GROUP BY {_q(c.nome)} ORDER BY n DESC LIMIT 10"
        )
        if not r.erro:
            c.top_valores = [[_curto(a), _num(b)] for a, b in r.linhas]

    r = fonte.consultar(f"SELECT * FROM {tabela} LIMIT 5")
    if r.erro:
        p.erros.append(f"amostra: {r.erro}")
    else:
        p.amostra_colunas = r.colunas
        p.amostra = [[_curto(x) for x in linha] for linha in r.linhas]
    log(f"   · {tabela}: {p.total_linhas if p.total_linhas is not None else '?'} linhas, {len(p.colunas)} colunas")
    return p


def perfil_texto(p: PerfilTabela) -> str:
    linhas = [f"TABELA {p.tabela}" + (f" — {p.comentario}" if p.comentario else "")]
    if p.total_linhas is not None:
        linhas.append(f"Linhas: {p.total_linhas}")
    linhas.append("Colunas:")
    for c in p.colunas:
        info = [c.tipo]
        if c.comentario:
            info.append(f'comentário: "{c.comentario}"')
        if c.nulos_pct is not None:
            info.append(f"nulos {c.nulos_pct}%")
        if c.distintos is not None:
            info.append(f"{c.distintos} distintos")
        if c.minimo is not None or c.maximo is not None:
            info.append(f"min={c.minimo} max={c.maximo}")
        if c.top_valores:
            info.append("valores: " + ", ".join(f"{v} ({n})" for v, n in c.top_valores))
        linhas.append(f"  - {c.nome}: " + "; ".join(str(x) for x in info))
    if p.amostra:
        linhas.append("Amostra:")
        linhas.append("  " + " | ".join(p.amostra_colunas))
        linhas += ["  " + " | ".join("NULL" if x is None else str(x) for x in l) for l in p.amostra]
    if p.erros:
        linhas.append("Observações: " + "; ".join(p.erros))
    return "\n".join(linhas)


# ------------------------------------------------------------------ estudo
ESTUDO_SISTEMA = """Você é um analista de dados sênior configurando um agente de BI conversacional
para um domínio de negócio. Você recebe a explicação do negócio escrita por um especialista
humano e o perfil REAL das tabelas (colunas, estatísticas, valores e amostra).

Produza o conhecimento que o agente vai usar. Seja específico ao que os dados mostram:
cite nomes reais de colunas, valores reais de status/categorias, granularidade, chaves de
junção e armadilhas (nulos, duplicidade, colunas de data, unidades). Não invente colunas.

Responda SOMENTE com um JSON válido, em português do Brasil:
{
  "nome": "nome curto do agente (ex.: Especialista em Crédito)",
  "descricao": "2-4 frases para o ROTEADOR saber quais perguntas este agente responde (assuntos, métricas, regras)",
  "exemplos": ["3 a 5 perguntas típicas que usuários de negócio fariam"],
  "perguntas_teste": ["5 perguntas DIFERENTES dos exemplos, para testar o roteamento"],
  "descricao_tabelas": {"catalogo.schema.tabela": "propósito, granularidade (1 linha = ?), colunas importantes e o significado de cada uma, valores possíveis, chaves de junção"},
  "instrucoes": "regras para o SQL: definição das métricas do negócio, filtros padrão, como tratar datas/nulos/status, joins corretos",
  "exemplos_sql": [{"pergunta": "...", "sql": "SELECT ... (Databricks SQL, nomes completos catalogo.schema.tabela, somente leitura)"}]
}
Gere de 3 a 6 exemplos_sql cobrindo as perguntas mais prováveis. Se não houver tabelas,
deixe descricao_tabelas e exemplos_sql vazios."""

ESTUDO_USUARIO = """Agente: {nome}

Explicação do negócio (escrita por quem configura):
{prompt}

{genie}
{documentos}
Perfil das tabelas:
{perfis}
"""

CORRECAO_SQL = """O SQL abaixo falhou. Corrija-o usando apenas as colunas existentes no perfil.
Responda SOMENTE com o SQL corrigido, sem explicação.

Pergunta: {pergunta}
SQL:
{sql}
Erro: {erro}

Perfil:
{perfis}"""


@dataclass
class EstudoAgente:
    nome: str = ""
    descricao: str = ""
    exemplos: List[str] = field(default_factory=list)
    perguntas_teste: List[str] = field(default_factory=list)
    descricao_tabelas: Dict[str, str] = field(default_factory=dict)
    instrucoes: str = ""
    exemplos_sql: List[Dict[str, str]] = field(default_factory=list)
    validacao_sql: List[Dict[str, Any]] = field(default_factory=list)
    perfis: List[Dict[str, Any]] = field(default_factory=list)
    avisos: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EstudoAgente":
        campos = cls.__dataclass_fields__
        return cls(**{k: v for k, v in (d or {}).items() if k in campos})


def _texto(msg) -> str:
    c = msg.content
    return c if isinstance(c, str) else "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)


def extrair_json(texto: str) -> Dict[str, Any]:
    texto = re.sub(r"```(?:json)?", "", texto)
    ini, fim = texto.find("{"), texto.rfind("}")
    if ini == -1 or fim == -1:
        raise ValueError("resposta sem JSON")
    return json.loads(texto[ini : fim + 1])


def _limpar_sql(texto: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", texto, re.S | re.I)
    return (m.group(1) if m else texto).strip().rstrip(";")


def _lista_str(v) -> List[str]:
    return [str(x).strip() for x in (v or []) if str(x).strip()]


def estudar_agente(
    llm: BaseChatModel,
    fonte: FonteEstudo,
    *,
    nome: str,
    prompt_negocio: str,
    tabelas: List[str],
    genie_space_id: Optional[str] = None,
    usa_documentos: bool = False,
    max_linhas: int = 200,
    log: Callable[[str], None] = print,
) -> EstudoAgente:
    tabelas = [t.lower() for t in tabelas]
    estudo = EstudoAgente(nome=nome)

    log(f"🔎 Estudando '{nome}'…")
    perfis = [perfilar_tabela(fonte, t, log) for t in tabelas]
    estudo.perfis = [asdict(p) for p in perfis]
    for p in perfis:
        estudo.avisos += [f"{p.tabela}: {e}" for e in p.erros]
    perfis_txt = "\n\n".join(perfil_texto(p) for p in perfis) or "(nenhuma tabela)"

    genie_txt = ""
    if genie_space_id:
        try:
            genie_txt = "Genie Space disponível para o agente:\n" + fonte.descrever_genie(genie_space_id) + "\n"
        except Exception as e:  # noqa: BLE001
            estudo.avisos.append(f"Genie {genie_space_id}: {e}")
    docs_txt = "O agente também terá documentação de regras de negócio (PDF/DOCX) indexada.\n" if usa_documentos else ""

    usuario = ESTUDO_USUARIO.format(
        nome=nome, prompt=prompt_negocio.strip() or "(não informado)", genie=genie_txt,
        documentos=docs_txt, perfis=perfis_txt,
    )
    dados: Dict[str, Any] = {}
    for tentativa in range(2):
        resp = llm.invoke([SystemMessage(ESTUDO_SISTEMA), HumanMessage(usuario)])
        try:
            dados = extrair_json(_texto(resp))
            break
        except Exception:  # noqa: BLE001
            usuario += "\n\nATENÇÃO: responda apenas com o JSON válido."
            if tentativa == 1:
                estudo.avisos.append("O LLM não devolveu JSON válido; preencha o conhecimento manualmente.")

    estudo.nome = nome or str(dados.get("nome") or "").strip()
    estudo.descricao = str(dados.get("descricao") or "").strip() or prompt_negocio.strip()[:400]
    estudo.exemplos = _lista_str(dados.get("exemplos"))[:5]
    estudo.perguntas_teste = _lista_str(dados.get("perguntas_teste"))[:8]
    estudo.instrucoes = str(dados.get("instrucoes") or "").strip()
    desc = dados.get("descricao_tabelas") or {}
    estudo.descricao_tabelas = {
        t: str(desc.get(t) or desc.get(t.split(".")[-1]) or "").strip() for t in tabelas
    }

    # ---- valida cada SQL executando de verdade
    for ex in (dados.get("exemplos_sql") or [])[:6]:
        if not isinstance(ex, dict) or not ex.get("sql"):
            continue
        pergunta, sql = str(ex.get("pergunta", "")).strip(), _limpar_sql(str(ex["sql"]))
        resultado = _validar_exemplo(llm, fonte, pergunta, sql, tabelas, max_linhas, perfis_txt)
        estudo.validacao_sql.append(resultado)
        if resultado["ok"]:
            estudo.exemplos_sql.append({"pergunta": pergunta, "sql": resultado["sql"]})
            log(f"   ✓ SQL validado: {pergunta}")
        else:
            log(f"   ✗ SQL descartado: {pergunta} — {resultado['erro']}")
    if tabelas and not estudo.exemplos_sql:
        estudo.avisos.append("Nenhum SQL de exemplo passou na validação; o agente vai depender só do perfil.")
    log(f"✅ '{nome}' estudado: {len(estudo.exemplos_sql)} SQL(s) validado(s), {len(estudo.perguntas_teste)} pergunta(s) de teste")
    return estudo


def _validar_exemplo(llm, fonte, pergunta, sql, tabelas, max_linhas, perfis_txt) -> Dict[str, Any]:
    erro = ""
    for tentativa in range(2):
        try:
            sql_final = validar_sql(sql, tabelas, max_linhas)
            r = fonte.consultar(sql_final)
            if not r.erro:
                return {"pergunta": pergunta, "sql": sql, "ok": True, "linhas": len(r.linhas),
                        "erro": None, "corrigido": tentativa > 0}
            erro = r.erro
        except SQLNaoPermitido as e:
            erro = f"bloqueado: {e}"
        if tentativa == 0:
            resp = llm.invoke([HumanMessage(CORRECAO_SQL.format(pergunta=pergunta, sql=sql, erro=erro, perfis=perfis_txt))])
            sql = _limpar_sql(_texto(resp))
    return {"pergunta": pergunta, "sql": sql, "ok": False, "linhas": 0, "erro": erro, "corrigido": True}
