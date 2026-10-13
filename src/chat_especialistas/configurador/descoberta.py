"""Descobre o que existe no workspace: LLMs, embeddings, Genie Spaces, SQL Warehouses e tabelas.

Roda no notebook com as credenciais de quem está configurando, então lista exatamente
o que essa pessoa pode usar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModeloIA:
    nome: str
    tarefa: str  # "chat" | "embeddings" | outra
    origem: str  # "foundation" | "externo" | "custom"
    pronto: bool
    detalhe: str = ""

    @property
    def rotulo(self) -> str:
        status = "" if self.pronto else " (não pronto)"
        extra = f" · {self.detalhe}" if self.detalhe else ""
        return f"{self.nome} [{self.origem}{extra}]{status}"


@dataclass
class GenieSpaceInfo:
    space_id: str
    titulo: str
    descricao: str = ""
    warehouse_id: Optional[str] = None

    @property
    def rotulo(self) -> str:
        return f"{self.titulo} ({self.space_id})"


@dataclass
class WarehouseInfo:
    id: str
    nome: str
    estado: str
    serverless: bool = False
    tamanho: str = ""

    @property
    def rotulo(self) -> str:
        tipo = "serverless" if self.serverless else self.tamanho
        return f"{self.nome} [{tipo}, {self.estado}]"


@dataclass
class Recursos:
    llms_chat: List[ModeloIA] = field(default_factory=list)
    embeddings: List[ModeloIA] = field(default_factory=list)
    genies: List[GenieSpaceInfo] = field(default_factory=list)
    warehouses: List[WarehouseInfo] = field(default_factory=list)
    avisos: List[str] = field(default_factory=list)


def _valor(enum_ou_str) -> str:
    return str(getattr(enum_ou_str, "value", enum_ou_str) or "")


def _tarefa(endpoint) -> str:
    t = (endpoint.task or "").lower()
    if "chat" in t or "completions" in t:
        return "chat"
    if "embedding" in t:
        return "embeddings"
    return t or "custom"


def _origem(endpoint) -> tuple[str, str]:
    entidades = (endpoint.config.served_entities if endpoint.config else None) or []
    for e in entidades:
        if e.external_model is not None:
            prov = _valor(getattr(e.external_model, "provider", "")) or "externo"
            return "externo", f"{prov}: {getattr(e.external_model, 'name', '')}"
        if e.foundation_model is not None:
            return "foundation", getattr(e.foundation_model, "display_name", "") or ""
        if e.entity_name:
            if e.entity_name.startswith("system.ai."):
                return "foundation", e.entity_name
            return "custom", e.entity_name
    if (endpoint.name or "").startswith("databricks-"):
        return "foundation", ""
    return "custom", ""


def listar_modelos(w) -> tuple[List[ModeloIA], List[ModeloIA]]:
    chat, emb = [], []
    for ep in w.serving_endpoints.list():
        tarefa = _tarefa(ep)
        origem, detalhe = _origem(ep)
        pronto = _valor(ep.state.ready if ep.state else "") == "READY"
        m = ModeloIA(ep.name, tarefa, origem, pronto, detalhe)
        if tarefa == "chat":
            chat.append(m)
        elif tarefa == "embeddings":
            emb.append(m)
    ordem = lambda m: (not m.pronto, m.origem != "foundation", m.nome)  # noqa: E731
    return sorted(chat, key=ordem), sorted(emb, key=ordem)


def listar_genies(w) -> List[GenieSpaceInfo]:
    espacos, token = [], None
    while True:
        resp = w.genie.list_spaces(page_size=100, page_token=token)
        for s in resp.spaces or []:
            espacos.append(GenieSpaceInfo(s.space_id, s.title or s.space_id, s.description or "", s.warehouse_id))
        token = resp.next_page_token
        if not token:
            break
    return sorted(espacos, key=lambda g: g.titulo.lower())


def listar_warehouses(w) -> List[WarehouseInfo]:
    lista = [
        WarehouseInfo(
            id=wh.id,
            nome=wh.name,
            estado=_valor(wh.state),
            serverless=bool(wh.enable_serverless_compute),
            tamanho=wh.cluster_size or "",
        )
        for wh in w.warehouses.list()
    ]
    return sorted(lista, key=lambda x: (not x.serverless, x.estado != "RUNNING", x.nome))


def mensagem_erro(e: Exception) -> str:
    """Resume erros do SDK; para falta de escopo, diz exatamente o que liberar na app."""
    import re

    texto = str(e)
    m = re.search(r"does not have required scopes?:\s*([\w.:\-, ]+?)\s*(\[|$)", texto)
    if m:
        escopos = m.group(1).strip()
        return (
            f"falta o escopo `{escopos}` na user authorization da app. Vá em Apps → (sua app) → "
            "Authorization → User authorization, adicione o escopo, salve e reabra a app em uma aba "
            "anônima (para autorizar de novo)."
        )
    return texto.split(" Config: ")[0][:300]


def descobrir(w) -> Recursos:
    """Lista tudo; uma falha (ex.: Genie desabilitado) vira aviso e não interrompe o resto."""
    r = Recursos()
    try:
        r.llms_chat, r.embeddings = listar_modelos(w)
    except Exception as e:  # noqa: BLE001
        r.avisos.append(f"Não foi possível listar endpoints de modelos: {mensagem_erro(e)}")
    try:
        r.genies = listar_genies(w)
    except Exception as e:  # noqa: BLE001
        r.avisos.append(f"Não foi possível listar Genie Spaces: {mensagem_erro(e)}")
    try:
        r.warehouses = listar_warehouses(w)
    except Exception as e:  # noqa: BLE001
        r.avisos.append(f"Não foi possível listar SQL Warehouses: {mensagem_erro(e)}")
    return r


# ----------------------------------------------------------- Unity Catalog
def listar_catalogos(w) -> List[str]:
    return sorted(c.name for c in w.catalogs.list() if c.name and c.name not in ("system", "__databricks_internal"))


def listar_schemas(w, catalogo: str) -> List[str]:
    return sorted(s.name for s in w.schemas.list(catalog_name=catalogo) if s.name != "information_schema")


def listar_tabelas(w, catalogo: str, schema: str) -> List[str]:
    return sorted(
        t.full_name
        for t in w.tables.list(catalog_name=catalogo, schema_name=schema, omit_columns=True, omit_properties=True)
    )
