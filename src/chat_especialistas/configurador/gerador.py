"""Transforma as escolhas do assistente (modelos, Genies, agentes, estudos) no config/agents.yaml."""
from __future__ import annotations

import json
import re
import shutil
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..config import AppConfig
from .estudo import EstudoAgente

PRECO_LLAMA_70B = {"input": 0.50, "output": 1.50}


@dataclass
class Selecao:
    llm_roteador: str = ""
    llm_especialistas: str = ""
    embedding: str = ""
    warehouse_id: str = ""
    genies: List[str] = field(default_factory=list)  # Genie Spaces liberados para os agentes
    catalogo: str = "main"
    schema: str = "chat_especialistas"
    vector_search_endpoint: str = "vs_chat_especialistas"
    autenticacao: str = "servico"
    nome_assistente: str = "Assistente"
    prompt_geral: str = ""
    precos: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class AgenteRascunho:
    nome: str
    prompt_negocio: str = ""
    tabelas: List[str] = field(default_factory=list)
    genie_space_id: Optional[str] = None
    documentos: bool = False
    id: str = ""
    estudo: Optional[Dict[str, Any]] = None  # EstudoAgente.to_dict()

    def __post_init__(self):
        self.id = self.id or slug(self.nome)
        self.tabelas = sorted({t.strip().lower() for t in self.tabelas if t.strip()})

    @property
    def estudado(self) -> bool:
        return bool(self.estudo)


def slug(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto or "").encode("ascii", "ignore").decode().lower()
    t = re.sub(r"[^a-z0-9]+", "_", t).strip("_")
    t = re.sub(r"^(especialista|agente)_(em_|de_|do_|da_)?", "", t) or t
    return (t or "agente")[:40]


PROMPT_GERAL_PADRAO = """Você é o {nome}, o assistente de dados e regras de negócio da empresa.
Você conversa com o usuário, entende o contexto da conversa e encaminha cada pergunta
ao agente especialista correto. Seja cordial, objetivo e responda sempre em português do Brasil."""

PROMPT_AGENTE = """{prompt_negocio}

Você é o {nome}. Responda perguntas sobre esse domínio consultando os dados e as regras
de negócio. Seja preciso com números e transparente sobre premissas."""


def _ids_unicos(agentes: List[AgenteRascunho]) -> None:
    vistos: Dict[str, int] = {}
    for a in agentes:
        base = a.id
        if base in vistos:
            vistos[base] += 1
            a.id = f"{base}_{vistos[base]}"
        else:
            vistos[base] = 1


def gerar_config(sel: Selecao, agentes: List[AgenteRascunho], base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Monta o dicionário do agents.yaml e valida com o mesmo modelo usado em produção."""
    if not agentes:
        raise ValueError("Crie ao menos um agente.")
    faltando = [n for n, v in (("LLM do roteador", sel.llm_roteador), ("LLM dos especialistas", sel.llm_especialistas)) if not v]
    if faltando:
        raise ValueError("Selecione: " + ", ".join(faltando))
    if any(a.tabelas for a in agentes) and not sel.warehouse_id:
        raise ValueError("Selecione um SQL Warehouse (há agentes com tabelas).")
    if any(a.documentos for a in agentes) and not sel.embedding:
        raise ValueError("Selecione um modelo de embeddings (há agentes com documentação).")
    _ids_unicos(agentes)

    base = base or {}
    llmops = dict(base.get("llmops") or {})
    precos = {"databricks-meta-llama-3-3-70b-instruct": PRECO_LLAMA_70B, **(llmops.get("precos_usd_por_milhao") or {})}
    precos.update(sel.precos)
    cat, sch = sel.catalogo, sel.schema

    especialistas = []
    for a in agentes:
        e = EstudoAgente.from_dict(a.estudo) if a.estudo else EstudoAgente()
        nome = a.nome.strip() or e.nome or a.id
        item: Dict[str, Any] = {
            "id": a.id,
            "nome": nome,
            "descricao": (e.descricao or a.prompt_negocio or nome).strip(),
            "exemplos": e.exemplos,
            "perguntas_teste": e.perguntas_teste,
            "prompt": PROMPT_AGENTE.format(prompt_negocio=a.prompt_negocio.strip(), nome=nome).strip(),
            "documentos": bool(a.documentos),
        }
        if a.tabelas:
            item["dados"] = {
                "tabelas": a.tabelas,
                "instrucoes": e.instrucoes,
                "exemplos_sql": e.exemplos_sql,
                "descricao_tabelas": {t: e.descricao_tabelas.get(t, "") for t in a.tabelas},
            }
        if a.genie_space_id:
            item["genie_space_id"] = a.genie_space_id
        especialistas.append(item)

    dados = {
        "projeto": {"catalogo": cat, "schema": sch, "nome_modelo": f"{cat}.{sch}.chat_especialistas"},
        "llm": {
            **(base.get("llm") or {}),
            "endpoint_roteador": sel.llm_roteador,
            "endpoint_especialistas": sel.llm_especialistas,
        },
        "vector_search": {
            **(base.get("vector_search") or {}),
            "endpoint": sel.vector_search_endpoint,
            "indice_padrao": f"{cat}.{sch}.documentos_index",
            "tabela_origem": f"{cat}.{sch}.documentos_chunks",
            "modelo_embedding": sel.embedding or (base.get("vector_search") or {}).get("modelo_embedding", "databricks-gte-large-en"),
        },
        "llmops": {
            **llmops,
            "arquivo_jsonl": f"/Volumes/{cat}/{sch}/llmops/interacoes_dev.jsonl",
            "precos_usd_por_milhao": precos,
        },
        "bi": {**(base.get("bi") or {}), "warehouse_id": sel.warehouse_id or None, "autenticacao": sel.autenticacao},
        "roteamento": base.get("roteamento") or {"limiar_confianca": 0.6, "janela_historico": 12},
        "agente_principal": {
            "nome": sel.nome_assistente,
            "prompt": (sel.prompt_geral.strip() or PROMPT_GERAL_PADRAO.format(nome=sel.nome_assistente)).strip(),
        },
        "especialistas": especialistas,
    }
    AppConfig.model_validate(dados)  # mesmo validador usado pelo agente em produção
    return dados


class _Dumper(yaml.SafeDumper):
    pass


def _str_multilinha(dumper, valor: str):
    estilo = "|" if "\n" in valor else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", valor, style=estilo)


_Dumper.add_representer(str, _str_multilinha)


def salvar_config(dados: Dict[str, Any], caminho: Path) -> Optional[Path]:
    """Grava o YAML; o anterior vira agents.yaml.<data>.bak. Devolve o caminho do backup."""
    caminho = Path(caminho)
    backup = None
    if caminho.exists():
        backup = caminho.with_name(f"{caminho.name}.{datetime.now():%Y%m%d_%H%M%S}.bak")
        shutil.copy2(caminho, backup)
    cabecalho = (
        "# Gerado pelo assistente (notebooks/00_COMECE_AQUI).\n"
        "# Pode editar à mão; rodar o assistente de novo sobrescreve (com backup).\n"
    )
    texto = yaml.dump(dados, Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=110)
    caminho.write_text(cabecalho + texto, encoding="utf-8")
    return backup


# ------------------------------------------------------- estado do assistente
def salvar_estado(caminho: Path, sel: Selecao, agentes: List[AgenteRascunho]) -> None:
    Path(caminho).write_text(
        json.dumps({"selecao": asdict(sel), "agentes": [asdict(a) for a in agentes]}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def carregar_estado(caminho: Path) -> tuple[Selecao, List[AgenteRascunho]]:
    caminho = Path(caminho)
    if not caminho.exists():
        return Selecao(), []
    d = json.loads(caminho.read_text(encoding="utf-8"))
    campos_sel = Selecao.__dataclass_fields__
    campos_ag = AgenteRascunho.__dataclass_fields__
    sel = Selecao(**{k: v for k, v in d.get("selecao", {}).items() if k in campos_sel})
    agentes = [AgenteRascunho(**{k: v for k, v in a.items() if k in campos_ag}) for a in d.get("agentes", [])]
    return sel, agentes
