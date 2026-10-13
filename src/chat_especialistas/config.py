"""Carregamento e validação da configuração (config/agents.yaml)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class LLMConfig(BaseModel):
    endpoint_roteador: str
    endpoint_especialistas: str
    temperatura_roteador: float = 0.0
    temperatura_especialistas: float = 0.2
    max_tokens: int = 1500


class VectorSearchConfig(BaseModel):
    endpoint: str
    indice_padrao: str
    tabela_origem: str = "main.chat_especialistas.documentos_chunks"
    modelo_embedding: str = "databricks-gte-large-en"
    colunas: List[str] = ["chunk_id", "especialista", "fonte", "conteudo"]
    coluna_texto: str = "conteudo"
    k: int = 5


class RoteamentoConfig(BaseModel):
    limiar_confianca: float = 0.6
    janela_historico: int = 12


class PrecoConfig(BaseModel):
    input: float = 0.0
    output: float = 0.0


class LLMOpsConfig(BaseModel):
    habilitado: bool = True
    versao_agente: str = "dev"
    destinos: List[str] = ["log", "mlflow_trace"]
    arquivo_jsonl: Optional[str] = None
    registrar_textos: bool = True
    precos_usd_por_milhao: Dict[str, PrecoConfig] = Field(default_factory=dict)
    preco_padrao_usd_por_milhao: PrecoConfig = PrecoConfig(input=1.0, output=3.0)

    @field_validator("destinos")
    @classmethod
    def _destinos_validos(cls, v: List[str]) -> List[str]:
        validos = {"log", "mlflow_trace", "jsonl"}
        invalidos = set(v) - validos
        if invalidos:
            raise ValueError(f"destinos inválidos em llmops: {invalidos} (use {validos})")
        return v

    def preco(self, modelo: str) -> PrecoConfig:
        return self.precos_usd_por_milhao.get(modelo, self.preco_padrao_usd_por_milhao)


class AgentePrincipalConfig(BaseModel):
    nome: str = "Assistente"
    prompt: str


class BIConfig(BaseModel):
    warehouse_id: Optional[str] = None
    autenticacao: str = "servico"  # "servico" | "usuario"
    max_linhas: int = 200
    linhas_para_o_llm: int = 50
    timeout_segundos: int = 50
    max_iteracoes: int = 8

    @field_validator("autenticacao")
    @classmethod
    def _auth(cls, v: str) -> str:
        if v not in ("servico", "usuario"):
            raise ValueError("bi.autenticacao deve ser 'servico' ou 'usuario'")
        return v


class ExemploSQL(BaseModel):
    pergunta: str
    sql: str


class DadosConfig(BaseModel):
    tabelas: List[str]
    warehouse_id: Optional[str] = None  # sobrescreve bi.warehouse_id
    instrucoes: str = ""
    exemplos_sql: List[ExemploSQL] = Field(default_factory=list)
    # Gerado pelo estudo das tabelas: o que cada tabela/coluna significa para o negócio.
    descricao_tabelas: Dict[str, str] = Field(default_factory=dict)

    @field_validator("tabelas")
    @classmethod
    def _qualificadas(cls, v: List[str]) -> List[str]:
        for t in v:
            if t.count(".") != 2:
                raise ValueError(f"tabela '{t}' deve ser catalogo.schema.tabela")
        return [t.lower() for t in v]


class EspecialistaConfig(BaseModel):
    id: str
    nome: str
    descricao: str
    prompt: str
    exemplos: List[str] = Field(default_factory=list)
    indice: Optional[str] = None  # sobrescreve vector_search.indice_padrao
    k: Optional[int] = None
    dados: Optional[DadosConfig] = None
    genie_space_id: Optional[str] = None
    max_iteracoes: Optional[int] = None
    # Usa documentação indexada no Vector Search? (false = só dados/Genie)
    documentos: bool = True
    # Perguntas geradas no estudo, usadas para avaliar o roteamento antes do deploy.
    perguntas_teste: List[str] = Field(default_factory=list)

    @property
    def eh_bi(self) -> bool:
        return self.dados is not None or self.genie_space_id is not None

    @model_validator(mode="after")
    def _tem_fonte(self):
        if not (self.documentos or self.eh_bi):
            raise ValueError(f"especialista '{self.id}' precisa de documentos, tabelas ou Genie")
        return self

    @field_validator("id")
    @classmethod
    def _id_valido(cls, v: str) -> str:
        if not v.replace("_", "").isalnum() or not v.islower():
            raise ValueError(f"id de especialista inválido: '{v}' (use minúsculas, números e _)")
        return v


class ProjetoConfig(BaseModel):
    """Onde o projeto grava o que é dele: chunks, índice, LLMOps e o modelo registrado."""
    catalogo: Optional[str] = None
    schema_: Optional[str] = Field(default=None, alias="schema")
    nome_modelo: Optional[str] = None

    model_config = {"populate_by_name": True}


class AppConfig(BaseModel):
    projeto: ProjetoConfig = ProjetoConfig()
    llm: LLMConfig
    vector_search: VectorSearchConfig
    roteamento: RoteamentoConfig = RoteamentoConfig()
    llmops: LLMOpsConfig = LLMOpsConfig()
    bi: BIConfig = BIConfig()
    agente_principal: AgentePrincipalConfig
    especialistas: List[EspecialistaConfig]

    @field_validator("especialistas")
    @classmethod
    def _ids_unicos(cls, v: List[EspecialistaConfig]) -> List[EspecialistaConfig]:
        ids = [e.id for e in v]
        if len(ids) != len(set(ids)):
            raise ValueError("ids de especialistas duplicados")
        if not v:
            raise ValueError("configure ao menos um especialista")
        return v

    @model_validator(mode="after")
    def _projeto_padrao(self):
        cat, sch, _ = (self.vector_search.tabela_origem.split(".") + ["", "", ""])[:3]
        self.projeto.catalogo = self.projeto.catalogo or cat
        self.projeto.schema_ = self.projeto.schema_ or sch
        self.projeto.nome_modelo = self.projeto.nome_modelo or f"{self.projeto.catalogo}.{self.projeto.schema_}.chat_especialistas"
        return self

    @property
    def usa_documentos(self) -> bool:
        return any(e.documentos for e in self.especialistas)

    def warehouse_de(self, esp: EspecialistaConfig) -> Optional[str]:
        return (esp.dados and esp.dados.warehouse_id) or self.bi.warehouse_id

    def especialista(self, id_: str) -> EspecialistaConfig:
        for e in self.especialistas:
            if e.id == id_:
                return e
        raise KeyError(id_)


def _caminho_padrao() -> Path:
    env = os.environ.get("CHAT_ESPECIALISTAS_CONFIG")
    if env:
        return Path(env)
    # src/chat_especialistas/config.py -> raiz do projeto/config/agents.yaml
    candidato = Path(__file__).resolve().parents[2] / "config" / "agents.yaml"
    if candidato.exists():
        return candidato
    # Quando empacotado pelo MLflow (code_paths), o yaml fica numa pasta do sys.path.
    import sys

    for pasta in sys.path:
        candidato = Path(pasta or ".") / "agents.yaml"
        if candidato.exists():
            return candidato
    raise FileNotFoundError(
        "agents.yaml não encontrado. Defina CHAT_ESPECIALISTAS_CONFIG com o caminho do arquivo."
    )


def carregar_config(caminho: str | Path | None = None) -> AppConfig:
    caminho = Path(caminho) if caminho else _caminho_padrao()
    with open(caminho, encoding="utf-8") as f:
        dados = yaml.safe_load(f)
    return AppConfig.model_validate(dados)
