"""Busca de documentação de cada especialista no Databricks Vector Search."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol

from .config import AppConfig, EspecialistaConfig


@dataclass
class Trecho:
    conteudo: str
    fonte: str
    score: float | None = None


class Retriever(Protocol):
    def buscar(self, pergunta: str, especialista: EspecialistaConfig) -> List[Trecho]: ...


class DatabricksRetriever:
    """Usa um índice compartilhado filtrado pela coluna `especialista`.

    Se o especialista tiver `indice` próprio no YAML, usa esse índice sem filtro.
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._stores: Dict[str, object] = {}

    def _store(self, indice: str):
        if indice not in self._stores:
            from databricks_langchain import DatabricksVectorSearch

            vs = self.cfg.vector_search
            self._stores[indice] = DatabricksVectorSearch(
                endpoint=vs.endpoint,
                index_name=indice,
                columns=vs.colunas,
            )
        return self._stores[indice]

    def buscar(self, pergunta: str, especialista: EspecialistaConfig) -> List[Trecho]:
        vs = self.cfg.vector_search
        indice = especialista.indice or vs.indice_padrao
        filtro = None if especialista.indice else {"especialista": especialista.id}
        k = especialista.k or vs.k

        docs = self._store(indice).similarity_search_with_score(pergunta, k=k, filter=filtro)
        return [
            Trecho(
                conteudo=doc.page_content,
                fonte=str(doc.metadata.get("fonte", "documento")),
                score=float(score) if score is not None else None,
            )
            for doc, score in docs
        ]


def formatar_contexto(trechos: List[Trecho]) -> str:
    if not trechos:
        return "(nenhum trecho encontrado)"
    return "\n\n".join(f"[{t.fonte}]\n{t.conteudo}" for t in trechos)
