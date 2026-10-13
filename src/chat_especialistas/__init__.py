"""Chat multi-especialistas (LangGraph + Databricks).

Imports preguiçosos: a app Streamlit usa só o configurador e não precisa carregar o LangGraph.
"""
from .config import AppConfig, carregar_config

__all__ = ["AppConfig", "carregar_config", "construir_grafo", "construir_grafo_databricks"]


def __getattr__(nome):
    if nome in ("construir_grafo", "construir_grafo_databricks"):
        from . import graph

        return getattr(graph, nome)
    raise AttributeError(nome)
