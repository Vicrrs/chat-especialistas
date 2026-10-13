"""Avaliação do roteamento, registro no MLflow/Unity Catalog e deploy do agente geral.

Usado pelo assistente (00_COMECE_AQUI) e pelo notebook 02.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage

from .config import AppConfig
from .graph import detectar_intencao

PACOTES_EXCLUIDOS = ("pytest", "pypdf", "python-docx", "ipywidgets", "#")


# --------------------------------------------------------------- avaliação
def casos_de_teste(cfg: AppConfig) -> List[Tuple[str, List[str]]]:
    """Perguntas de teste geradas no estudo (+ exemplos) com o agente esperado."""
    casos: List[Tuple[str, List[str]]] = []
    for e in cfg.especialistas:
        for p in (e.perguntas_teste or e.exemplos):
            casos.append((p, [e.id]))
    casos.append(("Bom dia! Tudo bem?", []))
    return casos


def avaliar_roteamento(
    cfg: AppConfig,
    llm_roteador,
    casos: Optional[List[Tuple[str, List[str]]]] = None,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Roda só o Agente 01 (detector de intenção) em cada pergunta de teste."""
    casos = casos if casos is not None else casos_de_teste(cfg)
    linhas, acertos = [], 0
    for pergunta, esperado in casos:
        r = detectar_intencao(cfg, llm_roteador, [HumanMessage(pergunta)])
        ok = sorted(r["especialistas"]) == sorted(esperado)
        acertos += ok
        linhas.append({
            "pergunta": pergunta, "esperado": esperado, "obtido": r["especialistas"],
            "intencao": r["intencao"], "confianca": r["confianca"], "acertou": ok,
        })
        log(f"   {'✓' if ok else '✗'} {pergunta}  →  {r['especialistas'] or r['intencao']}")
    acuracia = acertos / len(casos) if casos else 0.0
    log(f"Acurácia do roteamento: {acertos}/{len(casos)} ({acuracia:.0%})")
    return {"acuracia": acuracia, "acertos": acertos, "total": len(casos), "casos": linhas}


# ----------------------------------------------------------------- recursos
def recursos_mlflow(cfg: AppConfig) -> Dict[str, Any]:
    """Recursos que o endpoint precisa acessar (o Databricks provisiona as credenciais)."""
    from mlflow.models.resources import (
        DatabricksGenieSpace,
        DatabricksServingEndpoint,
        DatabricksSQLWarehouse,
        DatabricksTable,
        DatabricksVectorSearchIndex,
    )

    endpoints = {cfg.llm.endpoint_roteador, cfg.llm.endpoint_especialistas}
    recursos: list = []
    if cfg.usa_documentos:
        endpoints.add(cfg.vector_search.modelo_embedding)
        indices = {cfg.vector_search.indice_padrao} | {e.indice for e in cfg.especialistas if e.indice and e.documentos}
        recursos += [DatabricksVectorSearchIndex(index_name=i) for i in indices]
    recursos += [DatabricksServingEndpoint(endpoint_name=e) for e in endpoints]
    recursos += [DatabricksSQLWarehouse(warehouse_id=w) for w in {cfg.warehouse_de(e) for e in cfg.especialistas if e.dados} if w]
    recursos += [DatabricksTable(table_name=t) for t in sorted({t for e in cfg.especialistas if e.dados for t in e.dados.tabelas})]
    genies = sorted({e.genie_space_id for e in cfg.especialistas if e.genie_space_id})
    recursos += [DatabricksGenieSpace(genie_space_id=g) for g in genies]
    # Genie executa SQL no warehouse do próprio space; o endpoint também precisa dele.

    if cfg.bi.autenticacao == "usuario":
        from mlflow.models.auth_policy import AuthPolicy, SystemAuthPolicy, UserAuthPolicy

        return {
            "auth_policy": AuthPolicy(
                system_auth_policy=SystemAuthPolicy(resources=recursos),
                user_auth_policy=UserAuthPolicy(
                    api_scopes=["sql.statement-execution", "sql.warehouses", "dashboards.genie", "catalog.tables"]
                ),
            )
        }
    return {"resources": recursos}


def dependencias(raiz: Path) -> List[str]:
    with open(Path(raiz) / "requirements-agente.txt", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip() and not l.strip().startswith(PACOTES_EXCLUIDOS)]


# -------------------------------------------------------- registro e deploy
def registrar_e_publicar(
    cfg: AppConfig,
    raiz: Path,
    *,
    caminho_config: Optional[Path] = None,
    experimento: Optional[str] = None,
    publicar: bool = True,
    validar: bool = True,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    import mlflow

    raiz = Path(raiz).resolve()
    caminho_config = Path(caminho_config or raiz / "config" / "agents.yaml").resolve()
    os.environ["CHAT_ESPECIALISTAS_CONFIG"] = str(caminho_config)
    if experimento:
        mlflow.set_experiment(experimento)

    exemplo = {"input": [{"role": "user", "content": "Olá!"}]}
    log("📦 Registrando o agente no MLflow…")
    with mlflow.start_run(run_name="chat_especialistas") as run:
        info = mlflow.pyfunc.log_model(
            name="agent",
            python_model=str(raiz / "agent.py"),
            code_paths=[str(raiz / "src" / "chat_especialistas"), str(caminho_config)],
            pip_requirements=dependencias(raiz),
            input_example=exemplo,
            **recursos_mlflow(cfg),
        )
        mlflow.log_dict(cfg.model_dump(by_alias=True), "config_especialistas.json")
        mlflow.log_artifact(str(caminho_config), "config")

    if validar:
        log("🧪 Validando o modelo empacotado (ambiente isolado)…")
        mlflow.models.predict(model_uri=info.model_uri, input_data={"input": [{"role": "user", "content": "Olá!"}]}, env_manager="uv")

    mlflow.set_registry_uri("databricks-uc")
    versao = mlflow.register_model(info.model_uri, cfg.projeto.nome_modelo)
    log(f"✅ Registrado: {cfg.projeto.nome_modelo} versão {versao.version}")
    saida: Dict[str, Any] = {
        "run_id": run.info.run_id, "model_uri": info.model_uri,
        "nome_modelo": cfg.projeto.nome_modelo, "versao": versao.version,
    }

    if publicar:
        from databricks import agents

        log("🚀 Publicando no Model Serving (leva alguns minutos para ficar pronto)…")
        dep = agents.deploy(cfg.projeto.nome_modelo, versao.version, tags={"projeto": "chat_especialistas"}, scale_to_zero=True)
        saida.update(endpoint=dep.endpoint_name, review_app=getattr(dep, "review_app_url", None))
        log(f"✅ Endpoint: {dep.endpoint_name}")
        if saida.get("review_app"):
            log(f"🔗 Review App: {saida['review_app']}")
    return saida
