"""Proteções transversais: nada de credencial em mensagens de erro, logs ou telas."""
from __future__ import annotations

import re

# `campo` (quando existe) é mantido para o erro continuar compreensível: "password=***".
_SEGREDOS = [
    re.compile(r"dapi[0-9a-f]{32}(?:-\d+)?", re.I),                                   # PAT do Databricks
    re.compile(r"(?P<campo>bearer\s+)[A-Za-z0-9\-_.~+/=]{12,}", re.I),                 # header Authorization
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}"),       # JWT (tokens OAuth)
    re.compile(r"(?P<campo>(?:access_?token|client_?secret|password|senha|token)\s*[=:]\s*['\"]?)[^\s'\",&]{6,}", re.I),
    re.compile(r"AKIA[0-9A-Z]{16}"),                                                   # chave de acesso AWS
]


_PARTE_NOME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_\-]{0,254}$")


def nome_seguro(nome: str, partes: int = 1) -> str:
    """Valida um nome do Unity Catalog (catalogo / schema / catalogo.schema.tabela) antes de ir para um SQL.
    Só letras, números, _ e -: impede fechar a crase, emendar outra instrução ou comentar o resto."""
    pedacos = (nome or "").strip().split(".")
    if len(pedacos) != partes or not all(_PARTE_NOME.match(p) for p in pedacos):
        raise ValueError(f"nome inválido: {nome!r} (esperado {partes} parte(s) com letras, números, _ ou -)")
    return ".".join(pedacos)


def citar(nome: str, partes: int = 3) -> str:
    """`catalogo`.`schema`.`tabela`, já validado."""
    return ".".join(f"`{p}`" for p in nome_seguro(nome, partes).split("."))


def _trocar(m: re.Match) -> str:
    return (m.groupdict().get("campo") or "") + "***"


def ocultar_segredos(texto: str) -> str:
    """Troca tokens, senhas e chaves por ***."""
    texto = str(texto or "")
    for padrao in _SEGREDOS:
        texto = padrao.sub(_trocar, texto)
    return texto
