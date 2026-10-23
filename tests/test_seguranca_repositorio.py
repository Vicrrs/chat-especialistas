"""Higiene do repositório: nenhuma credencial versionada e nenhuma configuração insegura por padrão."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

RAIZ = Path(__file__).resolve().parents[1]

PADROES = {
    "token do Databricks": re.compile(r"dapi[0-9a-f]{32}"),
    "chave AWS": re.compile(r"AKIA[0-9A-Z]{16}"),
    "chave privada": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----"),
    "JWT": re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    "conexão com senha": re.compile(r"\w+://[^\s/:@]+:[^\s/@]{4,}@"),
    "senha/segredo literal": re.compile(r"(?i)\b(password|senha|client_secret|api_key|access_token)\b\s*[:=]\s*['\"][^'\"\s]{6,}['\"]"),
}
# Só este arquivo, que contém os próprios padrões. Credenciais falsas de outros testes devem ser
# montadas em tempo de execução, senão o push protection do GitHub recusa o push.
IGNORAR = {"tests/test_seguranca_repositorio.py"}


def _arquivos():
    try:
        # Versionados + novos ainda não commitados (sem os ignorados pelo .gitignore).
        nomes = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=RAIZ, capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        nomes = [str(p.relative_to(RAIZ)) for p in RAIZ.rglob("*") if p.is_file() and ".git" not in p.parts]
    return [n for n in nomes if n not in IGNORAR and (RAIZ / n).is_file()]


def test_nenhuma_credencial_versionada():
    achados = []
    for nome in _arquivos():
        try:
            texto = (RAIZ / nome).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for tipo, padrao in PADROES.items():
            for m in padrao.finditer(texto):
                linha = texto.count("\n", 0, m.start()) + 1
                achados.append(f"{nome}:{linha} ({tipo})")
    assert not achados, "Possíveis credenciais no repositório:\n" + "\n".join(achados)


@pytest.mark.parametrize("padrao", [".env", ".streamlit/secrets.toml", "*.pem", "*.key", "*.p12", ".databrickscfg"])
def test_arquivos_de_segredo_ignorados_pelo_git(padrao):
    nome = padrao.replace("*", "teste")
    r = subprocess.run(["git", "check-ignore", "-q", nome], cwd=RAIZ)
    assert r.returncode == 0, f"{nome} não está no .gitignore"


def test_app_yaml_sem_credenciais_e_sem_identidade_insegura():
    app = yaml.safe_load((RAIZ / "app.yaml").read_text(encoding="utf-8"))
    for var in app.get("env", []):
        nome = var["name"].upper()
        assert not any(p in nome for p in ("TOKEN", "SECRET", "PASSWORD", "SENHA")) or "valueFrom" in var, (
            f"{var['name']} deve vir de um secret do Databricks (valueFrom), não do app.yaml"
        )
    assert {v["name"]: v.get("value") for v in app["env"]}.get("IDENTIDADE_APP") in ("auto", "app")


def test_configuracao_padrao_limita_os_agentes():
    cfg = yaml.safe_load((RAIZ / "config" / "agents.yaml").read_text(encoding="utf-8"))
    bi = cfg["bi"]
    assert bi["autenticacao"] in ("servico", "usuario")
    assert 0 < bi["max_linhas"] <= 10_000 and 0 < bi["timeout_segundos"] <= 300 and 0 < bi["max_iteracoes"] <= 20
    for esp in cfg["especialistas"]:
        for t in (esp.get("dados") or {}).get("tabelas", []):
            assert t.count(".") == 2 and "*" not in t, f"{esp['id']}: tabela deve ser catalogo.schema.tabela explícita"
