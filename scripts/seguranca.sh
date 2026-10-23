#!/usr/bin/env bash
# Verificação de segurança do projeto (rodar antes de cada deploy e no CI).
#   pip install -r requirements-dev.txt && scripts/seguranca.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/3 Testes de segurança (SQL, isolamento, LGPD, credenciais, repositório)"
python -m pytest -q tests/test_seguranca_sql.py tests/test_seguranca_privacidade.py tests/test_seguranca_repositorio.py

echo "== 2/3 Análise estática do código (bandit: severidade média ou maior)"
python -m bandit -q -r src app.py agent.py -ll

echo "== 3/3 Vulnerabilidades conhecidas nas dependências instaladas (pip-audit; precisa de internet)"
# Audita o ambiente atual: instale antes requirements.txt e requirements-agente.txt (veja o README).
python -m pip_audit --skip-editable

echo "OK: nenhuma falha de segurança encontrada."
