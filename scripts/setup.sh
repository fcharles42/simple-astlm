#!/usr/bin/env bash
# Run on each node before launching orchestrator.py. Use `source`, not `bash` —
# bash runs it in a subshell so the venv activation wouldn't stick afterward:
#   source /scratch/cscs/ibadanin/projects/simple-astlm/scripts/setup.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[setup] repo: $REPO_ROOT"

if [ ! -d "$HOME/astlm-env" ]; then
    echo "[setup] creating venv at ~/astlm-env"
    python3 -m venv "$HOME/astlm-env"
fi

# shellcheck disable=SC1091
source "$HOME/astlm-env/bin/activate"

echo "[setup] installing pinned requirements"
python3 -m pip install --upgrade pip
python3 -m pip install -r "$REPO_ROOT/requirements.txt"

echo "[setup] done. venv active, cwd = $REPO_ROOT"
echo "[setup] next: python3 scripts/orchestrator.py"

set +euo pipefail  # this script is meant to be sourced — don't leave strict mode on in the interactive shell
