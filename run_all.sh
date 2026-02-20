#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

source ".venv/bin/activate"

python -m pip install --upgrade pip >/dev/null
pip install -r requirements.txt >/dev/null

if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

if [[ -z "${GROQ_API_KEY:-}" ]]; then
  echo "GROQ_API_KEY is missing. Set it in .env or export it in shell."
  exit 1
fi

echo "Starting AI Finance Agent on http://localhost:8501"
exec streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
