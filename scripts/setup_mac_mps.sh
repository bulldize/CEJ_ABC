#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "[WARN] This script is intended for macOS. Continuing anyway."
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
  echo "[WARN] Expected Apple Silicon arm64, got: $ARCH"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-mac.txt

python - <<'PY'
import platform
import torch
print("python", platform.python_version())
print("platform", platform.platform())
print("torch", torch.__version__)
print("mps_available", bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()))
print("cuda_available", torch.cuda.is_available())
PY
