#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname "$0")/.." && pwd)

DRY_RUN=0
INCLUDE_RAW_DERIVED=0
INCLUDE_SYNTH_RAW=0

usage() {
  cat <<'EOF'
Usage: bash scripts/clean_generated.sh [--dry-run] [--include-raw-derived] [--include-synth-raw]

Default cleanup:
  - outputs/
  - data/processed/
  - .pytest_cache/
  - __pycache__/ and *.pyc

Options:
  --dry-run             Print what would be removed without deleting.
  --include-raw-derived Also remove data/raw/*/points.json and mark_meta.json.
  --include-synth-raw   Also remove data/raw/case_0001 (synthetic case).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --include-raw-derived) INCLUDE_RAW_DERIVED=1; shift ;;
    --include-synth-raw) INCLUDE_SYNTH_RAW=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

rm_path() {
  local p="$1"
  if [[ -e "$p" ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "[dry-run] rm -rf $p"
    else
      rm -rf "$p"
      echo "[removed] $p"
    fi
  fi
}

echo "[clean] root: $ROOT_DIR"

# Core generated outputs
rm_path "$ROOT_DIR/outputs"
rm_path "$ROOT_DIR/data/processed"
rm_path "$ROOT_DIR/.pytest_cache"

# Python caches
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] find $ROOT_DIR -type d -name __pycache__ -prune -exec rm -rf {} +"
  echo "[dry-run] find $ROOT_DIR -type f -name '*.pyc' -delete"
else
  find "$ROOT_DIR" -type d -name __pycache__ -prune -exec rm -rf {} +
  find "$ROOT_DIR" -type f -name "*.pyc" -delete
fi

# Optional: raw-derived files
if [[ "$INCLUDE_RAW_DERIVED" -eq 1 ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] rm -f $ROOT_DIR/data/raw/*/points.json"
    echo "[dry-run] rm -f $ROOT_DIR/data/raw/*/mark_meta.json"
  else
    rm -f "$ROOT_DIR"/data/raw/*/points.json
    rm -f "$ROOT_DIR"/data/raw/*/mark_meta.json
    echo "[removed] data/raw/*/points.json, data/raw/*/mark_meta.json"
  fi
fi

# Optional: synthetic raw case
if [[ "$INCLUDE_SYNTH_RAW" -eq 1 ]]; then
  rm_path "$ROOT_DIR/data/raw/case_0001"
fi

echo "[clean] done"
