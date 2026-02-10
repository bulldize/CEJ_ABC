"""CEJ MVP package."""
import sys
from pathlib import Path


def _ensure_local_monai():
    root = Path(__file__).resolve().parents[1]
    monai_dir = root / "MONAI"
    if monai_dir.exists():
        monai_path = str(monai_dir)
        if monai_path not in sys.path:
            sys.path.insert(0, monai_path)


_ensure_local_monai()
