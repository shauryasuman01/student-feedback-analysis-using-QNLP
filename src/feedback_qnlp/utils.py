import os
from pathlib import Path
from typing import List

import joblib


def ensure_models_dir(base_dir: str) -> Path:
    """Ensure a `models/` directory exists within `base_dir` and return Path."""
    models_dir = Path(base_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def save_model(model, models_dir: os.PathLike | str, name: str | None = None) -> str:
    """Save `model` (any joblib-serializable object) under `models_dir`.

    Returns the string path to the saved file.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    if name is None:
        import uuid

        name = f"model_{uuid.uuid4().hex}.joblib"
    path = models_dir / name
    joblib.dump(model, path)
    return str(path)


def list_models(models_dir: os.PathLike | str) -> List[str]:
    """List joblib models in `models_dir` and return filenames."""
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []
    return [str(p.name) for p in models_dir.glob("*.joblib")]
