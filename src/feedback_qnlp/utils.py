import os
from pathlib import Path
import joblib


def ensure_models_dir(base_dir):
    models_dir = Path(base_dir) / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def save_model(model, models_dir, name=None):
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    if name is None:
        import uuid
        name = f"model_{uuid.uuid4().hex}.joblib"
    path = models_dir / name
    joblib.dump(model, path)
    return str(path)


def list_models(models_dir):
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []
    return [str(p.name) for p in models_dir.glob('*.joblib')]
