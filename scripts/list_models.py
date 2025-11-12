"""List saved models in the repository `models/` directory.

Usage:
    python scripts/list_models.py

This script prints model filenames saved by `feedback_qnlp.utils.save_model`.
"""
import os
from feedback_qnlp.utils import list_models, ensure_models_dir


def main():
    base = os.getcwd()
    models_dir = ensure_models_dir(base)
    models = list_models(models_dir)
    if not models:
        print("No models found in:", models_dir)
        return 0
    print("Saved models:")
    for m in models:
        print(" ", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
