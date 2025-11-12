"""Minimal training script.

Usage:
    python scripts/train.py --data data/student_feedback.csv

This will run a simple classical baseline. If lambeq/pennylane are installed and you want to experiment, call the functions in src.feedback_qnlp.model.
"""

import argparse

import pandas as pd

from feedback_qnlp.model import train_classical_baseline, train_quantum_model
from feedback_qnlp.preprocess import load_data, preprocess_texts, vectorize_texts


def main(data_path, quantum: bool = False):
    df = load_data(data_path)
    texts = preprocess_texts(df["feedback"].astype(str).tolist())
    y = (df["label"] == "positive").astype(int).tolist()
    if quantum:
        print("Running quantum (or simulated) training...")
        result = train_quantum_model(texts, y)
        print(f"Quantum pipeline accuracy: {result['accuracy']:.3f} (quantum={result.get('quantum')})")
    else:
        X, vec = vectorize_texts(texts)
        result = train_classical_baseline(X, y)
        print(f"Classical baseline accuracy: {result['accuracy']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", dest="data", default="data/student_feedback.csv")
    parser.add_argument("--quantum", dest="quantum", action="store_true", help="Run the quantum pipeline (or a simulated fallback if deps missing)")
    args = parser.parse_args()
    main(args.data, quantum=args.quantum)
