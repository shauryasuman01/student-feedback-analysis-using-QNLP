"""Minimal training script.

Usage:
    python scripts/train.py --data data/student_feedback.csv

This will run a simple classical baseline. If lambeq/pennylane are installed and you want to experiment, call the functions in src.feedback_qnlp.model.
"""
import argparse
import pandas as pd
from feedback_qnlp.preprocess import load_data, preprocess_texts, vectorize_texts
from feedback_qnlp.model import train_classical_baseline


def main(data_path):
    df = load_data(data_path)
    texts = preprocess_texts(df['feedback'].astype(str).tolist())
    X, vec = vectorize_texts(texts)
    y = (df['label'] == 'positive').astype(int).tolist()
    result = train_classical_baseline(X, y)
    print(f"Classical baseline accuracy: {result['accuracy']:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', dest='data', default='data/student_feedback.csv')
    args = parser.parse_args()
    main(args.data)
