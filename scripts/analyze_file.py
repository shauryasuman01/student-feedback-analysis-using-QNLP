#!/usr/bin/env python3
"""Simple analyzer to run locally on a CSV and report per-row probabilities and predicted labels.

Usage:
  python scripts\analyze_file.py "C:\path\to\file.csv" --feedback-col Feedback --label-col label --threshold 0.6

If --label-col is omitted or not present, the script will run the rule-based classifier and list any disagreements.
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np

from app import detect_columns, rule_based_sentiment


def train_and_predict(df, feedback_col, label_col, threshold=0.6):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split

    vec = CountVectorizer(max_features=2000)
    X_all = vec.fit_transform(df[feedback_col].astype(str))
    y_all = (df[label_col].astype(str) == 'positive').astype(int)

    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)

    model_for_proba = clf
    try:
        calib = CalibratedClassifierCV(base_estimator=clf, method='sigmoid', cv='prefit')
        calib.fit(X_val, y_val)
        model_for_proba = calib
    except Exception as e:
        print('Calibration failed, continuing with raw classifier:', e)

    try:
        probs_all = model_for_proba.predict_proba(X_all)[:, 1]
    except Exception:
        probs_all = clf.predict_proba(X_all)[:, 1]

    preds_all = np.where(probs_all >= threshold, 'positive', 'negative')

    out = df[[feedback_col]].copy()
    out['prob_positive'] = probs_all
    out['predicted_label'] = preds_all
    if label_col in df.columns:
        out['gold_label'] = df[label_col].astype(str)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('csv', help='Path to CSV file')
    p.add_argument('--feedback-col', default=None, help='Name of feedback text column (auto-detected if omitted)')
    p.add_argument('--label-col', default=None, help='Name of label column (optional)')
    p.add_argument('--threshold', type=float, default=0.6, help='Probability threshold for positive/negative (default 0.6)')
    p.add_argument('--out', default=None, help='Optional path to save per-row report CSV')
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print('File not found:', args.csv)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    fb_col = args.feedback_col
    if not fb_col or fb_col not in df.columns:
        detected = detect_columns(list(df.columns))
        fb_col = detected.get('feedback')
        print('Autodetected feedback column:', fb_col)
    if not fb_col or fb_col not in df.columns:
        print('Could not find a feedback column. Columns present:', df.columns.tolist())
        sys.exit(1)

    if args.label_col and args.label_col in df.columns:
        print('Running ML pipeline (train+calibrate) using label column:', args.label_col)
        report = train_and_predict(df, fb_col, args.label_col, threshold=args.threshold)
    else:
        print('Label column not provided or not found. Running rule-based sentiment and listing positives.\n')
        report = df[[fb_col]].copy()
        report['rule_sentiment'] = report[fb_col].map(rule_based_sentiment)
        # show which ones would be positive under the rule-based classifier
        positives = report[report['rule_sentiment'] == 1]
        print(f'Rule-based positives: {len(positives)} / {len(report)}')
        if not positives.empty:
            print('\nExamples predicted positive by rule-based classifier:')
            print(positives.head(10))
        # save and exit
        if args.out:
            report.to_csv(args.out, index=False)
            print('Saved report to', args.out)
        sys.exit(0)

    # summarize
    total = len(report)
    pos = (report['predicted_label'] == 'positive').sum()
    neg = (report['predicted_label'] == 'negative').sum()
    print(f'Total rows: {total} | positive: {pos} | negative: {neg}')

    # show top examples that the model called positive
    if pos > 0:
        print('\nExamples predicted positive (top by prob):')
        print(report.sort_values('prob_positive', ascending=False).head(10))

    if args.out:
        report.to_csv(args.out, index=False)
        print('Saved report to', args.out)
