import os
import sys
import uuid
import pandas as pd
from pathlib import Path

# Import helpers from app without starting the server
import importlib.util
spec = importlib.util.spec_from_file_location("app_module", os.path.join(os.path.dirname(__file__), '..', 'app.py'))
app_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_mod)

rule_based_sentiment = app_mod.rule_based_sentiment
plot_bar = app_mod.plot_bar
plot_interactive = app_mod.plot_interactive
plot_stacked = app_mod.plot_stacked
plot_overall_pos_neg = app_mod.plot_overall_pos_neg
detect_columns = app_mod.detect_columns
plot_grouped_pos_neg = app_mod.plot_grouped_pos_neg


def process(path):
    path = Path(path)
    if not path.exists():
        print('File not found:', path)
        return 1
    df = pd.read_csv(path)
    headers = list(df.columns)
    defaults = detect_columns(headers)
    feedback_col = defaults.get('feedback')
    label_col = defaults.get('label')
    teacher_col = defaults.get('teacher')
    facility_col = defaults.get('facility')
    branch_col = defaults.get('branch')
    department_col = defaults.get('department')
    print('Detected columns:')
    for k,v in [('feedback', feedback_col), ('label', label_col), ('teacher', teacher_col), ('facility', facility_col), ('branch', branch_col), ('department', department_col)]:
        print(f'  {k}: {v}')

    df['clean_feedback'] = df[feedback_col].astype(str)

    accuracy = None
    metrics = None
    saved_model = None
    if label_col and label_col in df.columns:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

        vec = CountVectorizer(max_features=2000)
        X_all = vec.fit_transform(df['clean_feedback'])
        y_all = (df[label_col].astype(str) == 'positive').astype(int)
        X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        acc = accuracy_score(y_val, preds)
        p, r, f, _ = precision_recall_fscore_support(y_val, preds, average='binary', zero_division=0)
        cm = confusion_matrix(y_val, preds).tolist()
        metrics = {'accuracy': float(acc), 'precision': float(p), 'recall': float(r), 'f1': float(f), 'confusion_matrix': cm}
        df['predicted'] = clf.predict(X_all)
        # save model
        from feedback_qnlp.utils import ensure_models_dir, save_model
        models_dir = ensure_models_dir(os.path.join(os.getcwd()))
        model_blob = {'model': clf, 'vectorizer': vec}
        saved_model = save_model(model_blob, models_dir)
    else:
        df['predicted'] = df['clean_feedback'].map(rule_based_sentiment)

    generated = []
    # teacher
    if teacher_col and teacher_col in df.columns:
        teacher_group = df.groupby(teacher_col)['predicted'].mean().sort_values(ascending=False)
        tfile = f"teacher_{uuid.uuid4().hex}.html"
        path = plot_grouped_pos_neg(teacher_group, 'Teacher positive vs negative (%)', tfile)
        generated.append(path)
        # overall teacher positive vs negative
        teacher_mask = df[teacher_col].notna()
        t_positive = int(df.loc[teacher_mask, 'predicted'].sum())
        t_total = int(teacher_mask.sum())
        t_negative = t_total - t_positive
        tover = f"teacher_overall_{uuid.uuid4().hex}.html"
        topath = plot_overall_pos_neg(t_positive, t_negative, 'Teacher overall Positive vs Negative', tover)
        generated.append(topath)
    # facility
    if facility_col and facility_col in df.columns:
        facility_group = df.groupby(facility_col)['predicted'].mean().sort_values(ascending=False)
        ffile = f"facility_{uuid.uuid4().hex}.html"
        path = plot_grouped_pos_neg(facility_group, 'Facility positive vs negative (%)', ffile)
        generated.append(path)
        # overall facility positive vs negative
        fac_mask = df[facility_col].notna()
        f_positive = int(df.loc[fac_mask, 'predicted'].sum())
        f_total = int(fac_mask.sum())
        f_negative = f_total - f_positive
        fover = f"facility_overall_{uuid.uuid4().hex}.html"
        fopath = plot_overall_pos_neg(f_positive, f_negative, 'Facility overall Positive vs Negative', fover)
        generated.append(fopath)
    # categories
    category_columns = [
        'Teacher Feedback', 'Course Content', 'Examination pattern',
        'Laboratory Library Facilities', 'Extra Co-Curricular Activities', 'Any other suggestion'
    ]
    present_categories = [c for c in category_columns if c in df.columns]
    if present_categories:
        category_scores = {}
        for c in present_categories:
            texts = df[c].astype(str).fillna('')
            preds = texts.map(rule_based_sentiment)
            category_scores[c] = float(preds.mean()) if len(preds)>0 else 0.0
        import pandas as _pd
        cat_series = _pd.Series(category_scores).sort_values(ascending=False)
    cfile = f"categories_{uuid.uuid4().hex}.html"
    path = plot_grouped_pos_neg(cat_series, 'Category positive vs negative (%)', cfile)
    generated.append(path)
    # branch/department
    if branch_col and branch_col in df.columns:
        branch_group = df.groupby(branch_col)['predicted'].mean().sort_values(ascending=False)
    bfile = f"branch_{uuid.uuid4().hex}.html"
    path = plot_grouped_pos_neg(branch_group, 'Branch positive vs negative (%)', bfile)
    generated.append(path)
    if department_col and department_col in df.columns:
        dept_group = df.groupby(department_col)['predicted'].mean().sort_values(ascending=False)
    dfile = f"department_{uuid.uuid4().hex}.html"
    path = plot_grouped_pos_neg(dept_group, 'Department positive vs negative (%)', dfile)
    generated.append(path)

    total = len(df)
    positive = int(df['predicted'].sum())
    negative = total - positive
    # overall positive vs negative two-column chart
    ofile = f"overall_{uuid.uuid4().hex}.html"
    opath = plot_overall_pos_neg(positive, negative, 'Overall Positive vs Negative', ofile)
    generated.append(opath)
    print('\nSummary:')
    print(' total rows:', total)
    print(' positive (predicted):', positive)
    print(' negative (predicted):', negative)
    if metrics is not None:
        print(' training accuracy:', metrics.get('accuracy'))
        print(' precision:', metrics.get('precision'))
        print(' recall:', metrics.get('recall'))
        print(' f1:', metrics.get('f1'))
        print(' confusion matrix:', metrics.get('confusion_matrix'))
    if saved_model:
        print('\nSaved model:', saved_model)
    if generated:
        print('\nGenerated plots:')
        for p in generated:
            print(' ', p)
    else:
        print('No plots generated (no grouping columns found).')
    return 0

if __name__ == '__main__':
    candidate = None
    # prefer uploads first
    up_dir = Path(os.path.join(os.path.dirname(__file__), '..', 'uploads'))
    up_dir = up_dir.resolve()
    if up_dir.exists():
        files = list(up_dir.glob('*.csv'))
        if files:
            candidate = files[0]
    if not candidate:
        candidate = Path(os.path.join(os.path.dirname(__file__), '..', 'data', 'student_feedback.csv'))
    sys.exit(process(candidate))
