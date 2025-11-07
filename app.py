import os
import uuid
import pandas as pd
import re
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import joblib

UPLOAD_FOLDER = 'uploads'
GENERATED_FOLDER = 'static/generated'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_FOLDER'] = GENERATED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

# small sentiment lexicon
POSITIVE_WORDS = {'good','great','helpful','clear','excellent','enjoyed','learned','useful','patient'}
NEGATIVE_WORDS = {'bad','terrible','boring','unclear','overwhelmed','slow','lack','lackluster','confusing'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_columns(headers):
    """Try to detect commonly named columns from a list of headers.

    Returns a dict with keys: feedback, label, teacher, facility, branch, department
    Values are header names or None.
    """
    def norm(s):
        return re.sub(r'[^a-z0-9]', '', s.lower())

    mapping = {'feedback': None, 'label': None, 'teacher': None, 'facility': None, 'branch': None, 'department': None}
    norms = {h: norm(h) for h in headers}

    for h, nh in norms.items():
        if mapping['feedback'] is None and ('feedback' in nh or 'comment' in nh or 'remark' in nh):
            mapping['feedback'] = h
        if mapping['label'] is None and ('label' in nh or 'sentiment' in nh or 'rating' in nh):
            mapping['label'] = h
        if mapping['teacher'] is None and 'teacher' in nh:
            mapping['teacher'] = h
        if mapping['facility'] is None and ('facility' in nh or 'laboratory' in nh or 'library' in nh):
            mapping['facility'] = h
        if mapping['branch'] is None and 'branch' in nh:
            mapping['branch'] = h
        if mapping['department'] is None and 'department' in nh:
            mapping['department'] = h

    # If no explicit feedback column found, pick the first long text-like column
    if mapping['feedback'] is None:
        # choose header with longest name or that contains 'feedback' parts
        sorted_by_len = sorted(headers, key=lambda x: -len(str(x)))
        mapping['feedback'] = sorted_by_len[0] if sorted_by_len else None

    return mapping


def rule_based_sentiment(text):
    if not isinstance(text, str):
        return 0
    text = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in text)
    neg = sum(1 for w in NEGATIVE_WORDS if w in text)
    if pos >= neg and (pos+neg) > 0:
        return 1
    if pos == 0 and neg == 0:
        # neutral -> treat as negative for conservative approach
        return 0
    return 0


def plot_bar(series, title, filename):
    # keep PNG fallback for compatibility
    plt.figure(figsize=(8, max(4, 0.5 * len(series))))
    sns.barplot(x=series.values, y=series.index, palette='viridis')
    plt.xlabel('Positive ratio')
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(app.config['GENERATED_FOLDER'], filename)
    plt.savefig(path)
    plt.close()
    return path


def plot_interactive(series, title, filename_html):
    # series: pandas Series with index labels and numeric values
    df = series.reset_index()
    df.columns = ['label', 'value']
    fig = px.bar(df, x='value', y='label', orientation='h', title=title, labels={'value':'Positive ratio', 'label':''})
    out_path = os.path.join(app.config['GENERATED_FOLDER'], filename_html)
    pio.write_html(fig, file=out_path, include_plotlyjs='cdn', full_html=True)
    return out_path


def plot_stacked(pos_series, title, filename_html):
    """Create a horizontal stacked bar showing Positive vs Negative percentages.

    pos_series: pandas Series indexed by label with values in [0,1] for positive ratio.
    """
    import pandas as _pd
    df = _pd.DataFrame({'label': pos_series.index, 'Positive': pos_series.values})
    df['Negative'] = 1.0 - df['Positive']
    long = df.melt(id_vars='label', value_vars=['Positive', 'Negative'], var_name='sentiment', value_name='value')
    fig = px.bar(long, x='value', y='label', color='sentiment', orientation='h', title=title, labels={'value':'Ratio','label':''}, barmode='stack', color_discrete_map={'Positive':'#2ca02c','Negative':'#d62728'})
    out_path = os.path.join(app.config['GENERATED_FOLDER'], filename_html)
    pio.write_html(fig, file=out_path, include_plotlyjs='cdn', full_html=True)
    return out_path


def plot_overall_pos_neg(positive_count, negative_count, title, filename_html):
    """Create a simple two-column Plotly bar chart showing Positive vs Negative percentages.

    positive_count / negative_count can be absolute counts (integers).
    """
    total = float(positive_count + negative_count) if (positive_count + negative_count) > 0 else 1.0
    pos_pct = (positive_count / total) * 100.0
    neg_pct = (negative_count / total) * 100.0
    labels = ['Positive', 'Negative']
    values = [pos_pct, neg_pct]
    import pandas as _pd
    df = _pd.DataFrame({'sentiment': labels, 'pct': values})
    fig = px.bar(df, x='sentiment', y='pct', title=title, labels={'pct':'Percent (%)', 'sentiment':''}, color='sentiment', color_discrete_map={'Positive':'#2ca02c','Negative':'#d62728'})
    fig.update_layout(yaxis=dict(range=[0,100]))
    out_path = os.path.join(app.config['GENERATED_FOLDER'], filename_html)
    pio.write_html(fig, file=out_path, include_plotlyjs='cdn', full_html=True)
    return out_path


def plot_grouped_pos_neg(pos_series, title, filename_html):
    """Create grouped bars (Positive, Negative) per label.

    pos_series: pandas Series indexed by label with values in [0,1] for positive ratio.
    """
    import pandas as _pd
    df = _pd.DataFrame({'label': list(pos_series.index), 'Positive': list(pos_series.values)})
    df['Negative'] = 1.0 - df['Positive']
    long = df.melt(id_vars='label', value_vars=['Positive', 'Negative'], var_name='sentiment', value_name='value')
    # express as percent
    long['pct'] = long['value'] * 100.0
    fig = px.bar(long, x='label', y='pct', color='sentiment', barmode='group', title=title, labels={'pct':'Percent (%)','label':''}, color_discrete_map={'Positive':'#2ca02c','Negative':'#d62728'})
    fig.update_layout(xaxis={'categoryorder':'total descending'}, yaxis=dict(range=[0,100]))
    out_path = os.path.join(app.config['GENERATED_FOLDER'], filename_html)
    pio.write_html(fig, file=out_path, include_plotlyjs='cdn', full_html=True)
    return out_path


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique = f"{uuid.uuid4().hex}_{filename}"
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        file.save(saved_path)
        # read headers
        try:
            df_sample = pd.read_csv(saved_path, nrows=5)
        except Exception as e:
            return f"Error reading CSV: {e}", 400
    headers = list(df_sample.columns)
    defaults = detect_columns(headers)
    # auto-process using detected defaults (skip manual mapping)
    df_full = pd.read_csv(saved_path)
    feedback_col = defaults.get('feedback')
    label_col = defaults.get('label')
    teacher_col = defaults.get('teacher')
    facility_col = defaults.get('facility')
    branch_col = defaults.get('branch')
    department_col = defaults.get('department')
    # call processing helper
    return process_dataframe(df_full, feedback_col=feedback_col, label_col=label_col, teacher_col=teacher_col, facility_col=facility_col, branch_col=branch_col, department_col=department_col, saved_model_path=None)


@app.route('/process', methods=['POST'])
def process_file():
    # read form or json and call processing helper
    filename = request.form.get('filename')
    if not filename:
        return 'Missing filename', 400
    feedback_col = request.form.get('feedback_col')
    label_col = request.form.get('label_col') or None
    teacher_col = request.form.get('teacher_col') or None
    facility_col = request.form.get('facility_col') or None
    branch_col = request.form.get('branch_col') or None
    department_col = request.form.get('department_col') or None
    use_model = request.form.get('use_model') or None
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(path):
        return 'File not found', 404
    df = pd.read_csv(path)
    return process_dataframe(df, feedback_col=feedback_col, label_col=label_col, teacher_col=teacher_col, facility_col=facility_col, branch_col=branch_col, department_col=department_col, use_model=use_model)


def process_dataframe(df, feedback_col=None, label_col=None, teacher_col=None, facility_col=None, branch_col=None, department_col=None, use_model=None, saved_model_path=None):
    """Process a dataframe and return rendered results template. Extracted from process_file."""
    if feedback_col not in df.columns:
        # fallback: try detect
        detected = detect_columns(list(df.columns))
        feedback_col = detected.get('feedback')
        if feedback_col not in df.columns:
            return f'Feedback column {feedback_col} missing', 400
    df['clean_feedback'] = df[feedback_col].astype(str)

    accuracy = None
    metrics = None
    saved_model = None
    if use_model:
        import joblib
        model_path = os.path.join('models', use_model)
        if os.path.exists(model_path):
            loaded = joblib.load(model_path)
            if isinstance(loaded, dict) and 'model' in loaded and 'vectorizer' in loaded:
                vec = loaded['vectorizer']
                clf = loaded['model']
                X = vec.transform(df['clean_feedback'])
                preds = clf.predict(X)
                df['predicted'] = preds
            else:
                df['predicted'] = df['clean_feedback'].map(rule_based_sentiment)
        else:
            df['predicted'] = df['clean_feedback'].map(rule_based_sentiment)
    elif label_col and label_col in df.columns:
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
        from feedback_qnlp.utils import save_model, ensure_models_dir
        models_dir = ensure_models_dir(os.path.join(os.getcwd()))
        model_blob = {'model': clf, 'vectorizer': vec}
        saved_model = save_model(model_blob, models_dir)
    else:
        df['predicted'] = df['clean_feedback'].map(rule_based_sentiment)

    agg_results = {}
    if teacher_col and teacher_col in df.columns:
        teacher_group = df.groupby(teacher_col)['predicted'].mean().sort_values(ascending=False)
        tfile = f"teacher_{uuid.uuid4().hex}.html"
        tpath = plot_grouped_pos_neg(teacher_group, 'Teacher positive vs negative (%)', tfile)
        agg_results['teacher_plot'] = tfile
        agg_results['teacher_stats'] = teacher_group.to_dict()
        # also create an overall teacher positive vs negative (two columns)
        teacher_mask = df[teacher_col].notna()
        t_positive = int(df.loc[teacher_mask, 'predicted'].sum())
        t_total = int(teacher_mask.sum())
        t_negative = t_total - t_positive
        tover_file = f"teacher_overall_{uuid.uuid4().hex}.html"
        tover_path = plot_overall_pos_neg(t_positive, t_negative, 'Teacher overall Positive vs Negative', tover_file)
        agg_results['teacher_overall_plot'] = tover_file
    if facility_col and facility_col in df.columns:
        facility_group = df.groupby(facility_col)['predicted'].mean().sort_values(ascending=False)
        ffile = f"facility_{uuid.uuid4().hex}.html"
        fpath = plot_grouped_pos_neg(facility_group, 'Facility positive vs negative (%)', ffile)
        agg_results['facility_plot'] = ffile
        agg_results['facility_stats'] = facility_group.to_dict()
        # overall facility positive vs negative
        fac_mask = df[facility_col].notna()
        f_positive = int(df.loc[fac_mask, 'predicted'].sum())
        f_total = int(fac_mask.sum())
        f_negative = f_total - f_positive
        fover_file = f"facility_overall_{uuid.uuid4().hex}.html"
        fover_path = plot_overall_pos_neg(f_positive, f_negative, 'Facility overall Positive vs Negative', fover_file)
        agg_results['facility_overall_plot'] = fover_file

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
            category_scores[c] = float(preds.mean()) if len(preds) > 0 else 0.0
    import pandas as _pd
    cat_series = _pd.Series(category_scores).sort_values(ascending=False)
    cfile = f"categories_{uuid.uuid4().hex}.html"
    # show categories as grouped positive/negative percentages
    cpath = plot_grouped_pos_neg(cat_series, 'Category positive vs negative (%)', cfile)
    agg_results['category_plot'] = cfile
    agg_results['category_stats'] = category_scores

    if branch_col and branch_col in df.columns:
        branch_group = df.groupby(branch_col)['predicted'].mean().sort_values(ascending=False)
        bfile = f"branch_{uuid.uuid4().hex}.html"
        bpath = plot_grouped_pos_neg(branch_group, 'Branch positive vs negative (%)', bfile)
        agg_results['branch_plot'] = bfile
        agg_results['branch_stats'] = branch_group.to_dict()
    if department_col and department_col in df.columns:
        dept_group = df.groupby(department_col)['predicted'].mean().sort_values(ascending=False)
        dfile = f"department_{uuid.uuid4().hex}.html"
        dpath = plot_grouped_pos_neg(dept_group, 'Department positive vs negative (%)', dfile)
        agg_results['department_plot'] = dfile
        agg_results['department_stats'] = dept_group.to_dict()

    total = len(df)
    positive = int(df['predicted'].sum())
    negative = total - positive
    # overall
    ofile = f"overall_{uuid.uuid4().hex}.html"
    opath = plot_overall_pos_neg(positive, negative, 'Overall Positive vs Negative', ofile)
    agg_results['overall_plot'] = ofile

    return render_template('results.html', accuracy=accuracy, total=total, positive=positive, negative=negative, agg=agg_results, metrics=metrics, saved_model_path=saved_model)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
