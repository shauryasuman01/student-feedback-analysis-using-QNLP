import os
import re
import uuid

import matplotlib
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

matplotlib.use('Agg')
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import seaborn as sns

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
POSITIVE_WORDS = {
    'good',
    'great',
    'helpful',
    'clear',
    'excellent',
    'enjoyed',
    'learned',
    'useful',
    'patient',
    'friendly',
    'supportive',
    'interesting',
    'engaging',
    'improved',
}
NEGATIVE_WORDS = {
    'bad',
    'terrible',
    'boring',
    'unclear',
    'overwhelmed',
    'slow',
    'lack',
    'lackluster',
    'confusing',
    'poor',
    'difficult',
    'hard',
    'stressful',
    'problem',
    'issues',
    'issue',
    'average',
    'late',
    'delay',
    'delayed',
    'missing',
    'need',
    'needs',
    'improve',
    'improvement',
    'disappointing',
    'disappointed',
    'worst',
    'strict',
}
NEGATIVE_PHRASES = {
    'not good',
    'not satisfied',
    'not clear',
    'not helpful',
    'not enough',
    'not adequate',
    'no proper',
    'could be better',
    'needs improvement',
    'need improvement',
    'need to improve',
    'needs to improve',
    'should improve',
    'more practical',
    'more practicle',
    'more practice',
    'need more',
    'needs more',
    'lack of',
    'improvement needed',
}


def safe_remove(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    except OSError:
        return


def sentiment_word_counts(text):
    if not isinstance(text, str):
        return 0, 0
    lowered = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in lowered)
    neg = sum(1 for w in NEGATIVE_WORDS if w in lowered)
    for phrase in NEGATIVE_PHRASES:
        if phrase in lowered:
            neg += 1
    for w in POSITIVE_WORDS:
        if any(f"{negator} {w}" in lowered for negator in ('not', 'no', 'never')):
            if pos > 0:
                pos -= 1
            neg += 1
    return pos, neg


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
    pos, neg = sentiment_word_counts(text)
    if pos >= neg and (pos + neg) > 0:
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


def build_plot_payload(fig):
    """Return both inline HTML and serializable Plotly dict for a figure."""
    return {
        'html': pio.to_html(fig, include_plotlyjs='cdn', full_html=False),
        'figure': fig.to_dict(),
    }


def _cors_json(payload, status=200):
    """Create a JSON or empty response with permissive CORS headers."""
    if status == 204:
        response = app.response_class(status=204)
    else:
        response = jsonify(payload)
        response.status_code = status
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    return response


def plot_grouped_pos_neg_payload(pos_series, title):
    """Return grouped Positive/Negative chart payload."""
    import pandas as _pd

    df = _pd.DataFrame({'label': list(pos_series.index), 'Positive': list(pos_series.values)})
    df['Negative'] = 1.0 - df['Positive']
    long = df.melt(id_vars='label', value_vars=['Positive', 'Negative'], var_name='sentiment', value_name='value')
    long['pct'] = long['value'] * 100.0
    fig = px.bar(
        long,
        x='label',
        y='pct',
        color='sentiment',
        barmode='group',
        title=title,
        labels={'pct': 'Percent (%)', 'label': ''},
        color_discrete_map={'Positive': '#2ca02c', 'Negative': '#d62728'},
    )
    fig.update_layout(xaxis={'categoryorder': 'total descending'}, yaxis=dict(range=[0, 100]))
    return build_plot_payload(fig)


def plot_overall_pos_neg_payload(positive_count, negative_count, title):
    total = float(positive_count + negative_count) if (positive_count + negative_count) > 0 else 1.0
    pos_pct = (positive_count / total) * 100.0
    neg_pct = (negative_count / total) * 100.0
    labels = ['Positive', 'Negative']
    values = [pos_pct, neg_pct]
    import pandas as _pd

    df = _pd.DataFrame({'sentiment': labels, 'pct': values})
    fig = px.bar(
        df,
        x='sentiment',
        y='pct',
        title=title,
        labels={'pct': 'Percent (%)', 'sentiment': ''},
        color='sentiment',
        color_discrete_map={'Positive': '#2ca02c', 'Negative': '#d62728'},
    )
    fig.update_layout(yaxis=dict(range=[0, 100]))
    return build_plot_payload(fig)


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
            safe_remove(saved_path)
            return f"Error reading CSV: {e}", 400
    headers = list(df_sample.columns)
    defaults = detect_columns(headers)
    # auto-process using detected defaults (skip manual mapping)
    try:
        df_full = pd.read_csv(saved_path)
    except Exception as e:
        safe_remove(saved_path)
        return f"Error reading CSV: {e}", 400
    feedback_col = defaults.get('feedback')
    label_col = defaults.get('label')
    teacher_col = defaults.get('teacher')
    facility_col = defaults.get('facility')
    branch_col = defaults.get('branch')
    department_col = defaults.get('department')
    try:
        response = process_dataframe(
            df_full,
            feedback_col=feedback_col,
            label_col=label_col,
            teacher_col=teacher_col,
            facility_col=facility_col,
            branch_col=branch_col,
            department_col=department_col,
            saved_model_path=None,
        )
    finally:
        safe_remove(saved_path)
    return response


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
    try:
        df = pd.read_csv(path)
    except Exception as e:
        safe_remove(path)
        return f"Error reading CSV: {e}", 400
    try:
        response = process_dataframe(
            df,
            feedback_col=feedback_col,
            label_col=label_col,
            teacher_col=teacher_col,
            facility_col=facility_col,
            branch_col=branch_col,
            department_col=department_col,
            use_model=use_model,
        )
    finally:
        safe_remove(path)
    return response


def generate_analysis(df, feedback_col=None, label_col=None, teacher_col=None, facility_col=None, branch_col=None, department_col=None, use_model=None, saved_model_path=None):
    """Produce analysis artifacts for a feedback dataframe."""

    df = df.copy()
    if feedback_col not in df.columns:
        detected = detect_columns(list(df.columns))
        feedback_col = detected.get('feedback')
        if feedback_col not in df.columns:
            raise ValueError(f'Feedback column {feedback_col} missing')

    df['clean_feedback'] = df[feedback_col].astype(str)

    accuracy = None
    metrics = None
    saved_model = saved_model_path
    model_used = 'rule_based'

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
                model_used = 'pretrained'
            else:
                df['predicted'] = df['clean_feedback'].map(rule_based_sentiment)
        else:
            df['predicted'] = df['clean_feedback'].map(rule_based_sentiment)
    elif label_col and label_col in df.columns:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            precision_recall_fscore_support,
        )
        from sklearn.model_selection import train_test_split

        vec = CountVectorizer(max_features=2000)
        X_all = vec.fit_transform(df['clean_feedback'])
        y_all = (df[label_col].astype(str).str.lower() == 'positive').astype(int)
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42
        )
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        acc = accuracy_score(y_val, preds)
        p, r, f, _ = precision_recall_fscore_support(
            y_val, preds, average='binary', zero_division=0
        )
        cm = confusion_matrix(y_val, preds).tolist()
        metrics = {
            'accuracy': float(acc),
            'precision': float(p),
            'recall': float(r),
            'f1': float(f),
            'confusion_matrix': cm,
        }
        accuracy = float(acc)
        df['predicted'] = clf.predict(X_all)
        from feedback_qnlp.utils import ensure_models_dir, save_model

        models_dir = ensure_models_dir(os.path.join(os.getcwd()))
        model_blob = {'model': clf, 'vectorizer': vec}
        saved_model = save_model(model_blob, models_dir)
        model_used = 'trained'
    else:
        df['predicted'] = df['clean_feedback'].map(rule_based_sentiment)

    total = int(len(df))
    positive = int(df['predicted'].sum())
    negative = int(total - positive)
    pos_pct = (positive / total * 100.0) if total else 0.0
    neg_pct = (negative / total * 100.0) if total else 0.0

    chart_html = {}
    charts = {}
    stats = {}

    if teacher_col and teacher_col in df.columns:
        teacher_mask = df[teacher_col].notna()
        teacher_group = (
            df.loc[teacher_mask]
            .groupby(teacher_col)['predicted']
            .mean()
            .sort_values(ascending=False)
        )
        if not teacher_group.empty:
            payload = plot_grouped_pos_neg_payload(
                teacher_group,
                'Positive vs Negative Feedback by Teacher',
            )
            chart_html['teacher_plot_html'] = payload['html']
            charts['teacher_sentiment'] = payload['figure']
            stats['teacher'] = {k: float(v) for k, v in teacher_group.items()}
        teacher_pos = int(df.loc[teacher_mask, 'predicted'].sum())
        teacher_total = int(teacher_mask.sum())
        if teacher_total > 0:
            payload = plot_overall_pos_neg_payload(
                teacher_pos,
                teacher_total - teacher_pos,
                'Teacher Overall Positive vs Negative',
            )
            chart_html['teacher_overall_html'] = payload['html']
            charts['teacher_overall'] = payload['figure']

    if facility_col and facility_col in df.columns:
        facility_mask = df[facility_col].notna()
        facility_group = (
            df.loc[facility_mask]
            .groupby(facility_col)['predicted']
            .mean()
            .sort_values(ascending=False)
        )
        if not facility_group.empty:
            payload = plot_grouped_pos_neg_payload(
                facility_group,
                'Positive vs Negative Feedback by Facility',
            )
            chart_html['facility_plot_html'] = payload['html']
            charts['facility_sentiment'] = payload['figure']
            stats['facility'] = {k: float(v) for k, v in facility_group.items()}
        facility_pos = int(df.loc[facility_mask, 'predicted'].sum())
        facility_total = int(facility_mask.sum())
        if facility_total > 0:
            payload = plot_overall_pos_neg_payload(
                facility_pos,
                facility_total - facility_pos,
                'Facility Overall Positive vs Negative',
            )
            chart_html['facility_overall_html'] = payload['html']
            charts['facility_overall'] = payload['figure']

    category_columns = [
        'Teacher Feedback',
        'Course Content',
        'Examination pattern',
        'Laboratory Library Facilities',
        'Extra Co-Curricular Activities',
        'Any other suggestion',
    ]
    present_categories = [c for c in category_columns if c in df.columns]
    if present_categories:
        import pandas as _pd

        category_scores = {}
        for c in present_categories:
            texts = (
                df[c]
                .dropna()
                .astype(str)
                .str.strip()
            )
            texts = texts[texts != '']
            if texts.empty:
                continue
            positive_votes = 0
            negative_votes = 0
            for text in texts:
                pos_count, neg_count = sentiment_word_counts(text)
                if pos_count == 0 and neg_count == 0:
                    continue
                if pos_count >= neg_count:
                    positive_votes += 1
                else:
                    negative_votes += 1
            total_votes = positive_votes + negative_votes
            if total_votes == 0:
                continue
            category_scores[c] = positive_votes / total_votes
        if category_scores:
            cat_series = _pd.Series(category_scores).sort_values(ascending=False)
            payload = plot_grouped_pos_neg_payload(
                cat_series,
                'Positive vs Negative Feedback by Category',
            )
            chart_html['category_plot_html'] = payload['html']
            charts['category_sentiment'] = payload['figure']
            stats['categories'] = {k: float(v) for k, v in cat_series.items()}

    if branch_col and branch_col in df.columns:
        branch_mask = df[branch_col].notna()
        branch_group = (
            df.loc[branch_mask]
            .groupby(branch_col)['predicted']
            .mean()
            .sort_values(ascending=False)
        )
        if not branch_group.empty:
            payload = plot_grouped_pos_neg_payload(
                branch_group,
                'Positive vs Negative Feedback by Branch',
            )
            chart_html['branch_plot_html'] = payload['html']
            charts['branch_sentiment'] = payload['figure']
            stats['branch'] = {k: float(v) for k, v in branch_group.items()}

    if department_col and department_col in df.columns:
        dept_mask = df[department_col].notna()
        dept_group = (
            df.loc[dept_mask]
            .groupby(department_col)['predicted']
            .mean()
            .sort_values(ascending=False)
        )
        if not dept_group.empty:
            payload = plot_grouped_pos_neg_payload(
                dept_group,
                'Positive vs Negative Feedback by Department',
            )
            chart_html['department_plot_html'] = payload['html']
            charts['department_sentiment'] = payload['figure']
            stats['department'] = {k: float(v) for k, v in dept_group.items()}

    overall_payload = plot_overall_pos_neg_payload(
        positive,
        negative,
        'Overall Positive vs Negative Feedback Distribution',
    )
    chart_html['overall_plot_html'] = overall_payload['html']
    charts['overall_sentiment'] = overall_payload['figure']

    columns_used = {
        'feedback': feedback_col,
        'label': label_col if label_col in df.columns else None,
        'teacher': teacher_col if teacher_col in df.columns else None,
        'facility': facility_col if facility_col in df.columns else None,
        'branch': branch_col if branch_col in df.columns else None,
        'department': department_col if department_col in df.columns else None,
    }

    model_info = {
        'strategy': model_used,
        'used_existing_model': bool(use_model),
        'saved_model_path': saved_model,
        'saved_model_name': os.path.basename(saved_model) if saved_model else None,
    }

    analysis = {
        'summary': {
            'total': total,
            'positive': positive,
            'negative': negative,
            'positive_pct': pos_pct,
            'negative_pct': neg_pct,
        },
        'metrics': metrics,
        'accuracy': accuracy,
        'charts': charts,
        'chart_html': chart_html,
        'stats': stats,
        'columns_used': columns_used,
        'model': model_info,
        'saved_model_path': saved_model,
    }

    return analysis


def process_dataframe(df, feedback_col=None, label_col=None, teacher_col=None, facility_col=None, branch_col=None, department_col=None, use_model=None, saved_model_path=None):
    """Process a dataframe and return rendered HTML results."""

    try:
        analysis = generate_analysis(
            df,
            feedback_col=feedback_col,
            label_col=label_col,
            teacher_col=teacher_col,
            facility_col=facility_col,
            branch_col=branch_col,
            department_col=department_col,
            use_model=use_model,
            saved_model_path=saved_model_path,
        )
    except ValueError as exc:
        return str(exc), 400

    summary = analysis['summary']
    return render_template(
        'results.html',
        accuracy=analysis['accuracy'],
        total=summary['total'],
        positive=summary['positive'],
        negative=summary['negative'],
        agg=analysis['chart_html'],
        metrics=analysis['metrics'],
        saved_model_path=analysis['saved_model_path'],
    )


@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
def api_analyze():
    """API endpoint that accepts a CSV upload and returns analysis JSON."""

    if request.method == 'OPTIONS':
        return _cors_json({}, status=204)

    uploaded_file = request.files.get('file')
    if uploaded_file is None or uploaded_file.filename == '':
        return _cors_json({'error': 'No file provided'}, status=400)
    if not allowed_file(uploaded_file.filename):
        return _cors_json({'error': 'Unsupported file type'}, status=400)

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        return _cors_json({'error': f'Error reading CSV: {exc}'}, status=400)

    detected = detect_columns(list(df.columns))

    # Allow overrides from form fields while falling back to detected columns.
    feedback_col = request.form.get('feedback_col') or detected.get('feedback')
    label_col = request.form.get('label_col') or detected.get('label')
    teacher_col = request.form.get('teacher_col') or detected.get('teacher')
    facility_col = request.form.get('facility_col') or detected.get('facility')
    branch_col = request.form.get('branch_col') or detected.get('branch')
    department_col = request.form.get('department_col') or detected.get('department')
    use_model = request.form.get('use_model') or None

    try:
        analysis = generate_analysis(
            df,
            feedback_col=feedback_col,
            label_col=label_col,
            teacher_col=teacher_col,
            facility_col=facility_col,
            branch_col=branch_col,
            department_col=department_col,
            use_model=use_model,
        )
    except ValueError as exc:
        return _cors_json({'error': str(exc)}, status=400)

    api_payload = {k: v for k, v in analysis.items() if k != 'chart_html'}
    api_payload['detected_columns'] = detected
    return _cors_json(api_payload)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
