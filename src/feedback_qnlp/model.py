"""Model stubs for QNLP student feedback analysis.

This file provides:
- a classical baseline trainer using scikit-learn
- a placeholder for a lambeq/pennylane-based quantum model

The quantum functions intentionally require lambeq/pennylane and will raise an informative ImportError if not available.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any

from sklearn.feature_extraction.text import CountVectorizer

try:
    import pennylane as qml  # type: ignore
    _PENNYLANE_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    qml = None  # type: ignore
    _PENNYLANE_AVAILABLE = False

try:
    import lambeq  # type: ignore
    _LAMBEQ_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    lambeq = None  # type: ignore
    _LAMBEQ_AVAILABLE = False

def train_classical_baseline(X, y, test_size=0.2, random_state=42):
    X = X
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return {"model": clf, "accuracy": acc}


def train_quantum_model(
    texts: List[str],
    labels: List[int],
    *,
    max_features: int = 100,
    test_size: float = 0.2,
    random_state: int = 42,
    n_qubits: int = 4,
    n_layers: int = 1,
    epochs: int = 20,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Small scaffold for a lambeq / pennylane pipeline.

    This function intentionally does not run anything by default. If the
    environment has `lambeq` and `pennylane` installed, you can implement a
    concrete pipeline here (diagram parsing, ansatz construction, conversion to
    a PennyLane circuit, and training). Example steps (high level):

    1. Use lambeq to parse sentences into DisCoCat diagrams and convert to
       quantum circuits (e.g. with `lambeq.ansatzes` and `lambeq.circuit` APIs).
    2. Use PennyLane to construct a device and QNode for the model ansatz.
    3. Train using an optimizer (e.g. `qml.AdamOptimizer`) on your labeled data.

    The project intentionally leaves the heavy QNLP implementation optional
    because `lambeq`/`pennylane` are large and platform-dependent. This stub
    only provides a clear ImportError with guidance when those packages are
    missing.
    """
    # Basic contract: texts (list of strings), labels (list of ints 0/1)
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")

    # Vectorize texts to compact numeric features
    # Optionally produce structural encodings using lambeq (when available).
    structural_feats = None
    if _LAMBEQ_AVAILABLE:
        try:
            # Best-effort: try known parser APIs and extract simple diagram stats
            feats = []
            try:
                from lambeq import BobcatParser
                parser = BobcatParser()
                # Some lambeq versions provide sentences2diagrams
                if hasattr(parser, 'sentences2diagrams'):
                    diagrams = parser.sentences2diagrams([str(t) for t in texts])
                else:
                    # fallback to parsing one-by-one
                    diagrams = [parser.sentence2diagram(str(t)) for t in texts]
            except Exception:
                # Try a generic high-level API
                try:
                    diagrams = lambeq.parse([str(t) for t in texts])
                except Exception:
                    diagrams = []

            for d in diagrams:
                # Extract a few robust numeric features from the diagram
                nb = 0
                nw = 0
                nwrd = 0
                try:
                    nb = len(getattr(d, 'boxes', []))
                except Exception:
                    nb = 0
                try:
                    nw = len(getattr(d, 'wires', []))
                except Exception:
                    nw = 0
                try:
                    nwrd = len(getattr(d, 'words', []))
                except Exception:
                    # some diagrams expose 'sentences' or 'tokens'
                    nwrd = 0
                feats.append([nb, nw, nwrd])

            if feats and len(feats) == len(texts):
                structural_feats = np.array(feats, dtype=float)
        except Exception:
            structural_feats = None

    vec = CountVectorizer(max_features=max_features)
    X_all = vec.fit_transform([str(t) for t in texts])
    # If structural features are available, concatenate them to the vectorized features
    if structural_feats is not None:
        try:
            from scipy.sparse import hstack as _hstack
            import numpy as _np

            struct_sparse = _np.asarray(structural_feats)
            # convert structural features to sparse and hstack
            from scipy.sparse import csr_matrix

            X_all = _hstack([X_all, csr_matrix(struct_sparse)])
        except Exception:
            # if concatenation fails, ignore structural features
            pass
    y_all = (np.array(labels)).astype(int)

    # If PennyLane is available, run a tiny variational circuit classifier
    if _PENNYLANE_AVAILABLE:
        # Simple encoding: map first n_qubits features to RY angles
        # Use PennyLane's numpy wrapper so gradients and requires_grad work as expected
        try:
            import pennylane.numpy as _qnp
        except Exception:
            import numpy as _qnp  # fallback

        X = X_all.toarray().astype(float)
        # reduce features to n_qubits by summing/averaging blocks
        n_samples, n_feats = X.shape
        if n_feats < n_qubits:
            # pad
            pad = _qnp.zeros((n_samples, n_qubits - n_feats))
            Xp = _qnp.hstack([X, pad])
        else:
            Xp = X[:, :n_qubits]

        # normalize to [0, pi]
        maxv = Xp.max() if Xp.max() > 0 else 1.0
        Xp = (Xp / maxv) * _qnp.pi

        # train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            Xp, y_all, test_size=test_size, random_state=random_state
        )

        dev = qml.device("default.qubit", wires=n_qubits)

        def layer(params):
            for i in range(n_qubits):
                qml.RY(params[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        @qml.qnode(dev)
        def circuit(inputs, params):
            # angle encoding
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            # variational layers
            for l in range(n_layers):
                layer(params[l])
            return qml.expval(qml.PauliZ(0))

        # initialize params: n_layers x n_qubits (use qml.numpy for trainable params)
        params = _qnp.zeros((n_layers, n_qubits), requires_grad=True)

        opt = qml.AdamOptimizer(stepsize=0.1)

        def predict_probs(Xb, params):
            probs = []
            for x in Xb:
                out = circuit(x.tolist(), params)
                # out may be an ArrayBox when using autograd; keep it as a numeric array
                prob = (1.0 + out) / 2.0
                probs.append(prob)
            return _qnp.array(probs)

        # simple training loop
        for epoch in range(epochs):
            # mini-batch SGD
            idx = _qnp.random.permutation(len(X_train))
            for i in range(0, len(idx), batch_size):
                batch_idx = idx[i : i + batch_size]
                Xb = X_train[batch_idx]
                yb = y_train[batch_idx]

                def cost(p):
                    preds = predict_probs(Xb, p)
                    # Use a differentiable loss on probabilities and keep it as an ArrayBox
                    y_true = _qnp.array(yb)
                    return ((preds - y_true) ** 2).mean()

                params = opt.step(cost, params)

        probs_test = predict_probs(X_test, params)
        preds_test = _qnp.where(probs_test >= 0.5, 1, 0)
        acc = float((_qnp.array(preds_test) == _qnp.array(y_test)).mean())

        # predictions for all inputs
        probs_all = predict_probs(Xp, params)
        preds_all = _qnp.where(probs_all >= 0.5, 1, 0)

        # Return a minimal artifact: vectorizer, params and predictions
        return {
            "model": {"pennylane_params": params, "n_qubits": n_qubits, "n_layers": n_layers},
            "vectorizer": vec,
            "accuracy": acc,
            "predictions": preds_all.tolist(),
            "quantum": True,
        }

    # If PennyLane is not available, provide a simulated fallback using classical logistic regression
    # This gives a workable 'quantum' path for users who don't have heavy deps installed.
    vec2 = vec
    X_all = vec2.fit_transform([str(t) for t in texts])
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=test_size, random_state=random_state
    )
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    preds_all = clf.predict(X_all)
    return {
        "model": clf,
        "vectorizer": vec2,
        "accuracy": float(acc),
        "predictions": preds_all.tolist(),
        "quantum": False,
    }
