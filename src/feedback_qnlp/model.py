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


def train_classical_baseline(X, y, test_size=0.2, random_state=42):
    X = X
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return {"model": clf, "accuracy": acc}


def train_quantum_model(*args, **kwargs):
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
    try:
        import lambeq  # noqa: F401
        import pennylane as qml  # noqa: F401
    except Exception as e:
        raise ImportError(
            "Quantum training requires the 'lambeq' and 'pennylane' packages. "
            "They are not installed in this environment. To experiment with QNLP, "
            "install them (see README) or run quantum experiments in a separate environment. "
            "Original error: {}".format(e)
        )

    # At this point you can implement the pipeline. For example (very high-level):
    #
    # from lambeq import BobcatParser, CategoricalDisCoCat, ansatz
    # from lambeq import (your preferred helpers)
    # dev = qml.device('default.qubit', wires=...)
    # @qml.qnode(dev)
    # def circuit(params, inputs):
    #     ...
    #     return qml.expval(...)
    #
    # Then build a training loop that maps sentences -> circuit inputs -> train.

    raise NotImplementedError(
        "Quantum training pipeline is intentionally left as a project extension. "
        "Install lambeq/pennylane and follow lambeq examples to implement the pipeline."
    )
