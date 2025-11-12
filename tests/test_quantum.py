from feedback_qnlp.model import train_quantum_model


def test_train_quantum_simulated():
    texts = [
        "This course was helpful and clear",
        "Terrible and unclear lectures",
        "I enjoyed the labs",
        "Confusing explanations",
    ]
    labels = [1, 0, 1, 0]
    result = train_quantum_model(texts, labels, max_features=10, epochs=1)
    assert "accuracy" in result
    assert "model" in result
    assert "vectorizer" in result
    assert result["accuracy"] >= 0.0