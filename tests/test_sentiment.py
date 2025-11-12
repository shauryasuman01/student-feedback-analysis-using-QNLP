import io

from app import classify_sentiment, rule_based_sentiment


def test_classify_sentiment_positive():
    assert classify_sentiment("This was really helpful and clear") == 1


def test_classify_sentiment_neutral():
    # ambiguous / empty should return None
    assert classify_sentiment("") is None
    assert classify_sentiment("Nothing to report.") is None


def test_rule_based_preserves_neutral():
    # rule_based_sentiment should not coerce neutral into negative
    assert rule_based_sentiment("") is None