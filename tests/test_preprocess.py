import os

from feedback_qnlp.preprocess import basic_clean, load_data, preprocess_texts


def test_basic_clean():
    assert basic_clean("Hello!!!") == "hello"
    assert basic_clean("  Multiple   spaces  ") == "multiple spaces"


def test_load_data(tmp_path):
    p = tmp_path / "tmp.csv"
    p.write_text("id,feedback,label\n1,Good,positive\n")
    df = load_data(str(p))
    assert df.shape[0] == 1
