import io
import os

import pytest

from app import app as flask_app


def test_upload_endpoint(client):
    # Create a small CSV in memory with multiple examples so training has both classes
    csv = (
        b"id,feedback,label\n"
        b"1,Good class,positive\n"
        b"2,Bad class,negative\n"
        b"3,Helpful and clear,positive\n"
        b"4,Confusing lecture,negative\n"
        b"5,Enjoyed the labs,positive\n"
        b"6,Slow and unclear,negative\n"
    )
    data = {
        'file': (io.BytesIO(csv), 'test.csv')
    }
    resp = client.post('/upload', data=data, content_type='multipart/form-data')
    # upload triggers processing and should return a rendered page
    assert resp.status_code == 200
    assert b"Sentiment dashboard" in resp.data


@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client
