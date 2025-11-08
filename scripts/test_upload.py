import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app

csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'student_feedback_full.csv')
assert os.path.exists(csv_path), 'CSV file not found: ' + csv_path

with open(csv_path, 'rb') as f:
    data = {
        'file': (f, 'student_feedback_full.csv')
    }
    with app.test_client() as c:
        r = c.post('/upload', data=data, content_type='multipart/form-data')
        print('STATUS', r.status_code)
        print(r.data.decode('utf-8')[:2000])
