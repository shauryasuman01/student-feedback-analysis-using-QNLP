import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app

with app.test_client() as c:
    r = c.get('/')
    print('STATUS', r.status_code)
    data = r.data.decode('utf-8', errors='replace')
    print('BODY_START')
    print(data[:1000])
    print('BODY_END')
