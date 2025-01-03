import requests
import json

data = {'input': {'number': 7}, 'id': 'local_test'}
headers = {"Content-Type": "application/json"}

r = requests.post('http://localhost:8006/run', data=json.dumps(data), headers=headers)

print(r.text)