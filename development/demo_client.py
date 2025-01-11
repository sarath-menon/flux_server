import requests
import json

data = {
    'input': {'number': 7},
    'id': 'local_test',
    'webhook': 'http://localhost:8000/webhook'
}
headers = {"Content-Type": "application/json"}

r = requests.post('http://localhost:8006/runsync', data=json.dumps(data), headers=headers)

print(r.text)

