import requests
import json

data = {
    'input': {'number': 7},
    'id': 'local_test',
    'webhook': 'http://localhost:8000/webhook'
}
headers = {"Content-Type": "application/json"}

# Stream the response
r = requests.post('http://localhost:8006/runsync', 
                 data=json.dumps(data), 
                 headers=headers,
                 stream=True)

# Print each line as it's received
for line in r.iter_lines():
    if line:
        # Decode the bytes to string and print
        decoded_line = line.decode('utf-8')
        print(decoded_line)

