import requests
import json

# Payload from test_input.json
payload = {
    "input": {
        "zip_url": "https://raw.githubusercontent.com/runpod-workers/sample-inputs/refs/heads/main/images/froggy.zip",
        "training_params": {
            "trigger_word": "TOK",
            "autocaption": True,
            "steps": 1000,
            "learning_rate": 0.0004,
            "batch_size": 1,
            "resolution": "512,768,1024",
            "lora_rank": 16,
            "caption_dropout_rate": 0.05,
            "optimizer": "adamw8bit"
        },
        "base64_images": {}
    }
}

def send_request(url, data):
    try:
        headers = {
            'Content-Type': 'application/json'
        }
        
        # Send POST request
        response = requests.post(url, json=data, headers=headers)
        
        # Check if request was successful
        response.raise_for_status()
        
        # Return response data
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        return None

if __name__ == "__main__":
    # Replace with your actual API endpoint
    api_url = "http://localhost:8006/runsync"
    
    # Send the request
    result = send_request(api_url, payload)
    
    if result:
        print("Response received:")
        print(json.dumps(result, indent=2))