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
            "optimizer": "adamw8bit",
            "mock_training": True,
            "mock_training_samples_interval": 2,
            "job_name": "test_job"

        },
        "base64_images": {}
    }
}

def send_request(url, data):
    try:
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
        
        # Send POST request with stream=True
        with requests.post(url, json=data, headers=headers, stream=True) as response:
            response.raise_for_status()
            
            # Process each line as it comes in
            for line in response.iter_lines():
                if line:
                    result = json.loads(line)
                    print("Received streaming response:")
                    print(json.dumps(result, indent=2))
                    print("-" * 50)
    
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