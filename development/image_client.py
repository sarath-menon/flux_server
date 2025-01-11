import requests
import json
import base64
import os

# Create output directory for received images
output_dir = "received_images"
os.makedirs(output_dir, exist_ok=True)

data = {
    'input': {'number': 7},
    'id': 'image_watcher_test',
    'webhook': 'http://localhost:8000/webhook'
}
headers = {"Content-Type": "application/json"}

# Use /run endpoint for streaming response
response = requests.post(
    'http://localhost:8006/runsync',
    data=json.dumps(data),
    headers=headers,
)

print(response)

try:
    result = json.loads(response.text)
    for key, value in result.items():
        print(f"{key}")
    
    # Check if there's an error
    if "error" in result:
        print(f"Error: {result['error']}")
        
    # Process image data
    for item in result:
        if item["type"] == "image":
            filename = item["filename"]
            base64_data = item["content"]
            print(f"Received image: {filename}")
            
        try:
            # Save the image
            image_path = os.path.join(output_dir, filename)
            with open(image_path, "wb") as f:
                f.write(base64.b64decode(base64_data))
            print(f"Saved image: {filename}")
        except Exception as e:
            print(f"Error saving image {filename}: {e}")
        
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
except Exception as e:
    print(f"Error processing response: {e}")

