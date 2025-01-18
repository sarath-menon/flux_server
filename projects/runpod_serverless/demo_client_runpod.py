#%% runpod_async_demo
import runpod
import os
from flux_server.custom_types import TrainingParams

runpod.api_key = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = "jycz4f0g9y6d2k"
endpoint = runpod.Endpoint(ENDPOINT_ID)
#%%

# Create training params using the Pydantic model
training_params = TrainingParams(
    trigger_word="TOK",
    autocaption=True,
    steps=1000,
    learning_rate=0.0004,
    batch_size=1,
    resolution="512,768,1024",
    lora_rank=16,
    caption_dropout_rate=0.05,
    optimizer="adamw8bit"
)

run_request = endpoint.run(
    {
        "zip_url": "https://raw.githubusercontent.com/runpod-workers/sample-inputs/refs/heads/main/images/froggy.zip",
        "training_params": training_params.model_dump(),  # Convert to dict for JSON serialization
        "base64_images": {}
    }
)

for output in run_request.stream():
    print(output)
# %%
