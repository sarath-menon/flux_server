#%% runpod_async_demo
import runpod
import os

runpod.api_key = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = "jycz4f0g9y6d2k"
endpoint = runpod.Endpoint(ENDPOINT_ID)
#%%

run_request = endpoint.run(
    {"number": 7, "stream": True}
)

for output in run_request.stream():
    print(output)
# %%
