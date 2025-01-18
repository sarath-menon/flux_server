#%% runpod_async_demo
import runpod
import os

runpod.api_key = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = "hw1xb1oimoyvie"
endpoint = runpod.Endpoint(ENDPOINT_ID)
#%%

run_request = endpoint.run_sync(
    {"number": 7}
)

# Check the status of the endpoint run request
print(run_request.status())

#%%

# Get the output of the endpoint run request, blocking until the endpoint run is complete.
print(run_request.output())

#%%
