import runpod
import asyncio

async def async_generator_handler(job):
    job_input = job["input"]
    the_number = job_input["number"]

    if not isinstance(the_number, int):
        yield {"error": "Please provide an integer."}
        return

    for i in range(the_number):
        yield i

        # Simulate an asynchronous task, such as processing time for a large language model
        await asyncio.sleep(1)

# Configure and start the RunPod serverless function
runpod.serverless.start(
    {
        "handler": async_generator_handler,  # Required: Specify the async handler
        "return_aggregate_stream": True,  # Optional: Aggregate results are accessible via /run endpoint
    }
)