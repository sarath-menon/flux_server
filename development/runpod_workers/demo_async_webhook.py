import runpod
import asyncio
import aiohttp  # Import aiohttp for making HTTP requests

async def async_generator_handler(job):
    job_input = job["input"]
    the_number = job_input["number"]

    if not isinstance(the_number, int):
        yield {"error": "Please provide an integer."}
        return

    async with aiohttp.ClientSession() as session:
        for i in range(the_number):
            yield i

            # Simulate an asynchronous task, such as processing time for a large language model
            await asyncio.sleep(1)

            # Send a POST request to the webhook endpoint
            async with session.post(
                "http://localhost:8000/webhook",
                json={"number": i}
            ) as response:
                # Print the response from the webhook server
                print(await response.json())

# Configure and start the RunPod serverless function
runpod.serverless.start(
    {
        "handler": async_generator_handler,  # Required: Specify the async handler
        "return_aggregate_stream": False,  # Optional: Aggregate results are accessible via /run endpoint
    }
)