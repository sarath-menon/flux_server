import runpod


def is_even(job):
    job_input = job["input"]
    the_number = job_input["number"]

    if not isinstance(the_number, int):
        return {"error": "Please provide an integer."}

    return the_number % 2 == 0


runpod.serverless.start({"handler": is_even})