import runpod


def reverse_string(s):
    return s[::-1]


def handler(job):
    print(f"string-reverser | Starting job {job['id']}")
    job_input = job["input"]

    input_string = job_input.get("text", "")

    if not input_string:
        return {"error": "No input text provided"}

    reversed_string = reverse_string(input_string)

    job_output = {"original_text": input_string, "reversed_text": reversed_string}

    return job_output


runpod.serverless.start({"handler": handler})