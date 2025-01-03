from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()


@app.post("/webhook")
async def handle_webhook(request: Request):
    payload = await request.json()
    print("Received webhook payload:", payload)
    return {"status": "success"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)