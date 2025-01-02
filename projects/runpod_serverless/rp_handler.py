
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pathlib import Path
import flux_server.train
import logging
from flux_server.custom_types import TrainingParams

# Add logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI()


@app.get("/")
async def root():
    return {"status": "Flux server is running"}

@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    params: str = Form(...)
):
    logger.info(f"Received training request with file: {file.filename}")
    
    if not file.filename.endswith('.zip'):
        logger.error(f"Invalid file format: {file.filename}")
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")

    # Parse JSON string into TrainingParams
    try:
        params_dict = TrainingParams.parse_raw(params)
        logger.debug(f"Parsed training parameters: {params_dict}")
    except Exception as e:
        logger.error(f"Failed to parse parameters: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid params format: {str(e)}")

    # Save uploaded file
    temp_path = Path("/tmp/input.zip")
    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded file to {temp_path}")
    finally:
        file.file.close()

    try:
        # Convert params to dict and remove None values
        params_dict = params_dict.dict(exclude_none=True)
        logger.info("Starting training process")
        
        # Run training
        output_path = train.handle_training(
            input_images_path=str(temp_path),
            **params_dict
        )
        
        logger.info(f"Training completed successfully. Output saved to {output_path}")
        return {"status": "success", "output_path": str(output_path)}

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup
        if temp_path.exists():
            logger.debug(f"Cleaning up temporary file: {temp_path}")
            temp_path.unlink()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 