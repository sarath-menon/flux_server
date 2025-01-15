import os
import logging
import shutil
from pathlib import Path
import base64
import asyncio
from watchdog.observers import Observer
import time

import runpod
from runpod.serverless.utils import rp_download, rp_cleanup
from runpod.serverless.utils.rp_validator import validate
from flux_server.custom_types import TrainingParams, INPUT_SCHEMA
import flux_server.train

from utils import save_base64_images, zip_images
from file_handler import OutputFileHandler

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

async def monitor_output_directory(path, timeout=300):  # 5 minutes timeout
    event_handler = OutputFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    logger.info(f"Monitoring output directory: {Path(path).absolute()}")
    
    try:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if event_handler.new_files:
                logger.info(f"New files detected: {event_handler.new_files}")
                files = event_handler.new_files.copy()
                event_handler.new_files.clear()
                for file_path in files:
                    # Only process image files
                    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        try:
                            with open(file_path, "rb") as image_file:
                                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                                yield {
                                    "type": "image",
                                    "path": file_path,
                                    "filename": Path(file_path).name,
                                    "content": image_data
                                }
                                return
                        except Exception as e:
                            msg = f"Failed to process image {file_path}: {str(e)}"
                            logger.error(msg)
                            yield {"error": msg}
                    else:
                        # For non-image files, just yield the path as before
                        yield {"type": "file_created", "path": file_path}
            await asyncio.sleep(1)
    finally:
        observer.stop()
        observer.join()

async def run(job):
    '''
    RunPod async handler for training requests
    '''
    job_input = job['input']
    
    # Validate inputs
    if 'errors' in (job_input := validate(job_input, INPUT_SCHEMA)):
        yield {'error': job_input['errors']}
        return  # Early return after yielding error

    job_input = job_input['validated_input']

    if not job_input.get('base64_images') and not job_input.get('zip_url'):          
        yield {'error': 'Images must be provided as base64 strings or a ZIP file URL'}
        return  # Early return after yielding error

    dataset_dir = Path("input_images")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Process images sequentially
    if 'zip_url' in job_input:
        zip_images(job_input['zip_url'], dataset_dir)

    if 'base64_images' in job_input:    
        save_base64_images(job_input['base64_images'], dataset_dir)
        
    # Parse and validate training parameters
    try:
        params_dict = TrainingParams.model_validate(job_input['training_params'])
        params_dict = params_dict.model_dump(exclude_none=True)
    except Exception as e:
        logger.error(f"Failed to parse parameters: {str(e)}")
        yield {"error": f"Invalid training parameters: {str(e)}"}
        return  # Early return after yielding error
    
    # Run training
    try:
        logger.info("Starting training process")
        params = TrainingParams.model_validate(job_input['training_params'])
        
        # Start monitoring the output directory
        output_dir = Path("./output")
        output_dir.mkdir(exist_ok=True)
        
        # Create monitoring task
        monitor_task = monitor_output_directory(output_dir)
        
        # Run training
        training_task = asyncio.create_task(
            flux_server.train.handle_training(training_params=params)
        )
        
        # Monitor for new files while training is running
        while not training_task.done():
            try:
                async for file_event in monitor_task:
                    yield file_event
            except StopAsyncIteration:
                break
            await asyncio.sleep(1)
        
        # Get the training result
        output_path = await training_task
        
        logger.info(f"Training completed successfully. Output at {output_path}")
        yield {
            "status": "success",
            "output_path": str(output_path)
        }
                
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        yield {"error": str(e)}
    
    finally:
        # Cleanup
        rp_cleanup.clean(['input_objects'])
        # Delete the input images directory if it exists
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)

async def handler(job):
    """
    Wrapper handler that consumes the async generator
    """
    results = []
    async for result in run(job):
        results.append(result)
    return results

runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": False  # Changed to False since we're aggregating in the handler
})
