import os
import logging
import shutil
from pathlib import Path
import base64
import asyncio

import runpod
from runpod.serverless.utils import rp_download, rp_cleanup
from runpod.serverless.utils.rp_validator import validate
from flux_server.custom_types import TrainingParams, INPUT_SCHEMA
import flux_server.train

from utils import save_base64_images, zip_images

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

async def async_handler(job):
    '''
    RunPod async handler for training requests
    '''
    job_input = job['input']
    
    # Validate inputs
    if 'errors' in (job_input := validate(job_input, INPUT_SCHEMA)):
        yield {'error': job_input['errors']}
        return

    job_input = job_input['validated_input']

    if not job_input.get('base64_images') and not job_input.get('zip_url'):          
        yield {'error': 'Images must be provided as base64 strings or a ZIP file URL'}
        return

    dataset_dir = Path("input_images")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create tasks for image processing
        tasks = []
        if 'zip_url' in job_input:
            tasks.append(asyncio.create_task(zip_images(job_input['zip_url'], dataset_dir)))

        if 'base64_images' in job_input:    
            tasks.append(asyncio.create_task(save_base64_images(job_input['base64_images'], dataset_dir)))
        
        # Wait for all image processing tasks to complete
        if tasks:
            yield {"status": "processing", "message": "Processing input images..."}
            await asyncio.gather(*tasks)
            
        # Parse and validate training parameters
        try:
            params_dict = TrainingParams.parse_obj(job_input['training_params'])
            params_dict = params_dict.dict(exclude_none=True)
        except Exception as e:
            logger.error(f"Failed to parse parameters: {str(e)}")
            yield {"error": f"Invalid training parameters: {str(e)}"}
            return
        
        # Run training
        logger.info("Starting training process")
        yield {"status": "processing", "message": "Starting training process..."}
        
        output_path = await flux_server.train.handle_training(
            **params_dict
        )
        
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
        # delete the input images   
        shutil.rmtree(dataset_dir)

runpod.serverless.start({
    "handler": async_handler,
    "return_aggregate_stream": False  # Set to False to stream individual results
})
