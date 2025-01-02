import os
import logging
import shutil
from pathlib import Path
import base64

import runpod
from runpod.serverless.utils import rp_download, rp_cleanup
from runpod.serverless.utils.rp_validator import validate
from flux_server.custom_types import TrainingParams, INPUT_SCHEMA
import flux_server.train

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def save_base64_images(base64_images: dict, dataset_dir: str):
    """Save base64 encoded images to the dataset directory."""
    for filename, data_uri in base64_images.items():
        # Split the data URI to get the base64 part
        if ',' in data_uri:
            b64_string = data_uri.split(',')[1]
        else:
            b64_string = data_uri

        # Decode base64 string
        img_data = base64.b64decode(b64_string)
        
        # Get extension from original filename
        extension = os.path.splitext(filename)[1].lower()
        if extension in ['.png', '.jpg', '.jpeg']:
            # Use the original filename as is
            file_path = os.path.join(dataset_dir, filename)
            
            # Save the image
            with open(file_path, 'wb') as f:
                f.write(img_data)

def run(job):
    '''
    RunPod handler for training requests
    '''
    job_input = job['input']
    
    # Validate inputs
    if 'errors' in (job_input := validate(job_input, INPUT_SCHEMA)):
        return {'error': job_input['errors']}

    job_input = job_input['validated_input']

    if not job_input.get('base64_images') and not job_input.get('zip_url'):          
        return {'error': 'Images must be provided as base64 strings or a ZIP file URL'} 

    allowed_extensions = [".jpg", ".jpeg", ".png", ".txt"]
    dataset_dir = Path("input_images")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if 'zip_url' in job_input:
        downloaded_input = rp_download.file(job_input['zip_url'])
        for root, dirs, files in os.walk(downloaded_input['extracted_path']):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file_path)[1].lower() in allowed_extensions:
                    shutil.copy(
                        os.path.join(downloaded_input['extracted_path'], file_path),
                        dataset_dir
                    )
        logger.info(f"Downloaded input files to {dataset_dir}")

    if 'base64_images' in job_input:    
        save_base64_images(job_input['base64_images'], dataset_dir)
        
    # Parse and validate training parameters
    try:
        params_dict = TrainingParams.parse_obj(job_input['training_params'])
        params_dict = params_dict.dict(exclude_none=True)
    except Exception as e:
        logger.error(f"Failed to parse parameters: {str(e)}")
        return {"error": f"Invalid training parameters: {str(e)}"}
    
    # Run training
    try:
        logger.info("Starting training process")
        output_path = flux_server.train.handle_training(
            # input_images_path=str(temp_path),
            **params_dict
        )
        
        logger.info(f"Training completed successfully. Output at {output_path}")
        return {
            "status": "success",
            "output_path": str(output_path)
        }
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        return {"error": str(e)}
    
    finally:
        # Cleanup
        rp_cleanup.clean(['input_objects'])
        # delete the input images   
        shutil.rmtree(dataset_dir)

runpod.serverless.start({"handler": run})
