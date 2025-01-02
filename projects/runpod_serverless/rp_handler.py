import os
import logging
import shutil
from pathlib import Path

import runpod
from runpod.serverless.utils import rp_download, rp_cleanup
from runpod.serverless.utils.rp_validator import validate
from flux_server.custom_types import TrainingParams

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

INPUT_SCHEMA = {
    'zip_url': {
        'type': str,
        'required': True
    },
    'training_params': {
        'type': dict,
        'required': True
    }
}

def run(job):
    '''
    RunPod handler for training requests
    '''
    job_input = job['input']
    
    # Validate inputs
    if 'errors' in (job_input := validate(job_input, INPUT_SCHEMA)):
        return {'error': job_input['errors']}
    job_input = job_input['validated_input']


    allowed_extensions = [".jpg", ".jpeg", ".png", ".txt"]
    dataset_dir = Path("input_images")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Download input file
    # temp_path = Path("/tmp/input.zip")
    downloaded_input = rp_download.file(job_input['zip_url'])

    try:
        for root, dirs, files in os.walk(downloaded_input['extracted_path']):
            print(f"Files in {root}:")
            print("Dirs:", dirs)
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file_path)[1].lower() in allowed_extensions:
                    shutil.copy(
                        os.path.join(downloaded_input['extracted_path'], file_path),
                        dataset_dir
                    )
        logger.info(f"Downloaded input files to {dataset_dir}")

        # shutil.move(downloaded_file, temp_path)
        # logger.info(f"Downloaded input file to {temp_path}")
        
        # Parse and validate training parameters
        try:
            params_dict = TrainingParams.parse_obj(job_input['training_params'])
            params_dict = params_dict.dict(exclude_none=True)
        except Exception as e:
            logger.error(f"Failed to parse parameters: {str(e)}")
            return {"error": f"Invalid training parameters: {str(e)}"}
        
        # Run training
        import flux_server.train
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
        # if temp_path.exists():
        #     temp_path.unlink()
        rp_cleanup.clean(['input_objects'])

runpod.serverless.start({"handler": run})