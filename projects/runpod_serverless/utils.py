

import os
import logging
import shutil
from pathlib import Path
import base64

import runpod
from runpod.serverless.utils import rp_download, rp_cleanup

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

def zip_images(zip_url: str , dataset_dir: str):
    downloaded_input = rp_download.file(zip_url)
    extracted_path =  downloaded_input['extracted_path']
    allowed_extensions = [".jpg", ".jpeg", ".png", ".txt"]

    for root, dirs, files in os.walk(extracted_path):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.splitext(file_path)[1].lower() in allowed_extensions:
                shutil.copy(
                    os.path.join(extracted_path, file_path),
                    dataset_dir
                )
    print(f"Downloaded input files to {dataset_dir}")