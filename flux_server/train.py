import os
import sys
import shutil
import subprocess
import logging
import asyncio
from functools import partial
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path

# Add necessary paths
sys.path.append("flux_server/ai-toolkit")

import time
from typing import Optional, OrderedDict
from zipfile import ZipFile, is_zipfile

import torch
from huggingface_hub import HfApi

from .wandb_client import WeightsAndBiasesClient, logout_wandb
from .layer_match import match_layers_to_optimize, available_layers_to_optimize
from .submodule_patches import patch_submodules

from .caption import Captioner
from .custom_types import TrainingParams

# Set environment variables
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["LANG"] = "en_US.UTF-8"


@dataclass
class TrainingPaths:
    job_name: str = "default_job"
    weights_path: Path = Path("./FLUX.1-dev")
    input_dir: Path = Path("./input_images")
    output_dir: Path = Path("./output")
    samples_dir: Path = Path("./testdata/harrison_ford")
    
    @property
    def job_dir(self) -> Path:
        return self.output_dir / self.job_name

# Replace global variables with instance
training_paths = TrainingPaths()

logger = logging.getLogger(__name__)


def clean_up():
    logout_wandb()
    # if training_paths.input_dir.exists():
    #     shutil.rmtree(training_paths.input_dir)
    if training_paths.output_dir.exists():
        shutil.rmtree(training_paths.output_dir)

def extract_zip(input_images: Path, input_dir: Path):
    if not is_zipfile(input_images):
        logger.error(f"Invalid zip file: {input_images}")
        raise ValueError("input_images must be a zip file")

    input_dir.mkdir(parents=True, exist_ok=True)
    image_count = 0
    logger.info(f"Extracting zip file: {input_images}")
    
    with ZipFile(input_images, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            if not file_info.filename.startswith("__MACOSX/") and not file_info.filename.startswith("._"):
                zip_ref.extract(file_info, input_dir)
                image_count += 1

    logger.info(f"Extracted {image_count} files from zip to {input_dir}")

def download_weights():
    if not training_paths.weights_path.exists():
        t1 = time.time()
        subprocess.check_output([
            "pget",
            "-xf",
            "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar",
            str(training_paths.weights_path.parent),
        ])
        t2 = time.time()
        print(f"Downloaded base weights in {t2 - t1} seconds")

async def handle_training(
    training_params: TrainingParams,
) -> Path:
    """Handle the training process with parameters from TrainingParams model"""
    logger.info("Starting training process")

    training_paths.job_name = training_params.job_name
    
    # Add mock handling at the start
    if training_params.mock_training:
        logger.info("Using mock training mode")
        delay = training_params.mock_training_samples_interval
        
        # Create output directory structure
        mock_samples_dir = training_paths.job_dir / "samples"
        mock_samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy sample images with delay
        if not training_paths.samples_dir.exists():
            logger.error("SAMPLES_DIR does not exist, skipping sample image generation.")
            return Path(mock_samples_dir)

        # Get all image files in samples directory
        sample_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            sample_files.extend(list(training_paths.samples_dir.glob(ext)))
        logger.info(f"Found {len(sample_files)} sample files in {training_paths.samples_dir}")

        # Create progress bar
        pbar = tqdm(total=len(sample_files), desc="Generating mock samples")
        
        for i, sample_file in enumerate(sample_files, 1):
            output_file = mock_samples_dir / f"sample_{i:04d}.png"
            shutil.copy2(sample_file, output_file)
            await asyncio.sleep(delay)
            pbar.update(1)
            
        pbar.close()
        
        return Path(mock_samples_dir)

    from jobs import BaseJob
    from toolkit.config import get_config
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    from .custom_trainers import CustomJob, CustomSDTrainer
    
    patch_submodules()

    logger.debug(f"Training parameters: trigger_word={training_params.trigger_word}, steps={training_params.steps}, learning_rate={training_params.learning_rate}")
    
    clean_up()
    output_path = Path("/tmp/trained_model.tar")

    layers_to_optimize = None
    if training_params.layers_to_optimize_regex:
        logger.info(f"Matching layers with regex: {training_params.layers_to_optimize_regex}")
        layers_to_optimize = match_layers_to_optimize(training_params.layers_to_optimize_regex)
        if not layers_to_optimize:
            logger.error(f"No layers matched regex: {training_params.layers_to_optimize_regex}")
            raise ValueError(
                f"The regex '{training_params.layers_to_optimize_regex}' didn't match any layers. These layers can be optimized:\n"
                + "\n".join(available_layers_to_optimize)
            )

    sample_prompts = []
    if training_params.sample_prompts:
        sample_prompts = [p.strip() for p in training_params.sample_prompts.split("\n")]
        logger.info(f"Sample prompts: {sample_prompts}")

    # Create training config
    train_config = create_training_config(
        training_params.trigger_word, training_params.steps, 
        training_params.learning_rate, training_params.batch_size,
        training_params.resolution, training_params.lora_rank,
        training_params.caption_dropout_rate, training_params.optimizer,
        training_params.cache_latents,
        training_params.cache_latents_to_disk, layers_to_optimize,
        training_params.wandb_api_key, training_params.wandb_save_interval,
        training_params.wandb_sample_interval, training_params.sample_prompts
    )

    logger.info("Setting up W&B client")
    wandb_client = setup_wandb(
        training_params.wandb_api_key, training_params.trigger_word,
        training_params.autocaption, training_params.autocaption_prefix,
        training_params.autocaption_suffix, training_params.steps,
        training_params.learning_rate, training_params.batch_size,
        training_params.resolution, training_params.lora_rank,
        training_params.caption_dropout_rate, training_params.optimizer,
        sample_prompts, training_params.wandb_project,
        training_params.wandb_entity, training_params.wandb_run
    )

    # # download_weights()
    # extract_zip(Path(input_images_path), INPUT_DIR)

    if not training_params.trigger_word:
        logger.debug("No trigger word provided, removing from config")
        del train_config["config"]["process"][0]["trigger_word"]

    logger.info("Handling image captioning")
    handle_captioning(training_params.autocaption, training_params.autocaption_prefix, training_params.autocaption_suffix)

    logger.info("Starting training job")
    job = CustomJob(get_config(train_config, name=None), wandb_client)
    job.run()

    if wandb_client:
        logger.info("Finishing W&B logging")
        wandb_client.finish()

    job.cleanup()
    process_output_files()

    logger.info(f"Creating output archive at {output_path}")
    os.system(f"tar -cvf {output_path} {training_paths.job_dir}")
    logger.info("Training completed successfully")
    return output_path

def create_training_config(trigger_word, steps, learning_rate, batch_size, resolution,
                         lora_rank, caption_dropout_rate, optimizer, cache_latents_to_disk,
                         cache_latents, layers_to_optimize, wandb_api_key, wandb_save_interval,
                         wandb_sample_interval, sample_prompts):
    train_config = OrderedDict(
        {
            "job": "custom_job",
            "config": {
                "name": training_paths.job_name,
                "process": [
                    {
                        "type": "custom_sd_trainer",
                        "training_folder": str(training_paths.output_dir),
                        "device": "cuda:0",
                        "trigger_word": trigger_word,
                        "network": {
                            "type": "lora",
                            "linear": lora_rank,
                            "linear_alpha": lora_rank,
                        },
                        "save": {
                            "dtype": "float16",
                            "save_every": (
                                wandb_save_interval if wandb_api_key else steps + 1
                            ),
                            "max_step_saves_to_keep": 1,
                        },
                        "datasets": [
                            {
                                "folder_path": str(training_paths.input_dir),
                                "caption_ext": "txt",
                                "caption_dropout_rate": caption_dropout_rate,
                                "shuffle_tokens": False,
                                "cache_latents_to_disk": cache_latents_to_disk,
                                "cache_latents": cache_latents,
                                "resolution": [
                                    int(res) for res in resolution.split(",")
                                ],
                            }
                        ],
                        "train": {
                            "batch_size": batch_size,
                            "steps": steps,
                            "gradient_accumulation_steps": 1,
                            "train_unet": True,
                            "train_text_encoder": False,
                            "content_or_style": "balanced",
                            "gradient_checkpointing": True,
                            "noise_scheduler": "flowmatch",
                            "optimizer": optimizer,
                            "lr": learning_rate,
                            "ema_config": {"use_ema": True, "ema_decay": 0.99},
                            "dtype": "bf16",
                            "skip_first_sample": True,
                        },
                        "model": {
                            # "name_or_path": str(WEIGHTS_PATH),
                            "name_or_path": "black-forest-labs/FLUX.1-dev",
                            "is_flux": True,
                            "quantize": True,
                        },
                        "sample": {
                            "sampler": "flowmatch",
                            "sample_every": training_params.sample_every,
                            "width": training_params.sample_width,
                            "height": training_params.sample_height,
                            "prompts": sample_prompts,
                            "neg": "",
                            "seed": training_params.sample_seed,
                            "walk_seed": training_params.sample_walk_seed,
                            "guidance_scale": training_params.sample_guidance_scale,
                            "sample_steps": training_params.sample_steps,
                        },
                    }
                ],
            },
            "meta": {"name": "[name]", "version": "1.0"},
        }
    )

    if layers_to_optimize:
        train_config["config"]["process"][0]["network"]["network_kwargs"] = {
            "only_if_contains": layers_to_optimize
        }

    return train_config

def setup_wandb(wandb_api_key, *args):
    # Implementation of W&B setup
    # (Move the existing W&B setup logic here)
    pass

def handle_captioning(autocaption, prefix, suffix):
    captioner = Captioner()
    if autocaption and not captioner.all_images_are_captioned(INPUT_DIR):
        captioner.caption_images(INPUT_DIR, prefix, suffix)

    del captioner
    torch.cuda.empty_cache()

def process_output_files():
    lora_file = training_paths.job_dir / f"{training_paths.job_name}.safetensors"
    lora_file.rename(training_paths.job_dir / "lora.safetensors")

    samples_dir = training_paths.job_dir / "samples"
    if samples_dir.exists():
        shutil.rmtree(samples_dir)

    # Remove any intermediate lora paths
    lora_paths = training_paths.job_dir.glob("*.safetensors")
    for path in lora_paths:
        if path.name != "lora.safetensors":
            path.unlink()

    # Optimizer is used to continue training, not needed in output
    optimizer_file = training_paths.job_dir / "optimizer.pt"
    if optimizer_file.exists():
        optimizer_file.unlink()

    # Copy generated captions to the output tar
    captions_dir = training_paths.job_dir / "captions"
    captions_dir.mkdir(exist_ok=True)
    for caption_file in training_paths.input_dir.glob("*.txt"):
        shutil.copy(caption_file, captions_dir)

def handle_training_sync(
    training_params: TrainingParams,
) -> Path:
    """
    Synchronous wrapper for handle_training using asyncio.run()
    """
    return asyncio.run(handle_training(training_params)) 