import os
import sys
import shutil
import subprocess
import logging
import asyncio
from functools import partial

# Add necessary paths
sys.path.append("flux_server/ai-toolkit")

import time
from pathlib import Path
from typing import Optional, OrderedDict
from zipfile import ZipFile, is_zipfile

import torch
from huggingface_hub import HfApi

from .wandb_client import WeightsAndBiasesClient, logout_wandb
from .layer_match import match_layers_to_optimize, available_layers_to_optimize
from .submodule_patches import patch_submodules

from .caption import Captioner
from jobs import BaseJob
from toolkit.config import get_config
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from .custom_types import TrainingParams

# Set environment variables
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["LANG"] = "en_US.UTF-8"

patch_submodules()

JOB_NAME = "flux_train_replicate"
WEIGHTS_PATH = Path("./FLUX.1-dev")
INPUT_DIR = Path("input_images")
OUTPUT_DIR = Path("output")
JOB_DIR = OUTPUT_DIR / JOB_NAME

logger = logging.getLogger(__name__)

class CustomSDTrainer(SDTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_samples = set()
        self.wandb = None

    def hook_train_loop(self, batch):
        loss_dict = super().hook_train_loop(batch)
        if self.wandb:
            self.wandb.log_loss(loss_dict, self.step_num)
        return loss_dict

    def sample(self, step=None, is_first=False):
        super().sample(step=step, is_first=is_first)
        output_dir = JOB_DIR / "samples"
        all_samples = set([p.name for p in output_dir.glob("*.jpg")])
        new_samples = all_samples - self.seen_samples
        if self.wandb:
            image_paths = [output_dir / p for p in sorted(new_samples)]
            self.wandb.log_samples(image_paths, step)
        self.seen_samples = all_samples

    def post_save_hook(self, save_path):
        super().post_save_hook(save_path)
        lora_path = JOB_DIR / f"{JOB_NAME}.safetensors"
        if not lora_path.exists():
            lora_path = sorted(JOB_DIR.glob("*.safetensors"))[-1]
        if self.wandb:
            print(f"Saving weights to W&B: {lora_path.name}")
            self.wandb.save_weights(lora_path)

class CustomJob(BaseJob):
    def __init__(self, config: OrderedDict, wandb_client: Optional[WeightsAndBiasesClient] = None):
        super().__init__(config)
        self.device = self.get_conf("device", "cpu")
        self.process_dict = {"custom_sd_trainer": CustomSDTrainer}
        self.load_processes(self.process_dict)
        for process in self.process:
            process.wandb = wandb_client

    def run(self):
        super().run()
        print(f"Running {len(self.process)} process{'' if len(self.process) == 1 else 'es'}")
        for process in self.process:
            process.run()

def clean_up():
    logout_wandb()
    # if INPUT_DIR.exists():
    #     shutil.rmtree(INPUT_DIR)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

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
    if not WEIGHTS_PATH.exists():
        t1 = time.time()
        subprocess.check_output([
            "pget",
            "-xf",
            "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar",
            str(WEIGHTS_PATH.parent),
        ])
        t2 = time.time()
        print(f"Downloaded base weights in {t2 - t1} seconds")

async def handle_training(
    training_params: TrainingParams,
) -> Path:
    """Handle the training process with parameters from TrainingParams model"""
    logger.info("Starting training process")
    
    # Add mock handling at the start
    if training_params.mock_training:
        logger.info("Using mock training mode")
        mock_path = Path(training_params.mock_output_path)
        # Create a mock tar file
        if not mock_path.parent.exists():
            mock_path.parent.mkdir(parents=True)
        os.system(f"touch {mock_path}")
        return mock_path

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
    if training_params.wandb_sample_prompts:
        sample_prompts = [p.strip() for p in training_params.wandb_sample_prompts.split("\n")]

    # Create training config
    train_config = create_training_config(
        training_params.trigger_word, training_params.steps, 
        training_params.learning_rate, training_params.batch_size,
        training_params.resolution, training_params.lora_rank,
        training_params.caption_dropout_rate, training_params.optimizer,
        training_params.cache_latents_to_disk, layers_to_optimize,
        training_params.wandb_api_key, training_params.wandb_save_interval,
        training_params.wandb_sample_interval, sample_prompts
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
    os.system(f"tar -cvf {output_path} {JOB_DIR}")
    logger.info("Training completed successfully")
    return output_path

def create_training_config(trigger_word, steps, learning_rate, batch_size, resolution,
                         lora_rank, caption_dropout_rate, optimizer, cache_latents_to_disk,
                         layers_to_optimize, wandb_api_key, wandb_save_interval,
                         wandb_sample_interval, sample_prompts):
    train_config = OrderedDict(
        {
            "job": "custom_job",
            "config": {
                "name": JOB_NAME,
                "process": [
                    {
                        "type": "custom_sd_trainer",
                        "training_folder": str(OUTPUT_DIR),
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
                                "folder_path": str(INPUT_DIR),
                                "caption_ext": "txt",
                                "caption_dropout_rate": caption_dropout_rate,
                                "shuffle_tokens": False,
                                "cache_latents_to_disk": cache_latents_to_disk,
                                "cache_latents": True,
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
                            "sample_every": (
                                wandb_sample_interval
                                if wandb_api_key and sample_prompts
                                else steps + 1
                            ),
                            "width": 1024,
                            "height": 1024,
                            "prompts": sample_prompts,
                            "neg": "",
                            "seed": 42,
                            "walk_seed": True,
                            "guidance_scale": 3.5,
                            "sample_steps": 28,
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
    lora_file = JOB_DIR / f"{JOB_NAME}.safetensors"
    lora_file.rename(JOB_DIR / "lora.safetensors")

    samples_dir = JOB_DIR / "samples"
    if samples_dir.exists():
        shutil.rmtree(samples_dir)

    # Remove any intermediate lora paths
    lora_paths = JOB_DIR.glob("*.safetensors")
    for path in lora_paths:
        if path.name != "lora.safetensors":
            path.unlink()

    # Optimizer is used to continue training, not needed in output
    optimizer_file = JOB_DIR / "optimizer.pt"
    if optimizer_file.exists():
        optimizer_file.unlink()

    # Copy generated captions to the output tar
    captions_dir = JOB_DIR / "captions"
    captions_dir.mkdir(exist_ok=True)
    for caption_file in INPUT_DIR.glob("*.txt"):
        shutil.copy(caption_file, captions_dir) 

def handle_training_sync(
    training_params: TrainingParams,
) -> Path:
    """
    Synchronous wrapper for handle_training using asyncio.run()
    """
    return asyncio.run(handle_training(training_params)) 