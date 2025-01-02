from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Optional
from typing import Optional

class TrainingParams(BaseModel):
    trigger_word: str = "TOK"
    autocaption: bool = False
    autocaption_prefix: Optional[str] = None
    autocaption_suffix: Optional[str] = None
    steps: int = 1000
    learning_rate: float = 0.0004
    batch_size: int = 1
    resolution: str = "512,768,1024"
    lora_rank: int = 16
    caption_dropout_rate: float = 0.05
    optimizer: str = "adamw8bit"
    cache_latents_to_disk: bool = False
    layers_to_optimize_regex: Optional[str] = None
    wandb_api_key: Optional[str] = None
    wandb_project: str = train.JOB_NAME
    wandb_run: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_sample_interval: int = 100
    wandb_sample_prompts: Optional[str] = None
    wandb_save_interval: int = 100