from pathlib import Path
from typing import Optional, OrderedDict

from jobs import BaseJob
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from tqdm import tqdm

from .wandb_client import WeightsAndBiasesClient

JOB_NAME = "flux_train_replicate"
JOB_DIR = Path("output") / JOB_NAME

class CustomSDTrainer(SDTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_samples = set()
        self.wandb = None
        self.pbar = None

    def hook_train_loop(self, batch):
        if self.pbar is None:
            self.pbar = tqdm(total=self.max_steps, desc="Training progress")
            
        loss_dict = super().hook_train_loop(batch)
        if self.wandb:
            self.wandb.log_loss(loss_dict, self.step_num)
            
        self.pbar.update(1)
        self.pbar.set_postfix(loss=f"{loss_dict['loss']:.4f}")
        
        return loss_dict

    def cleanup(self):
        if self.pbar is not None:
            self.pbar.close()
        super().cleanup()

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