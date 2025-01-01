# First, in your terminal.
#
# $ python3 -m virtualenv env
# $ source env/bin/activate
# $ pip install torch torchvision transformers sentencepiece protobuf accelerate
# $ pip install git+https://github.com/huggingface/diffusers.git
# $ pip install optimum-quanto

import torch
from quanto import quantize, freeze, safe_save, qfloat8, safe_load
from pathlib import Path
import logging

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast

dtype = torch.bfloat16

# schnell is the distilled turbo model. For the CFG distilled model, use:
bfl_repo = "black-forest-labs/FLUX.1-dev"
revision = "main"

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", revision=revision)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision)
tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype, revision=revision)
vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision)
transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype, revision=revision)

# Experimental: Try this to load in 4-bit for <16GB cards.
#
# from optimum.quanto import qint4
# quantize(transformer, weights=qint4, exclude=["proj_out", "x_embedder", "norm_out", "context_embedder"])
# freeze(transformer)

WEIGHTS_CACHE = Path("weights_cache")

# Set up logging
logging.basicConfig(
    filename='inference_demo.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Log the console output
# logging.info("Quantizing transformer")
# quantize(transformer, weights=qfloat8)
# freeze(transformer)
# safe_save(transformer.state_dict(), WEIGHTS_CACHE / "transformer.sd")

# logging.info("Quantizing text_encoder_2")
# quantize(text_encoder_2, weights=qfloat8)
# freeze(text_encoder_2)
# safe_save(text_encoder_2.state_dict(), WEIGHTS_CACHE / "text_encoder_2.sd")

# logging.info("Loading quantized transformer weights")
# transformer.to_empty(device="cuda")
# state_dict = safe_load(WEIGHTS_CACHE / "transformer.sd")
# transformer.load_state_dict(state_dict)

logging.info("Creating pipeline")
pipe = FluxPipeline(
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=None,
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=None,
)
pipe.text_encoder_2 = text_encoder_2
pipe.transformer = transformer
pipe.enable_model_cpu_offload()
# pipe.to("cuda")

generator = torch.Generator().manual_seed(12345)
logging.info("Generating image")
image = pipe(
    prompt='nekomusume cat girl, digital painting', 
    width=1024,
    height=1024,
    num_inference_steps=4, 
    generator=generator,
    guidance_scale=3.5,
).images[0]
image.save('test_flux_distilled.png')