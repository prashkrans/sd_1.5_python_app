# 1. Text to image using sd_1.5
# Basic version only uses a. base_model_path and b. vae (or simply base_model_path)

import os
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL

from utils import save_generated_images, show_images_as_grid

# base_model_path = "SG161222/Realistic_Vision_V5.1_noVAE" # Using single_file _model_path instead of base_model_path
# single_file_model_path = '/home/nashprat/workspace/comfy_ui_tools/ComfyUI/models/checkpoints/Realistic_Vision_V4.0.safetensors'
single_file_model_path = '/home/nashprat/workspace/comfy_ui_tools/ComfyUI/models/checkpoints/Realistic_Vision_V5.1_fp16-no-ema.safetensors'
# Download the sd_1.5 model from either civit.ai or hugging face and use directly from local without any config.json
# file
# Once downloaded could be used by any file in from any directory (no need to download each time)

vae_model_path = "stabilityai/sd-vae-ft-mse"
device = ("cuda")

vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

pipe = StableDiffusionPipeline.from_single_file(
    single_file_model_path,
    vae=vae,  # Disabled for testing
    torch_dtype=torch.float16,
    original_config_file=None
).to(device)

# For optimizing speed and memory usage (Note - this is not low vram mode of Comfyui)
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
# pipe.enable_sequential_cpu_offload() - keep this commented generally
pipe.enable_model_cpu_offload()
# if using torch < 2.0 use `pipe.enable_xformers_memory_efficient_attention()`
pipe.enable_xformers_memory_efficient_attention()

prompt = "a ghibli style art of a woman"
negative_prompt = "text, watermark, deformed, bad quality, nsfw"

batch_size = 4

output_dir = "/home/nashprat/workspace/sd_1.5_python_app/output_dir_sd_1.5/basic_sd/"
os.makedirs(output_dir, exist_ok=True)

# Text2Img documentation - https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/text2img
# List of PIL images <PIL.Image.Image>
images = pipe(
    prompt,
    height=512,
    width=512,
    negative_prompt=negative_prompt,
    num_inference_steps=35,  # Default = 50 (ideal)
    guidance_scale=4,  # cfg in comfyui
    num_images_per_prompt=batch_size
).images

save_generated_images(images, output_dir)
print(f"Generated images saved at {output_dir}")

show_images_as_grid(images)
