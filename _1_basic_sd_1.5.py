# 1. Text to image using sd_1.5
# Basic version only uses a. base_model_path and b. vae (or simply base_model_path)

import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL

from utils import save_generated_images, show_images_as_grid, basic_sd_output_dir

base_model_path = "SG161222/Realistic_Vision_V5.1_noVAE"
# single_file_model_path = '/ComfyUI/models/checkpoints/Realistic_Vision_V4.0.safetensors'
single_file_model_path = '/ComfyUI/models/checkpoints/Realistic_Vision_V5.1_fp16-no-ema.safetensors'
# SD_1.5 model - This downloads the file in .cache/hugging_face
# Once downloaded could be used by any file in from any directory (no need to download each time)

vae_model_path = "stabilityai/sd-vae-ft-mse"
device = ("cuda")

vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

# Use this if base_model_path is used from hugging face
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    vae=vae,  # Disabled for testing
    torch_dtype=torch.float16
).to(device)  # or in the next line pipe = pipe.to(device)

# Use this if base_model_path is used from a single model locally
# pipe = StableDiffusionPipeline.from_single_file(
#     single_file_model_path,
#     vae=vae,  # Disabled for testing
#     torch_dtype=torch.float16,
#     original_config_file=None
# ).to(device)

# For optimizing speed and memory usage (Note - this is not low vram mode of Comfyui)
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
# pipe.enable_sequential_cpu_offload() - keep this commented generally
pipe.enable_model_cpu_offload()
# if using torch < 2.0 use `pipe.enable_xformers_memory_efficient_attention()`
pipe.enable_xformers_memory_efficient_attention()

prompt = "a cat with its two kittens"
negative_prompt = "text, watermark, deformed, bad quality, nsfw"

batch_size = 4

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

save_generated_images(images, basic_sd_output_dir)
print(f"Generated images saved at {basic_sd_output_dir}")

show_images_as_grid(images)
