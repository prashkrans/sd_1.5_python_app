import cv2
import torch
from PIL import Image
from datetime import datetime
from diffusers import StableDiffusionInpaintPipeline, AutoencoderKL, DDIMScheduler

from descale_image import descale_image
from utils import input_dir, mask_dir, show_images_as_grid


#  The area to inpaint is represented by white pixels and the area to keep is represented by black pixels. The white
#  pixels are filled in by the prompt.
def inpaint(init_images, mask_images, img_names_with_ext, output_dir):
    print('Initiating the inpainting process:')
    torch.cuda.empty_cache()  # Not as useful, could be skipped
    # The below if block is only used when masks were already generated
    if (len(init_images) == 0 and len(img_names_with_ext) != 0):
        print('Extracting images and their corresponing masks previously created')
        for img_name_with_ext in img_names_with_ext:
            init_image_path = f'{input_dir}{img_name_with_ext}'
            img_name = img_name_with_ext.split('.')[0]
            mask_image_path = f'{mask_dir}mask_{img_name}.png'
            init_images.append(cv2.imread(init_image_path))
            mask_images.append(cv2.imread(mask_image_path))
    else:
        print('Using images and their newly created corresponing masks')

    for np_init_image, np_mask_image, img_name_with_ext in zip(init_images, mask_images, img_names_with_ext):
        np_init_image = descale_image(np_init_image)
        np_mask_image = descale_image(np_mask_image)

        cv2.imshow("Image", cv2.addWeighted(np_init_image, 0.7, np_mask_image, 0.3, 0))
        cv2.waitKey(3000)  # Waits for 3 seconds or user input to close the window
        cv2.destroyAllWindows()

        # Converting np array images to PIL images as pipeline works with PIL images
        init_image = Image.fromarray(cv2.cvtColor(np_init_image, cv2.COLOR_BGR2RGB))
        mask_image = Image.fromarray(cv2.cvtColor(np_mask_image, cv2.COLOR_BGR2RGB))

        img_name = img_name_with_ext.split('.')[0]
        inpainted_img_path = f'{output_dir}{img_name}_inpaint'
        width, height = init_image.size
        # It is mandatory to have width and height divisible by 8. So, make both width and height multiples of 8
        # E.g. if size = (512, 684) then we make new size as (512, 680)
        rem_width = width % 8
        rem_height = height % 8
        width -= rem_width
        height -= rem_height

        # Note - SD_1.5 or SDXL base models don't work for inpainting, hence use the inpainting models only
        # base_model_path = "SG161222/Realistic_Vision_V4.0-inpainting" - this won't work here
        # base_model_path = "runwayml/stable-diffusion-inpainting" # This inpainting model works but has censorship

        # SD_1.5 Inpainting works great while SDXL inpainting gives CUDA OOM
        # Realistic Vision is one of the best models for both inpainting and image generation
        # This downloads the file in .cache/hugging_face
        # Once downloaded could be used by any file in from any directory (no need to download each time)
        inpainting_base_model_path = 'https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/blob/main/Realistic_Vision_V5.1-inpainting.safetensors'
        # or,
        # inpainting_base_model_path = ('/home/nashprat/workspace/comfy_ui_tools/ComfyUI/models/checkpoints/epicrealism_pureEvolutionV5-inpainting.safetensors')
        vae_model_path = "stabilityai/sd-vae-ft-mse"
        device = ("cuda")
        vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

        # https://huggingface.co/docs/diffusers/en/api/schedulers/ddim#:~:text=Choose%20from%20linear%2C%20scaled_linear%2C%20or%20squaredcos_cap_v2.
        # May or may not use this scheduler, as by comparison, it performed slightly worse than
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        # Note - Used .from_single_file here instead of from_pretrained to use the inpainting_model specifically
        # pipe = StableDiffusionInpaintPipeline.from_pretrained(
        inpaint_pipe = StableDiffusionInpaintPipeline.from_single_file(
            inpainting_base_model_path,
            torch_dtype=torch.float16,
            # scheduler=noise_scheduler,
            vae=vae,
            # feature_extractor=None,
            # safety_checker=None,
            # original_config_file=None
        ).to(device)

        # For optimizing speed and memory usage (Note - this is not low vram mode of Comfyui)
        # pipe.enable_vae_slicing() # Both slicing and tiling were not supported by inpaint
        # pipe.enable_vae_tiling()
        # pipe.enable_sequential_cpu_offload() - keep this commented generally
        inpaint_pipe.enable_model_cpu_offload()
        # if using torch < 2.0 use `pipe.enable_xformers_memory_efficient_attention()`
        inpaint_pipe.enable_xformers_memory_efficient_attention()

        print(f'Inpainting image {img_name_with_ext}')
        prompt = input('Please input positive prompt: ')
        negative_prompt = """
        text, watermark, deformed, bad quality, (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, 
        sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), 
        poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, 
        mutation, mutated, ugly, disgusting, amputation
        """
        # additional_negative_prompt = input('Please input any additional negative prompt [Optional]: ')
        # negative_prompt += additional_negative_prompt

        # Inpaint pipeline params:
        padding_mask_crop = 6  # Generally a value of 2-10 is considered good
        strength = 1  # "The value of strength should be between [0.0, 1.0]
        num_inference_steps = 30  # Default is 50, but 20 to 40 works great | Ideal = 30 | Max = 63
        guidance_scale = 7.0  # a.k.a cfg value in comfy ui | Ideal = 7.0
        batch_size = 2  # Ideal to generate at-least 4 images | If getting cuda OOM error, simply reduce it 2 or lower
        generator = torch.Generator(device).manual_seed(2024)  # seed number could be any random number

        inpainted_images = inpaint_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            mask_image=mask_image,
            height=height,
            width=width,
            padding_mask_crop=padding_mask_crop,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=batch_size,
            generator=generator
        ).images

        show_images_as_grid(inpainted_images)

        current_datetime = datetime.now()
        date_time_suffix = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")

        for i, inpainted_image in enumerate(inpainted_images, start=1):
            inpainted_image.save(f'{inpainted_img_path}_{i}_{date_time_suffix}.png', 'PNG')

    print(f"All inpainted images saved at {output_dir}")
    print(f"Inpainting process completed successfully")


from mask_generator import mask_generator
from utils import input_dir, output_dir

if __name__ == "__main__":
    init_images, mask_images, img_names_with_ext = mask_generator(input_dir)
    inpaint(init_images, mask_images, img_names_with_ext, output_dir)
