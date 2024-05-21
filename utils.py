import json
import os
from datetime import datetime

from diffusers.utils import make_image_grid

with open('config.json') as f:
    config = json.load(f)

input_dir = config['INPUT_DIR']
mask_dir = config['MASK_DIR']
output_dir = config['OUTPUT_DIR']
os.makedirs(input_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


def save_generated_images(images, output_dir):
    idx = 1
    for image in images:
        png_image = image.convert("RGB")
        img_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f'{output_dir}{img_name}_{idx}.png'
        png_image.save(output_path, 'PNG')
        idx += 1


def get_image_names_from_dir(dir):
    # List files in the directory
    files = os.listdir(dir)
    img_names = []

    for file in files:
        # Check if the file is a .jpg or .png file
        if file.lower().endswith(('.jpg', '.png', 'jpeg')):
            img_names.append(file)

    return img_names


# images: list of PIL images | len(images) == row x col
def show_images_as_grid(images):
    row = 1
    col = 1
    if len(images) == 2:
        col = 2
    elif len(images) == 3:
        col = 3
    elif len(images) == 4:
        row = 2
        col = 2
    elif len(images) == 5:
        col = 5
    elif len(images) == 6:
        row = 2
        col = 3
    elif len(images) == 7:
        col = 7
    elif len(images) == 8:
        row = 2
        col = 4
    elif len(images) == 9:
        row = 3
        col = 3
    else:
        print('Cannot display images as a grid')

    grid = make_image_grid(images, row, col)
    grid.show()
