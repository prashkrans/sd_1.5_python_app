import os

import cv2
import numpy as np

from utils import mask_dir, get_image_names_from_dir


def mask_generator(input_dir):
    init_images = []
    mask_images = []

    # Create the input mask directory if it doesn't exist
    os.makedirs(mask_dir, exist_ok=True)

    img_names_with_ext = get_image_names_from_dir(input_dir)
    print(f'Found {len(img_names_with_ext)} number of input images')

    create_masks = input('Do you want to create masks (y/n)? ')
    if create_masks == 'y':

        for img_name_with_ext in img_names_with_ext:

            img_name = img_name_with_ext.split('.')[0]
            img_path = f'{input_dir}{img_name_with_ext}'

            # Input could be .jpg or .png but saving as .png (as png is a loss less format)

            # Step 1: Take input of an image from the user
            image = cv2.imread(img_path)
            init_images.append(image)  # Creating a list of numpy nd arrays

            # Step 2: Open the image in a 1280x1280 window
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Image", 1280, 1280)
            cv2.imshow("Image", image)

            # Step 3: Allow the user to select a mask on the image using the mouse
            mask = np.ones_like(image, dtype=np.uint8) * 255  # initialize mask as white image
            drawing = False
            brush_size = 30  # Default brush size | Ideal brush size = 30-50

            def draw_mask(event, x, y, flags, param):
                global drawing
                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
                    cv2.circle(mask, (x, y), brush_size, (0, 0, 0), -1)

                elif event == cv2.EVENT_MOUSEMOVE:
                    if drawing:
                        cv2.circle(mask, (x, y), brush_size, (0, 0, 0), -1)

                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False
                    cv2.circle(mask, (x, y), brush_size, (0, 0, 0), -1)

            cv2.setMouseCallback("Image", draw_mask)

            while True:
                cv2.imshow("Image", cv2.addWeighted(image, 0.7, mask, 0.3, 0))
                key = cv2.waitKey(1) & 0xFF
                # Scroll up to zoom in while scroll down to zoom out
                if key == 27:  # Press 'Esc' to reset the mask
                    mask = np.ones_like(image, dtype=np.uint8) * 255
                elif key == 13:  # Press 'Enter' to save the selected mask and exit
                    break
                elif key == ord('=') or key == 82:  # Increase brush size by pressing '=' or up arrow)
                    brush_size += 2
                elif key == ord(
                        '-') or key == 84 and brush_size > 4:  # Decrease brush size by pressing '-' or down arrow)
                    brush_size -= 2

            # Avoid this step : Save the mask in black and white in the same size as the image
            # mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # It is usually not a problem, but it has one dimension lesser than the source image
            # print(img.shape) => (h, w, 3) - 3 dimensions while print(mask.shape) => (h, w) - 2 dimensions

            # Step 4: Inverting the mask obtained originally as simple lama inpainting works the opposite
            # inverted_mask = cv2.bitwise_not(mask_gray)
            inverted_mask = cv2.bitwise_not(mask)
            mask_images.append(inverted_mask)

            # Step 5: Saving the masked image
            mask_path = f'{mask_dir}mask_{img_name}.png'
            cv2.imwrite(mask_path, inverted_mask)
            print(f'Mask saved at {mask_path}')

            cv2.destroyAllWindows()
    else:
        print('Skipping mask generation as masks were already generated')

    return init_images, mask_images, img_names_with_ext
