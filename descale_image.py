import cv2


def descale_image(input_image):
    # Get the original image dimensions
    height, width = input_image.shape[:2]

    # Here, min performs better than max, since the current model works well with 512 pixels - only for inpainting
    # Since we have to upscale the inpainted low res image, 1024 is a better choice
    # Using min as 1024/1024 = 1 as inpainted image once descaled can be used again without descaling
    # multiple revisions could be done without descaling - [Verified]
    descale_factor = min(1024 / height, 1024 / width)  # Verified 1024 x 1024 works perfectly fine
    # descale_factor = min(512 / height, 512 / width)  # 512 x 512 can actually be avoided | Used only for testing
    print(f'descale_factor = {descale_factor}')

    # Calculate the new dimensions while maintaining the aspect ratio
    if descale_factor < 1:
        width = int(width * descale_factor)
        height = int(height * descale_factor)

    # Resize the image using OpenCV
    descaled_image = cv2.resize(input_image, (width, height), interpolation=cv2.INTER_AREA)

    return descaled_image
