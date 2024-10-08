# Stable Diffusion 1.5 Python App
Basic usage of stable diffusion 1.5 including `text to image generation` and `inpainting`.

### Demo Videos:
SD 1.5 Text to Image Generation Demo:

https://github.com/user-attachments/assets/0d9afd14-9476-4658-a798-53369cba312e

SD 1.5 Inpainting Demo:

https://github.com/user-attachments/assets/521a6a2d-7f13-4203-82b0-2487ba495c13


### Prerequisites:
- Python 3.10 (Might work with other versions as well)
- Nvidia drivers installed along with CUDA
- Nvidia GPU with VRAM atleast 6GB
- CPU can also be used but is awfully slow

### Setup:
1. Clone the repo and move to the root dir.
2. Create a python virtual environment.
3. Install torch separately, then install the requirements (Might take some time).
```commandline
pip install torch==2.3.0
pip install -r requirements.txt
```
4. Either download sd 1.5 models manually or use hugging face models auto-download option.

### Usage:
- Run either of the _1_ or _2_ for text to image generation, give a prompt and wait for the image generation. 
- Run _3_ for inpainting, mask the source image (./input_dir/source_image.jpg), give appropriate prompt and watch the magic happen.

### License
This python app, stable diffusion code and model weights are released under the MIT License. See [LICENSE](LICENSE)
for further details.

### Responsible usage of Stable Diffusion
By forking or cloning this repository, you agree to adhere to responsible usage guidelines. This ensures that the
powerful capabilities of Stable Diffusion are used ethically and do not cause harm. Below are the key principles and
practices to follow:

**1. Ethical Use:** Do not use this technology for generating content that is harmful, offensive, or illegal. This
includes,
but is not limited to, the creation of:

- Pornographic or sexually explicit material.
- Violent or graphic content intended to shock or disturb.
- Discriminatory or hateful imagery against individuals or groups based on race, ethnicity, religion, gender, sexual
  orientation, disability, or any other characteristic.
- Misleading information, including deepfakes, that could spread misinformation or false narratives.

**2. Respect Privacy:** Ensure that the generated images do not infringe on the privacy rights of individuals. Avoid
creating
images that depict real people without their consent, especially in compromising or damaging contexts.

**3. Intellectual Property:** Respect intellectual property rights by not using this technology to create and distribute
images that infringe on copyrights, trademarks, or other proprietary rights. Ensure that generated content is either
original
or used in a manner compliant with fair use guidelines.

**4. Transparency and Accountability:** Be transparent about the use of AI-generated content. Clearly disclose when
images are created using Stable Diffusion, especially in contexts where the origin of the content might be significant (
e.g.,
news, research publications, social media).

**5. Mitigate Bias:** Be aware of and actively mitigate any biases present in the AI model. Avoid generating content
that reinforces harmful stereotypes or biases. Strive for fairness and inclusivity in the images you create.

**6. Environmental Considerations:** Use computational resources efficiently and be mindful of the environmental impact
of running large models. Where possible, optimize your use of hardware and consider the energy consumption associated
with
training and inference processes.

**7. Continuous Learning and Improvement:** Stay informed about the latest developments, best practices, and ethical
guidelines in the field of AI and machine learning. Be open to updating your practices in line with new insights and
recommendations.

By following these guidelines, you contribute to the responsible development and use of AI technologies, ensuring they
benefit society while minimizing potential harm.
