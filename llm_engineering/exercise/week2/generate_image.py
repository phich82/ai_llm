import os
import sys
import time

from PIL.ImageFile import ImageFile

# Current fdir
script_path = os.path.realpath(os.path.dirname(__name__))
os.chdir(script_path)
# Append the relative paths
sys.path.append("..")

from ImageGenerator import ImageGenerator

image_prompt = "A futuristic cityscape at sunset"
image_generator: ImageGenerator = ImageGenerator(provider='StableDiffusion', base_url='http://127.0.0.1:7860')
# image_generator: ImageGenerator = ImageGenerator(provider='GPT', base_url='http://127.0.0.1:7860')
# generated_image: ImageFile = image_generator.generate(prompt=image_prompt, size='1024x1024')
# generated_image: ImageFile = image_generator.generate(prompt=image_prompt, size='512x512')
generated_image: ImageFile = image_generator.generate(prompt=image_prompt, size='256x256')
generated_image.show()
save_path = f"images/image_generated_{time.time()}.png"
generated_image.save(save_path)
print(f'A new imaged generated at {save_path}.')
