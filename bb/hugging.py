import fire
import imageio

import requests
from PIL import Image
from io import BytesIO
from diffusers import LDMSuperResolutionPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "CompVis/ldm-super-resolution-4x-openimages"

# load model and scheduler
pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
pipeline = pipeline.to(device)


def upscale(input_path, output_path):
    array_rgb = imageio.v3.imread(input_path)
    image_rgb = Image.fromarray(array_rgb)

    # run pipeline in inference (sample random noise and denoise)
    upscaled_image = pipeline(image_rgb, num_inference_steps=100, eta=1).images[0]
    # save image
    upscaled_image.save(output_path)


if __name__ == "__main__":
    fire.Fire()
