import torch
from diffusers import FluxControlPipeline
import os
import sys
from diffusers.utils import load_image
from utils.dataUtils import getCategory
from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '..')


class Flux_depth():
    def __init__(self, device):
        self.device = device
        # Get absolute paths for model files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        models_dir = os.path.join(project_root, "models", "nunchaku")
        text_encoder_path = os.path.join(models_dir, "awq-int4-flux.1-t5xxl.safetensors")
        transformer_path = os.path.join(models_dir, "svdq-int4_r32-flux.1-depth-dev.safetensors")
        cache_dir = os.path.join(project_root, "models")
        
        text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(text_encoder_path)
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(transformer_path)
        self.pipe = FluxControlPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Depth-dev",
            text_encoder_2=text_encoder_2,
            transformer=transformer,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
        ).to(self.device)

    def generate(self, depth_image, flag, size=1024, guidance_scale=10.0, num_inference_steps=30):
        """
        Generate image from depth control image
        
        Parameters:
            depth_image: PIL.Image.Image or numpy array - depth control image
            prompt: str - text prompt for generation
            size: int - output image size (width and height)
            guidance_scale: float - guidance scale for generation
            num_inference_steps: int - number of inference steps
            
        Returns:
            PIL.Image.Image - generated image
        """
        flag = getCategory(flag)
        # prompt = f'A raw photo of a {flag}. {flag} has simple geometric design, no reflections, high quality, rich details. Shot with a macro lens (f/2.8, 50mm) and a Canon EOSR5'
        prompt = f'A raw photo of a {flag}. no reflections, high quality, rich details. Shot with a macro lens (f/2.8, 50mm) and a Canon EOSR5'
        print(prompt)
        return self.pipe(
            prompt=prompt,
            control_image=depth_image,
            height=size,
            width=size,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=512,
        ).images[0]


def depth2img(depth_image, prompt, save_path, size=512):
    """
    Legacy function for backward compatibility
    Generate image from depth control image using Flux depth model
    
    Parameters:
        depth_image: PIL.Image.Image - depth control image
        prompt: str - text prompt for generation
        save_path: str - output image save path
        size: int - output image size (width and height)
    """
    depth_flux = Flux_depth("cuda")
    image = depth_flux.generate(depth_image, prompt, size)
    image.save(save_path)


if __name__ == "__main__":
    # Load depth control image
    depth_image = load_image("workspace/depth.jpeg")
    depth_image = depth_image.resize((512, 512))

    prompt = (
        "An elephant is standing inside a building. "
        "The background is clean."
    )
    
    # Using the new class-based approach
    depth_flux = Depth_Flux("cuda")
    image = depth_flux.generate(depth_image, prompt, size=1024)
    image.save('output_depth.png')