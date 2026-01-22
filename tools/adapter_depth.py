from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
from controlnet_aux.midas import MidasDetector
import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from utils.utils_2d import *
from utils.dataUtils import *


class Adapter_Depth():
    def __init__(self, device="cuda", cache_dir="./models"):
        self.device = device
        self.cache_dir = cache_dir
        self._load_models()
    
    def _load_models(self):
        """Load T2I Adapter models and pipeline"""
        print("Loading T2I Adapter Depth models...")
        
        # Load adapter
        self.adapter = T2IAdapter.from_pretrained(
            "TencentARC/t2i-adapter-depth-zoe-sdxl-1.0", 
            torch_dtype=torch.float16, 
            varient="fp16",
            cache_dir=self.cache_dir
        ).to(self.device)
        
        # Load scheduler
        model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
        self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            model_id, 
            subfolder="scheduler",
            cache_dir=self.cache_dir
        )
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch.float16,
            cache_dir=self.cache_dir
        )
        
        # Create pipeline
        self.pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            model_id, 
            vae=self.vae, 
            adapter=self.adapter, 
            scheduler=self.scheduler, 
            torch_dtype=torch.float16, 
            variant="fp16",
            cache_dir=self.cache_dir,
        ).to(self.device)
        
        self.pipe.enable_xformers_memory_efficient_attention()
        print("T2I Adapter Depth models loaded successfully!")
    
    def generate(self, image, flag, num_inference_steps=30, 
                 adapter_conditioning_scale=1, guidance_scale=7.5):
        """
        Generate image from depth control image using T2I Adapter
        
        Parameters:
            image: PIL.Image.Image or numpy array - depth control image
            flag: str - object category flag
            num_inference_steps: int - number of inference steps
            adapter_conditioning_scale: float - adapter conditioning scale
            guidance_scale: float - guidance scale
            
        Returns:
            PIL.Image.Image - generated image
        """
        category = getCategory(flag)
        prompt = f"A photo of {category}, high resolution,high quality,highly detailed,highly realistic,clean look,no shadow,"
        negative_prompt = "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"
        
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Resize image
        image = transforms.Resize((1024, 1024))(image)
        
        print(" From depth to image...")
        gen_image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            adapter_conditioning_scale=adapter_conditioning_scale,
            guidance_scale=guidance_scale,
        ).images[0]
        
        return gen_image


# Legacy function for backward compatibility
def Adapter_depth(image, flag):
    """
    Legacy function - creates a new instance each time (not recommended for multiple calls)
    """
    adapter_depth_model = Adapter_Depth()
    return adapter_depth_model.generate(image, flag)

def pcn_t2i_depth(category, name=None):
    """
    Batch processing function for PCN depth to image conversion
    """
    adapter_model = Adapter_Depth()
    
    prompt = f"A photo of a {name} ,3d model,smooth surface, clean look, no texture"
    
    path = os.path.join('workspace', category)
    file_number = len([f for f in os.listdir(path) if f.startswith(category)])
    i = 1
    
    for flag in os.listdir(path):
        if not flag.startswith(category):
            continue
        print(f'-------------{i}/{file_number}---{flag}--------')
        depth_path = os.path.join(path, flag, 'depth.png')
        image = load_image(depth_path)
        
        gen_image = adapter_model.generate(image, flag, num_inference_steps=20)
        gen_image.save(f'{path}/{flag}/t2i_{flag}.png')
        i += 1


if __name__ == "__main__":
    flag = "swivel chair"
    image = load_image(f'workspace/{flag}/depth.png')
    
    # Using the new class-based approach
    adapter_model = Adapter_Depth()
    t2i_image = adapter_model.generate(image, flag)
    t2i_image.save(f'output_adapter_{flag.replace(" ", "_")}.png')

