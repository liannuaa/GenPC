from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler
from PIL import Image
import torch
import numpy as np
import cv2
import os
from utils.dataUtils import getCategory


class ControlNet_Depth():
    def __init__(self, device="cuda", cache_dir="./models"):
        self.device = device
        self.cache_dir = cache_dir
        self._load_models()
    
    def _load_models(self):
        """Load ControlNet models and pipeline"""
        print("Loading ControlNet Depth models...")
        
        # Load scheduler
        self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            subfolder="scheduler"
        )
        
        # Load ControlNet
        self.controlnet = ControlNetModel.from_pretrained(
            "xinsir/controlnet-depth-sdxl-1.0",
            torch_dtype=torch.float16,
            cache_dir=self.cache_dir
        ).to(self.device)
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch.float16,
            cache_dir=self.cache_dir
        )
        
        # Create pipeline
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=self.controlnet,
            vae=self.vae,
            safety_checker=None,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16,
            scheduler=self.scheduler,
        ).to(self.device)
        
        print("ControlNet Depth models loaded successfully!")

    # self.image = self.depth2Image.generate(self.depth, flag, size=self.cfg.generate_res)
    def generate(self, img, flag, size, controlnet_conditioning_scale=1.0, 
                 num_inference_steps=30, save_path=None):
        """
        Generate image from depth control image
        
        Parameters:
            img: numpy array - depth control image
            flag: str - object category flag
            controlnet_conditioning_scale: float - conditioning scale
            num_inference_steps: int - number of inference steps
            save_path: str - optional save path
            
        Returns:
            PIL.Image.Image - generated image
        """
        category = getCategory(flag)
        prompt = f"A photo of {category}, 3d model, high resolution,high quality,highly detailed,highly realistic,clean look,no shadow,"
        negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

        # Resize image to optimal resolution
        controlnet_img = img
        # Support both PIL.Image and numpy array input
        if isinstance(controlnet_img, Image.Image):
            controlnet_img = controlnet_img.convert("RGB")
            width, height = controlnet_img.size
            ratio = np.sqrt(size * size / (width * height))
            new_width, new_height = max(1, int(width * ratio)), max(1, int(height * ratio))
            controlnet_img = controlnet_img.resize((new_width, new_height), Image.LANCZOS)
        else:
            # Assume numpy array (H, W, C)
            if controlnet_img.ndim == 3 and controlnet_img.shape[2] == 3:
                height, width, _ = controlnet_img.shape
                ratio = np.sqrt(size * size / (width * height))
                new_width, new_height = max(1, int(width * ratio)), max(1, int(height * ratio))
                controlnet_img = cv2.resize(controlnet_img, (new_width, new_height))
                # Convert BGR to RGB if needed (cv2.imread gives BGR)
                controlnet_img = cv2.cvtColor(controlnet_img, cv2.COLOR_BGR2RGB)
                controlnet_img = Image.fromarray(controlnet_img)
            else:
                raise TypeError("Unsupported image type: expected PIL.Image or numpy array with shape (H, W, 3)")
        
        # Generate image
        images = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            image=controlnet_img,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            width=new_width,
            height=new_height,
            num_inference_steps=num_inference_steps,
        ).images
        
        # Save if path provided
        if save_path:
            images[0].save(save_path)
        
        return images[0]


# Legacy function for backward compatibility
def controlnet_depth(img, flag):
    """
    Legacy function - creates a new instance each time (not recommended for multiple calls)
    """
    controlnet_depth_model = ControlNet_Depth()
    return controlnet_depth_model.generate(img, flag, save_path='zz.png')


if __name__ == '__main__':
    # Example usage with new class
    img = cv2.imread(f"./zzz/09639/depth.png")
    
    # Using the new class-based approach (recommended)
    controlnet_model = ControlNet_Depth()
    result = controlnet_model.generate(img, flag='swivel chair', save_path='output_controlnet.png')