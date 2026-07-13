import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def load_dinov2(model_name, device, local_files_only):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
    model = model.to(device).eval()
    patch_size = int(getattr(model.config, "patch_size", 14))
    hidden_size = int(getattr(model.config, "hidden_size", 0))
    return model, {"patch_size": patch_size, "hidden_size": hidden_size}


def resize_to_patch_multiple(image, patch_size):
    width, height = image.size
    resized_width = max(patch_size, int(round(width / patch_size)) * patch_size)
    resized_height = max(patch_size, int(round(height / patch_size)) * patch_size)
    if (resized_width, resized_height) == (width, height):
        return image, (width, height)
    return image.resize((resized_width, resized_height), Image.Resampling.BICUBIC), (resized_width, resized_height)


def extract_feature_grid(model, image, patch_size, device):
    resized, model_input_size = resize_to_patch_multiple(image, patch_size)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    tensor = transform(resized)[None].to(device)
    with torch.no_grad():
        output = model(pixel_values=tensor)
    tokens = output.last_hidden_state[:, 1:, :]
    grid_h = model_input_size[1] // patch_size
    grid_w = model_input_size[0] // patch_size
    features = tokens.reshape(1, grid_h, grid_w, tokens.shape[-1]).permute(0, 3, 1, 2)
    features = F.normalize(features, p=2, dim=1)[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
    return features, model_input_size


def run(args):
    image_path = Path(args.image)
    output_path = Path(args.output)
    image = Image.open(image_path).convert("RGB")
    model, model_info = load_dinov2(args.model_name, args.device, args.local_files_only)
    features, model_input_size = extract_feature_grid(
        model,
        image,
        patch_size=model_info["patch_size"],
        device=args.device,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=features,
        image_size=np.asarray(image.size, dtype=np.int32),
        model_input_size=np.asarray(model_input_size, dtype=np.int32),
        model_name=np.asarray(args.model_name),
        patch_size=np.asarray(model_info["patch_size"], dtype=np.int32),
        hidden_size=np.asarray(model_info["hidden_size"], dtype=np.int32),
        source_image=np.asarray(str(image_path)),
    )
    print(
        json.dumps(
            {
                "image": str(image_path),
                "output": str(output_path),
                "model": args.model_name,
                "image_size": list(image.size),
                "model_input_size": list(model_input_size),
                "feature_grid": list(features.shape),
            },
            indent=2,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="facebook/dinov2-large")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
