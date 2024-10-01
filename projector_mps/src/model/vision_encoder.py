from PIL import Image
import requests
import torch
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification, AutoImageProcessor, AutoModel
import clip
import os
from torchvision import transforms as T

class VisionEncoder:
    def __init__(self, model_path):
        print(f"Loading Vision Encoder with model_path: {model_path}")
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.model_path = model_path
        if 'openai/clip' in model_path:
            # See issue here: https://huggingface.co/openai/clip-vit-large-patch14/discussions/1
            # Personally, I've found original CLIP implementation to be much faster.
            if model_path == 'openai/clip-vit-large-patch14-336':
                clip_model = 'ViT-L/14@336px'
            elif model_path == 'openai/clip-vit-large-patch14':
                clip_model = 'ViT-L/14'
            elif model_path == 'openai/clip-vit-base-patch16':
                clip_model = 'ViT-B/16'
            print(f"Using clip_model: {clip_model}")
            self.model, self.processor = clip.load(clip_model, device=self.device)
        elif 'facebook/dinov2' in model_path:
            self.processor = T.Compose([
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg').to(self.device)
        else:
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = AutoModelForZeroShotImageClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def encode_images(self, images_files):
        images = [Image.open(img).convert('RGB') for img in images_files]
        if 'openai/clip' in self.model_path:
            images = [self.processor(img).unsqueeze(0).to(self.device) for img in images]
            with torch.no_grad():
                image_features = self.model.encode_image(torch.cat(images))
        elif 'facebook/dinov2' in self.model_path:
            images = [self.processor(img).unsqueeze(0).to(self.device) for img in images]
            with torch.no_grad():
                image_features = self.model.forward_features(torch.cat(images))["x_norm_clstoken"]
        else:
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
        return image_features

