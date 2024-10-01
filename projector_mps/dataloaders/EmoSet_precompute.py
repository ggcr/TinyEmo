import os
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
import random
import sys
import json

class EmosetDataset_precompute():
    def __init__(self, vision_encoder, phase: str = 'train'):
        # phase must be 'train', 'val' or 'test'
        print(f'EmoSet phase: {phase}')
        if vision_encoder == 'openai/clip-vit-large-patch14':
            dataset_root_path = '/home/cgutierrez/shared/features_clip-vit-large-patch14/EmoSet/'
        elif vision_encoder == 'openai/clip-vit-large-patch14-336':
            dataset_root_path = '/home/cgutierrez/shared/features_clip-vit-large-patch14-336/EmoSet/'
        elif vision_encoder == 'openai/clip-vit-base-patch16':
            dataset_root_path = '/home/cgutierrez/shared/features_clip-vit-base-patch16/EmoSet/'
        elif vision_encoder == 'facebook/dinov2-large':
            dataset_root_path = '/home/cgutierrez/shared/features_dinov2-large/EmoSet/'
        elif vision_encoder == 'google/siglip-so400m-patch14-384':
            dataset_root_path = '/home/cgutierrez/shared/features_siglip-so400m-patch14-384/EmoSet/'
        self.phase = phase
        if self.phase not in ['train', 'validation', 'test']:
            raise Exception(f"dataloaders/WEBEmo.py: Not a valid phase chosen: {self.phase}")
        self.path = os.path.join(dataset_root_path, self.phase)
        self.images = []
        self.labels = []
        self.captions = [] 
        self.categories = {'amusement': 25, 'anger': 26, 'awe': 27, 'contentment': 3, 'disgust': 5, 'excitement': 28, 'fear': 29, 'sadness': 19}
        for cat, _ in sorted(self.categories.items()):
            regexp = os.path.join(self.path, cat, '*.pt')
            imgs = sorted(glob.glob(regexp))
            for img in imgs:
                self.images.append(img)
                self.labels.append(self.categories[cat])
                self.captions.append(cat)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return torch.load(self.images[index], weights_only=False), torch.tensor(self.labels[index]), self.captions[index]

