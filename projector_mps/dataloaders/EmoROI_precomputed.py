import os
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
import random
import sys
import json

class EmoROIDataset_precompute():
    def __init__(self, vision_encoder, phase='train'):
        self.name = "EmoROI"
        if vision_encoder == 'openai/clip-vit-large-patch14':
            dataset_root_path = '/home/cgutierrez/shared/features_clip-vit-large-patch14/EmoROI/'
        elif vision_encoder == 'openai/clip-vit-base-patch16':
            dataset_root_path = '/home/cgutierrez/shared/features_clip-vit-base-patch16/EmoROI/'
        elif vision_encoder == 'openai/clip-vit-large-patch14-336':
            dataset_root_path = '/home/cgutierrez/shared/features_clip-vit-large-patch14-336/EmoROI/'
        elif vision_encoder == 'facebook/dinov2-large':
            dataset_root_path = '/home/cgutierrez/shared/features_dinov2-large/EmoROI/'
        elif vision_encoder == 'google/siglip-so400m-patch14-384':
            dataset_root_path = '/home/cgutierrez/shared/features_siglip-so400m-patch14-384/EmoROI/'
        self.path = os.path.join(dataset_root_path, phase)
        self.testing_split = []
        self.images = []
        self.labels = []
        self.captions = []
        self.categories = {k: cat for k, cat in enumerate(sorted(os.listdir(self.path))) if not cat.startswith('.')}     
        self.mapping = {'anger': 26, 'disgust': 5, 'fear': 29, 'joy': 3, 'sadness': 19, 'surprise': 22}
        ctr = 0
        for k, cat in sorted(self.categories.items()):
            regexp = os.path.join(self.path, cat, '*.pt')
            imgs = sorted(glob.glob(regexp))
            for img in imgs:
                self.images.append(img)
                self.labels.append(self.mapping[cat])
                self.captions.append(cat)
            ctr += 1


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return torch.load(self.images[index], weights_only=False), torch.tensor(self.labels[index]), self.captions[index]