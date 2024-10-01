import os
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
import random
import sys
import json

class FI_Dataset():
    def __init__(self, path: str = '/home/cgutierrez/shared/Aff-COCO-compilate/emotion_dataset/'):
        self.path = os.path.join(path)
        self.testing_split = []
        self.images = []
        self.labels = []
        self.captions = []
        self.categories = {k: cat for k, cat in enumerate(sorted(os.listdir(self.path))) if not cat.startswith('.')}     
        ctr = 0
        for k, cat in sorted(self.categories.items()):
            regexp = os.path.join(self.path, cat, '*.jpg')
            imgs = sorted(glob.glob(regexp))
            for img in imgs:
                self.images.append(img)
                self.labels.append(ctr)
                self.captions.append(cat)
            ctr += 1


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], torch.tensor(self.labels[index]), self.captions[index]
