import os
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
import random
import sys
import json

class Emotion6Dataset():
    def __init__(self, path: str = '/home/cgutierrez/shared/Emotion6/Emotion6/'):
        self.name = "Emotion6"
        self.path = os.path.join(path, 'images')
        self.testing_split = []
        self.images = []
        self.labels = []
        self.captions = []
        self.categories = {k: cat for k, cat in enumerate(sorted(os.listdir(self.path))) if not cat.startswith('.')}     
        self.mapping = {'anger': 26, 'disgust': 5, 'fear': 29, 'joy': 30, 'sadness': 19, 'surprise': 22}
        ctr = 0
        for k, cat in sorted(self.categories.items()):
            regexp = os.path.join(self.path, cat, '*.jpg')
            imgs = sorted(glob.glob(regexp))
            for img in imgs:
                self.images.append(img)
                self.labels.append(self.mapping[cat])
                self.captions.append(cat)
            ctr += 1


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], torch.tensor(self.labels[index]), self.captions[index]
    
    def get_sentiments(self):
        res = {}
        for label, sentiment in zip(self.labels, self.captions):
            if sentiment not in res.keys():
                res[sentiment] = label
        return res