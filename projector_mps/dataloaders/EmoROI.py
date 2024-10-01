import os
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
import random
import sys
import json

class EmotionROIDataset():
    def __init__(self, path: str = '/home/cgutierrez/shared/EmotionROI/EmotionROI/', phase='test'):
        self.name = "EmotionROIDataset"
        self.path = os.path.join(path, 'images')
        self.testing_split = []
        self.images = []
        self.labels = []
        self.captions = []
        self.categories = {'anger': 26, 'disgust': 5, 'fear': 29, 'joy': 30, 'sadness': 19, 'surprise': 22}
        txt_file = os.path.join(path, 'training_testing_split', 'testing.txt') if phase == 'test' else os.path.join(path, 'training_testing_split', 'training.txt')
        with open(txt_file, 'r') as f:
            for line in f:
                l = line.strip()
                cat, filename = l.split('/')[:]
                self.images.append(os.path.join(self.path, cat, filename))
                self.labels.append(self.categories[cat])
                self.captions.append(cat)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], torch.tensor(self.labels[index]), self.captions[index]
