import os
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
import random
import sys
import json

class ArtPhotoDataset():
    def __init__(self, path: str = '/home/cgutierrez/shared/ArtPhoto/testImages_artphoto/'):
        self.path = path
        self.testing_split = []
        self.images = []
        self.labels = []
        self.captions = []
        self.mapcat = {'amusement': 'amusement', 'anger': 'anger', 'awe': 'awe', 'contentment': 'contentment', 'disgust': 'disgust', 'excitement': 'excitement', 'fear': 'fear', 'sad': 'sadness'}
        self.maplab = {'amusement': 0, 'anger': 1, 'awe': 2, 'contentment': 3, 'disgust': 4, 'excitement': 5, 'fear': 6, 'sad': 7}
        regexp = os.path.join(self.path, '*.jpg')
        imgs = sorted(glob.glob(regexp))
        for img in imgs:
            cat_name = os.path.basename(img).split('_')[0]
            self.images.append(img)
            self.labels.append(self.maplab[cat_name])
            self.captions.append(self.mapcat[cat_name])


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], torch.tensor(self.labels[index]), self.captions[index]
