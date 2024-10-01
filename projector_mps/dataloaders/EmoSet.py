import os
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
import random
import sys
import json

class EmosetDataset():
    def __init__(self, path: str = '/home/cgutierrez/shared/EmoSet/', phase: str = 'train'):
        # phase must be 'train', 'val' or 'test'
        print(f'EmoSet phase: {phase}')
        self.json_file = None
        self.images = []
        self.labels = []
        self.captions = []
        with open(os.path.join(path, phase + '.json'), 'rb') as f:
            self.json_file = json.load(f)   
        mapping_cats = {'amusement': 25, 'anger': 26, 'awe': 27, 'contentment': 3, 'disgust': 5, 'excitement': 28, 'fear': 29, 'sadness': 19}
        for record in self.json_file:
            if os.path.exists(os.path.join(path, record[1]).replace('/image/', '/image336/')):
                self.images.append(os.path.join(path, record[1]).replace('/image/', '/image336/'))
            else:
                self.images.append(os.path.join(path, record[1]))
            self.labels.append(mapping_cats[record[0]])
            self.captions.append(record[0])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], torch.tensor(self.labels[index]), self.captions[index]

