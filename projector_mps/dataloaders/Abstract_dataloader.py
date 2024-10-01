import os
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
import random
import sys
import json
import csv

class AbstractDataset():
    def __init__(self, path: str = '/home/cgutierrez/shared/Abstract/testImages_abstract/'):
        self.path = path
        self.images = []
        self.labels = []
        self.captions = []
        self.mapcat = {'Amusement': 'amusement', 'Anger': 'anger', 'Awe': 'awe', 'Content': 'contentment', 'Disgust': 'disgust', 'Excitement': 'excitement', 'Fear': 'fear', 'Sad': 'sadness'}
        self.maplab = {'amusement': 0, 'anger': 1, 'awe': 2, 'contentment': 3, 'disgust': 4, 'excitement': 5, 'fear': 6, 'sadness': 7}
        
        self.gt_path = os.path.join(self.path, 'ABSTRACT_groundTruth.csv')
        with open(self.gt_path, mode='r') as file:
            data = csv.reader(file)
            self.gt = [r for r in data]
        headers = self.gt[0]
        self.gt = self.gt[1:]
        for i, record in enumerate(self.gt):
            img = record[0]
            annots = record[1:]
            if annots.count(max(annots)) > 1:
                continue
            cat = self.mapcat[headers[annots.index(max(annots)) + 1].strip("'")]
            self.images.append(os.path.join(self.path, img.strip("'")))
            self.labels.append(self.maplab[cat])
            self.captions.append(cat)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], torch.tensor(self.labels[index]), self.captions[index]
