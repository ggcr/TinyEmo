import os
import torch
import glob

class WEBEmoDataset():
    def __init__(self, dataset_root_path='/root/TFM_CristianGutierrez/data/partitioned_WEBEmo_fine/', phase='train'):
        print(f"Loading WEBEmo with phase {phase}")
        self.name = "WEBEmo"
        self.phase = phase
        if self.phase not in ['train', 'validation', 'test']:
            raise Exception(f"dataloaders/WEBEmo.py: Not a valid phase chosen: {self.phase}")
        self.path = os.path.join(dataset_root_path, self.phase)
        self.images = []
        self.labels = []
        self.sentiments = []
        self.categories = {0: 'affection', 1: 'cheerfullness', 2: 'confusion', 3: 'contentment', 4: 'disappointment', 5: 'disgust', 6: 'enthrallment', 7: 'envy', 8: 'exasperation', 9: 'gratitude', 10: 'horror', 11: 'irritabilty', 12: 'lust', 13: 'neglect', 14: 'nervousness', 15: 'optimism', 16: 'pride', 17: 'rage', 18: 'relief', 19: 'sadness', 20: 'shame', 21: 'suffering', 22: 'surprise', 23: 'sympathy', 24: 'zest'}
        ctr = 0
        for _, cat in sorted(self.categories.items()):
            regexp = os.path.join(self.path, cat, '*.jpg')
            imgs = sorted(glob.glob(regexp))
            for img in imgs:
                self.images.append(img)
                self.labels.append(ctr)
                self.sentiments.append(cat)
            ctr += 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]
        sentiment = self.sentiments[index]
        return img_path, torch.tensor(label), sentiment