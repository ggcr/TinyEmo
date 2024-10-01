import os
import torch
import glob
import random

synonyms = { # taken and modified from CLIP-E paper by Cristina Bustos
    'affection': ['Love', 'Fondness', 'Tenderness'],
    'cheerfullness': ['Happiness', 'Joy', 'Gladness'],
    'confusion': ['Perplexity', 'Bewilderment', 'Puzzlement'],
    'contentment': ['Satisfaction', 'Fulfillment', 'Serenity'],
    'disappointment': ['Letdown', 'Frustration', 'Regret'],
    'disgust': ['Revulsion', 'Loathing', 'Aversion'],
    'enthrallment': ['Fascination', 'Enchantment', 'Allurement'],
    'envy': ['Jealousy', 'Resentment', 'Covetousness'],
    'exasperation': ['Irritation', 'Aggravation', 'Annoyance'],
    'gratitude': ['Thankfulness', 'Appreciation', 'Recognition'],
    'horror': ['Terror', 'Dread', 'Fear'],
    'irritabilty': ['Grumpiness', 'Testiness', 'Edginess'],
    'lust': ['Desire', 'Passion', 'Love'],
    'neglect': ['Abandonment', 'Disregard', 'Negligence'],
    'nervousness': ['Anxiety', 'Tension', 'Apprehension'],
    'optimism': ['Hopefulness', 'Confidence', 'Positivity'],
    'pride': ['Self-esteem', 'Dignity', 'Vanity'],
    'rage': ['Fury', 'Wrath', 'Outrage'],
    'relief': ['Comfort', 'Ease', 'Reassurance'],
    'sadness': ['Sorrow', 'Grief', 'Despair'],
    'shame': ['Embarrassment', 'Humiliation', 'Guilt'],
    'suffering': ['Pain', 'Distress', 'Agony'],
    'surprise': ['Astonishment', 'Amazement', 'Startlement'],
    'sympathy': ['Compassion', 'Empathy', 'Understanding'],
    'zest': ['Enthusiasm', 'Energy', 'Vigor']
}

class WEBEmoDataset_precompute():
    def __init__(self, vision_encoder, phase='train'):
        self.synonims = False
        print(f"Loading WEBEmo with phase {phase}")
        if vision_encoder == 'openai/clip-vit-large-patch14':
            dataset_root_path = '/home/cgutierrez/shared/features_clip-vit-large-patch14/WEBEmo/'
        elif vision_encoder == 'openai/clip-vit-large-patch14-336':
            dataset_root_path = '/home/cgutierrez/shared/features_clip-vit-large-patch14-336/WEBEmo/'
        elif vision_encoder == 'facebook/dinov2-large':
            dataset_root_path = '/home/cgutierrez/shared/features_dinov2-large/WEBEmo/'
        elif vision_encoder == 'openai/clip-vit-base-patch16':
            dataset_root_path = '/home/cgutierrez/shared/features_clip-vit-base-patch16/WEBEmo/'
        elif vision_encoder == 'google/siglip-so400m-patch14-384':
            dataset_root_path = '/home/cgutierrez/shared/features_siglip-so400m-patch14-384/WEBEmo/'
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
            regexp = os.path.join(self.path, cat, '*.pt')
            imgs = sorted(glob.glob(regexp))
            for img in imgs:
                self.images.append(img)
                self.labels.append(ctr)
                if phase != 'test' and self.synonims and random.random() < 0.5 and cat in synonyms:
                    sentiment = random.choice(synonyms[cat]).lower()
                    print(sentiment)
                else:
                    sentiment = cat
                self.sentiments.append(sentiment)
            ctr += 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]
        sentiment = self.sentiments[index]
        return torch.load(self.images[index], weights_only=False), torch.tensor(label), sentiment