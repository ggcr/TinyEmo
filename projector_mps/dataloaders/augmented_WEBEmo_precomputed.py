import os
import torch
import glob
from typing import List, Tuple, Dict

class MergedEmotionDataset:
    def __init__(self, vision_encoder: str, phase: str = 'train'):
        print(f"Loading Merged Emotion Dataset with phase {phase}")
        if vision_encoder == 'openai/clip-vit-large-patch14':
            dataset_root_path = '/home/cgutierrez/shared/features_clip-vit-large-patch14/WEBEmo/'
        elif vision_encoder == 'google/siglip-so400m-patch14-384':
            dataset_root_path = '/home/cgutierrez/shared/features_siglip-so400m-patch14-384/WEBEmo/'
        else:
            raise ValueError(f"Unknown vision encoder: {vision_encoder}")

        self.name = "MergedEmotion"
        self.phase = phase
        if self.phase not in ['train', 'validation', 'test']:
            raise ValueError(f"Not a valid phase chosen: {self.phase}")

        self.path = os.path.join(dataset_root_path, self.phase)
        self.images: List[str] = []
        self.labels: List[int] = []
        self.taxonomies: List[str] = []
        self.emotions: List[str] = []

        self.parrot_categories: Dict[int, str] = {
            0: 'affection', 1: 'cheerfullness', 2: 'confusion', 3: 'contentment', 4: 'disappointment',
            5: 'disgust', 6: 'enthrallment', 7: 'envy', 8: 'exasperation', 9: 'gratitude', 10: 'horror',
            11: 'irritabilty', 12: 'lust', 13: 'neglect', 14: 'nervousness', 15: 'optimism', 16: 'pride',
            17: 'rage', 18: 'relief', 19: 'sadness', 20: 'shame', 21: 'suffering', 22: 'surprise',
            23: 'sympathy', 24: 'zest'
        }

        self.ekman_categories = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
        self.mikels_categories = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']

        self.parrot_to_ekman: Dict[str, str] = {
            'anger': 'anger', 'disgust': 'disgust', 'fear': 'fear', 'cheerfullness': 'joy',
            'sadness': 'sadness', 'surprise': 'surprise', 'affection': 'joy',
            'disappointment': 'sadness', 'exasperation': 'anger', 'pride': 'joy',
            'shame': 'sadness', 'zest': 'joy', 'contentment': 'joy',
            'enthrallment': 'surprise', 'envy': 'anger', 'gratitude': 'joy',
            'horror': 'fear', 'irritabilty': 'anger', 'neglect': 'sadness', 'nervousness': 'fear',
            'optimism': 'joy', 'rage': 'anger', 'relief': 'joy', 'suffering': 'sadness',
            'sympathy': 'sadness', 'confusion': None, 'lust': None
        }

        self.parrot_to_mikels: Dict[str, str] = {
            'anger': 'anger', 'disgust': 'disgust', 'fear': 'fear', 'cheerfullness': 'amusement',
            'sadness': 'sadness', 'contentment': 'contentment', 'surprise': 'awe',
            'affection': 'contentment', 'disappointment': 'sadness', 'exasperation': 'anger',
            'pride': 'contentment', 'shame': 'sadness', 'zest': 'excitement', 'enthrallment': 'awe',
            'envy': 'anger', 'gratitude': 'contentment', 'horror': 'fear', 'irritabilty': 'anger',
            'neglect': 'sadness', 'nervousness': 'fear', 'optimism': 'excitement', 'rage': 'anger',
            'relief': 'contentment', 'suffering': 'sadness', 'sympathy': 'sadness',
            'confusion': None, 'lust': None
        }

        self.emotion_to_label: Dict[str, int] = self._create_emotion_label_mapping()
        self._load_data()

    def _create_emotion_label_mapping(self) -> Dict[str, int]:
        all_emotions = set(self.parrot_categories.values()) | set(self.ekman_categories) | set(self.mikels_categories)
        return {emotion: i for i, emotion in enumerate(sorted(all_emotions))}

    def _load_data(self):
        for parrot_label, parrot_emotion in sorted(self.parrot_categories.items()):
            regexp = os.path.join(self.path, parrot_emotion, '*.pt')
            imgs = sorted(glob.glob(regexp))
            for img in imgs:
                # Add Parrot entry
                self.images.append(img)
                self.labels.append(self.emotion_to_label[parrot_emotion])
                self.taxonomies.append('parrot')
                self.emotions.append(parrot_emotion)

                # Add Ekman entry if mapping exists
                ekman_emotion = self.parrot_to_ekman.get(parrot_emotion)
                if ekman_emotion:
                    self.images.append(img)
                    self.labels.append(self.emotion_to_label[ekman_emotion])
                    self.taxonomies.append('ekman')
                    self.emotions.append(ekman_emotion)

                # Add Mikels entry if mapping exists
                mikels_emotion = self.parrot_to_mikels.get(parrot_emotion)
                if mikels_emotion:
                    self.images.append(img)
                    self.labels.append(self.emotion_to_label[mikels_emotion])
                    self.taxonomies.append('mikels')
                    self.emotions.append(mikels_emotion)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_path = self.images[index]
        label = self.labels[index]
        emotion = self.emotions[index]
        return torch.load(img_path, weights_only=False), torch.tensor(label), emotion

    def get_num_classes(self) -> int:
        return len(self.emotion_to_label)

    def get_emotion_label_mapping(self) -> Dict[str, int]:
        return self.emotion_to_label