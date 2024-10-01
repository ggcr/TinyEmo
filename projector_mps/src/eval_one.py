import argparse
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from src.model.model import Model
from dataloaders.WEBEmo_precompute import WEBEmoDataset_precompute
from dataloaders.Emotion6_precompute import Emotion6Dataset_precompute
from dataloaders.EmoROI import EmotionROIDataset
from dataloaders.EmoROI_precomputed import EmoROIDataset_precompute
from dataloaders.EmoSet_precompute import EmosetDataset_precompute
from dataloaders.FI_dataloader import FI_Dataset
from dataloaders.Abstract_dataloader import AbstractDataset
from dataloaders.ArtPhoto_dataloader import ArtPhotoDataset
from dataloaders.UnbiasedEmo_dataloader import UnbiasedEmo_Dataset
from dataloaders.UnbiasedEmo_dataloader_precomputed import UnbiasedEmoDataset_precompute
from dataloaders.FI_dataloader_precomputed import FIDataset_precompute
from src.model.utils import compute_top_accuracy

from pytorch_metric_learning import losses


def test(model, dataloader, projector_weights_path=None):
    if projector_weights_path is not None:
        # Load projector weights
        print(f"Loading projector weights from {projector_weights_path}")
        model.projector.load_state_dict(torch.load(projector_weights_path))
    test_loss, embeds = model.eval(dataloader, return_image_embeds=True)
    print(f'Test Loss: {test_loss}')
    top_1 = compute_top_accuracy(model, embeds)
    return top_1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vision_encoder_path', type=str, help='Path to the vision encoder model.', required=True)
    parser.add_argument('--model_path', type=str, help='Huggingface LLM path.', required=True)
    parser.add_argument('--tokenizer_path', type=str, help='Path to the tokenizer model.', default=None, required=False)
    parser.add_argument('--projector_weights_path', type=str, help='Path to the projector weights.', required=True)
    parser.add_argument('--loss_name', type=str, help='Loss to use during training', default='infoNCE', required=True)
    args = parser.parse_args()

    root_path = args.projector_weights_path
    
    model = Model(
        vision_encoder_path=args.vision_encoder_path, 
        model_path=args.model_path, 
        tokenizer_path=args.tokenizer_path,
        precomputed=True
    )

    model.loss_name = args.loss_name
    if args.loss_name == 'CosineEmbedding':
        print(f"Using loss {args.loss_name}")
        model.loss_func = torch.nn.CosineEmbeddingLoss(margin=0.5)
    elif args.loss_name == 'NTXent':
        print(f"Using loss {args.loss_name}")
        model.loss_func = losses.NTXentLoss()

    WEBEMO_test_dataset = WEBEmoDataset_precompute(args.vision_encoder_path, phase='test')
    Emotion6_test_dataset = Emotion6Dataset_precompute(args.vision_encoder_path, phase='test')
    EmoROI_test_dataset = EmoROIDataset_precompute(args.vision_encoder_path, phase='test')
    EmoSet_test_dataset = EmosetDataset_precompute(args.vision_encoder_path, phase='test')
    FI_test_dataset = FIDataset_precompute(args.vision_encoder_path, phase='test')
    Unbiased_test_dataset = UnbiasedEmoDataset_precompute(args.vision_encoder_path, phase='test')
    Abstract_test_dataset = AbstractDataset()
    ArtPhoto_test_dataset = ArtPhotoDataset()

    test_datasets_precomputed = [EmoROI_test_dataset, EmoSet_test_dataset, WEBEMO_test_dataset, Emotion6_test_dataset, FI_test_dataset, Unbiased_test_dataset]
    test_datasets_not_precomputed = [ArtPhoto_test_dataset, Abstract_test_dataset]
    test_datasets = [EmoROI_test_dataset, EmoSet_test_dataset, WEBEMO_test_dataset, Emotion6_test_dataset, FI_test_dataset, ArtPhoto_test_dataset, Abstract_test_dataset, Unbiased_test_dataset]

    # Compute the avg performance of each model individually
    ckpt = root_path
    print(ckpt)
    res = []
    for test_dataset in test_datasets:
        print(test_dataset)
        if test_dataset in test_datasets_precomputed:
            model.precomputed = True
        else:
            model.precomputed = False
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=False, num_workers=4)
        top_1 = test(model, test_dataloader, projector_weights_path=ckpt)
        res.append(top_1 * 100)

    print(f"{', '.join([str(round(i, 2)) for i in res])}, {round(sum(res) / len(res), 2)}")

