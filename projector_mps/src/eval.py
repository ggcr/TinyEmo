import argparse
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from projector_mps.src.model.model import Model
from projector_mps.dataloaders.WEBEmo_precompute import WEBEmoDataset_precompute
from projector_mps.dataloaders.Emotion6_precompute import Emotion6Dataset_precompute
from projector_mps.dataloaders.EmoROI import EmotionROIDataset
from projector_mps.dataloaders.EmoROI_precomputed import EmoROIDataset_precompute
from projector_mps.dataloaders.EmoSet_precompute import EmosetDataset_precompute
from projector_mps.dataloaders.FI_dataloader import FI_Dataset
from projector_mps.dataloaders.Abstract_dataloader import AbstractDataset
from projector_mps.dataloaders.ArtPhoto_dataloader import ArtPhotoDataset
from projector_mps.dataloaders.UnbiasedEmo_dataloader import UnbiasedEmo_Dataset
from projector_mps.dataloaders.FI_dataloader_precomputed import FIDataset_precompute
from projector_mps.src.model.utils import compute_top_accuracy

from pytorch_metric_learning import losses


def test(model, dataloader, projector_weights_path=None):
    if projector_weights_path is not None:
        # Load projector weights
        print(f"Loading projector weights from {projector_weights_path}")
        model.projector.load_state_dict(torch.load(projector_weights_path, map_location=model.device, weights_only=True))
    test_loss, embeds = model.eval(dataloader, return_image_embeds=True)
    print(f'Test Loss: {test_loss}')
    compute_top_accuracy(model, embeds)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vision_encoder_path', type=str, help='Path to the vision encoder model.', required=True)
    parser.add_argument('--model_path', type=str, help='Huggingface LLM path.', required=True)
    parser.add_argument('--tokenizer_path', type=str, help='Path to the tokenizer model.', default=None, required=False)
    parser.add_argument('--dataset', type=str, help='Dataset.', required=True)
    parser.add_argument('--projector_weights_path', type=str, help='Path to the projector weights.', required=True)
    parser.add_argument('--per_device_eval_batch_size', type=int, help='Eval batch size', default=16, required=True)
    parser.add_argument('--num_train_epochs', type=int, help='Train Eval size', default=5, required=False)
    parser.add_argument('--loss_name', type=str, help='Loss to use during training', default='infoNCE', required=True)
    parser.add_argument('--run_name', type=str, help='Where to report', default='', required=True)
    parser.add_argument('--precomputed', type=str, help='Use precomputed dataloaders.', default=False)
    args = parser.parse_args()

    dataset = args.dataset
    if dataset == 'WEBEmo':
        train_dataset = WEBEmoDataset_precompute(args.vision_encoder_path, phase='train')
        validation_dataset = WEBEmoDataset_precompute(args.vision_encoder_path, phase='validation')
        test_dataset = WEBEmoDataset_precompute(args.vision_encoder_path, phase='test')
    elif dataset == 'Emotion6':
        train_dataset = Emotion6Dataset_precompute(args.vision_encoder_path, phase='train')
        validation_dataset = Emotion6Dataset_precompute(args.vision_encoder_path, phase='validation')
        test_dataset = Emotion6Dataset_precompute(args.vision_encoder_path, phase='test')
    elif dataset == 'EmoROI':
        train_dataset = EmoROIDataset_precompute(args.vision_encoder_path, phase='train')
        validation_dataset = EmoROIDataset_precompute(args.vision_encoder_path, phase='validation')
        test_dataset = EmoROIDataset_precompute(args.vision_encoder_path, phase='test')
    elif dataset == "EmoSet":
        train_dataset = EmosetDataset_precompute(args.vision_encoder_path, phase='train')
        validation_dataset = EmosetDataset_precompute(args.vision_encoder_path, phase='validation')
        test_dataset = EmosetDataset_precompute(args.vision_encoder_path, phase='test')
    elif dataset == "FI":
        train_dataset = FIDataset_precompute(args.vision_encoder_path, phase='train')
        validation_dataset = FIDataset_precompute(args.vision_encoder_path, phase='validation')
        test_dataset = FIDataset_precompute(args.vision_encoder_path, phase='test')
    elif dataset == "UnbiasedEmo":
        test_dataset = UnbiasedEmo_Dataset()
    elif dataset == 'Abstract':
        test_dataset = AbstractDataset()
    elif dataset == 'ArtPhoto':
        test_dataset = ArtPhotoDataset()
    else:
        raise NotImplementedError(f'Dataset: {dataset}')
    
    test_dataloader = DataLoader(test_dataset, batch_size=args.per_device_eval_batch_size, shuffle=True, drop_last=True, num_workers=4)

    model = Model(vision_encoder_path=args.vision_encoder_path, model_path=args.model_path, tokenizer_path=args.tokenizer_path, num_epochs=args.num_train_epochs, dataset=dataset, run_name=args.run_name, precomputed=args.precomputed)
    model.loss_name = args.loss_name
    if args.loss_name == 'CosineEmbedding':
        print(f"Using loss {args.loss_name}")
        model.loss_func = torch.nn.CosineEmbeddingLoss(margin=0.5)
    elif args.loss_name == 'NTXent':
        print(f"Using loss {args.loss_name}")
        model.loss_func = losses.NTXentLoss()

    print(model.projector)
    test(model, test_dataloader, projector_weights_path=args.projector_weights_path)
