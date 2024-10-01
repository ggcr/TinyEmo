import os
import time
import argparse
import sys

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.dataset import random_split

from src.model.model import Model
from dataloaders.WEBEmo import WEBEmoDataset
from dataloaders.Emotion6 import Emotion6Dataset
from dataloaders.EmoROI import EmotionROIDataset
from dataloaders.EmoSet import EmosetDataset

from dataloaders.WEBEmo_precompute import WEBEmoDataset_precompute
from dataloaders.augmented_WEBEmo_precomputed import MergedEmotionDataset
from dataloaders.Emotion6_precompute import Emotion6Dataset_precompute
from dataloaders.EmoROI_precomputed import EmoROIDataset_precompute
from dataloaders.EmoSet_precompute import EmosetDataset_precompute

from src.model.utils import compute_top_accuracy

def train(model, train_dataloader, validation_dataloader, test_datalaoder, num_epochs, projector_weights_path):
    best_acc = 0
    for epoch in range(1, num_epochs+1):
        train_loss = model.train(train_dataloader, epoch)
        eval_loss, embeds = model.eval(validation_dataloader, return_image_embeds=True)
        top1 = compute_top_accuracy(model, embeds)
        if top1 >= best_acc:
            print(f"Saving model {projector_weights_path}")
            os.makedirs(projector_weights_path, exist_ok=True)
            connector_output_path = os.path.join(projector_weights_path, f'pytorch_model.bin')
            torch.save(model.projector.state_dict(), connector_output_path)
        print(f"Train Loss for Epoch {epoch}: {train_loss}")
        print(f"Validation Loss for Epoch {epoch}: {eval_loss}")
    test_loss, embeds = model.eval(test_datalaoder, return_image_embeds=True)
    print(f'Test Loss: {test_loss}')
    top1 = compute_top_accuracy(model, embeds)

def test(model, dataloader, projector_weights_path=None):
    if projector_weights_path is not None:
        # Load projector weights
        print(f"Loading projector weights from {projector_weights_path}")
        model.projector.load(projector_weights_path)
    test_loss, embeds = model.eval(dataloader, return_image_embeds=True)
    print(f'Test Loss: {test_loss}')
    compute_top_accuracy(model, embeds)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vision_encoder_path', type=str, help='Path to the vision encoder model.', required=True)
    parser.add_argument('--model_path', type=str, help='Hugginface LLM path.', required=True)
    parser.add_argument('--tokenizer_path', type=str, help='Path to the tokenizer model.', default=None, required=False)
    parser.add_argument('--dataset', type=str, help='Dataset.', default=None, required=True)
    parser.add_argument('--per_device_train_batch_size', type=int, help='Train Batch size', default=32, required=True)
    parser.add_argument('--per_device_eval_batch_size', type=int, help='Train Eval size', default=32, required=True)
    parser.add_argument('--num_train_epochs', type=int, help='Train Eval size', default=5, required=True)
    parser.add_argument('--report_to', type=str, help='Where to report', default='wandb', required=False)
    parser.add_argument('--run_name', type=str, help='Where to report', default='wandb', required=True)
    parser.add_argument('--output_dir', type=str, help='Where to save', required=True)
    parser.add_argument('--precomputed', type=str, help='Use precomputed dataloaders.', default=False)
    args = parser.parse_args()

    dataset = args.dataset
    if dataset == 'WEBEmo':
        if args.precomputed == True:
            train_dataset = WEBEmoDataset_precompute(args.vision_encoder_path, phase='train')
            validation_dataset = WEBEmoDataset_precompute(args.vision_encoder_path, phase='validation')
            test_dataset = WEBEmoDataset_precompute(args.vision_encoder_path, phase='test')
        else:
            train_dataset = WEBEmoDataset(phase='train')
            validation_dataset = WEBEmoDataset(phase='validation')
            test_dataset = WEBEmoDataset(phase='test')
    elif dataset == 'Emotion6':
        if args.precomputed == True:
            train_dataset = Emotion6Dataset_precompute(args.vision_encoder_path, phase='train')
            validation_dataset = Emotion6Dataset_precompute(args.vision_encoder_path, phase='validation')
            test_dataset = Emotion6Dataset_precompute(args.vision_encoder_path, phase='test')
        else:
            dataset = Emotion6Dataset()
            test_size = int(0.3 * len(dataset)) # same images as with EmotionROI
            train_size = len(dataset) - test_size
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    elif dataset == 'EmoROI':
        if args.precomputed == True:
            train_dataset = EmoROIDataset_precompute(args.vision_encoder_path, phase='train')
            validation_dataset = EmoROIDataset_precompute(args.vision_encoder_path, phase='validation')
            test_dataset = EmoROIDataset_precompute(args.vision_encoder_path, phase='test')
        else:
            train_dataset = EmotionROIDataset(phase='train')
            validation_dataset = EmotionROIDataset(phase='validation')
            test_dataset = EmotionROIDataset(phase='test')
    elif dataset == "EmoSet":
        if args.precomputed == True:
            train_dataset = EmosetDataset_precompute(args.vision_encoder_path, phase='train')
            validation_dataset = EmosetDataset_precompute(args.vision_encoder_path, phase='validation')
            test_dataset = EmosetDataset_precompute(args.vision_encoder_path, phase='test')
        else:
            train_dataset = EmosetDataset(phase='train')
            validation_dataset = EmosetDataset(phase='validation')
            test_dataset = EmosetDataset(phase='test')
    elif dataset == "Joint":
        train_dataset = MergedEmotionDataset(args.vision_encoder_path, phase='train')
        validation_dataset = EmosetDataset_precompute(args.vision_encoder_path, phase='test')
        test_dataset = EmosetDataset_precompute(args.vision_encoder_path, phase='test')
    else:
        raise NotImplementedError(f'Dataset: {dataset}')
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_dataloader = DataLoader(validation_dataset, batch_size=args.per_device_eval_batch_size, shuffle=True, drop_last=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.per_device_eval_batch_size, shuffle=True, drop_last=True, num_workers=2)

    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        free_mem_gb = free_mem / (1024 ** 3)
        total_mem_gb = total_mem / (1024 ** 3)
        print(f"Free: {free_mem_gb:.2f} GB / Total: {total_mem_gb:.2f} GB")
    
    model = Model(vision_encoder_path=args.vision_encoder_path, model_path=args.model_path, tokenizer_path=args.tokenizer_path, num_epochs=args.num_train_epochs, len_dataloader=len(train_dataloader), report_to=args.report_to, dataset=dataset, run_name=args.run_name, output_dir=args.output_dir, precomputed=args.precomputed)

    print(model.projector)
    train(model, train_dataloader, test_dataloader, test_dataloader, args.num_train_epochs, args.output_dir)