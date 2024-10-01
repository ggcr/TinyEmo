import os
import time
import argparse
import sys
import tqdm
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.dataset import random_split

from projector_mps.src.model.model import Model
from projector_mps.dataloaders.WEBEmo import WEBEmoDataset
from projector_mps.dataloaders.Emotion6 import Emotion6Dataset
from projector_mps.dataloaders.EmoROI import EmotionROIDataset
from projector_mps.dataloaders.EmoSet import EmosetDataset
from projector_mps.dataloaders.FI_dataloader import FI_Dataset
from projector_mps.dataloaders.UnbiasedEmo_dataloader import UnbiasedEmo_Dataset
from projector_mps.src.model.utils import compute_top_accuracy


if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

def precompute_features(model, dataloader, path, save_also_dirname=True):
    os.makedirs(path, exist_ok=True)
    for batch_idx, (images, labels, sentiments) in enumerate(tqdm.tqdm(dataloader)):
        features = model.vision_encoder.encode_images(images).to(device)
        for i, img in enumerate(images): 
            if save_also_dirname:
                if 'UnbiasedEmo' in path:
                    os.makedirs(os.path.join(path, os.path.dirname(img).split('/')[-2]), exist_ok=True)
                    torch.save(features[i].detach().cpu(), os.path.join(path, os.path.dirname(img).split('/')[-2], os.path.dirname(img).split('/')[-1] + '_' + os.path.basename(img).split('.')[0] + '.pt').replace(' ', '_'))
                else:
                    os.makedirs(os.path.join(path, os.path.dirname(img).split('/')[-1]), exist_ok=True)
                    torch.save(features[i].detach().cpu(), os.path.join(path, os.path.dirname(img).split('/')[-1], os.path.basename(img).split('.')[0] + '.pt'))
            else:
                torch.save(features[i].detach().cpu(), os.path.join(path, os.path.basename(img).split('.')[0] + '.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vision_encoder_path', type=str, help='Path to the vision encoder model.', required=True)
    parser.add_argument('--model_path', type=str, help='Hugginface LLM path.', required=True)
    parser.add_argument('--tokenizer_path', type=str, help='Path to the tokenizer model.', default=None, required=False)
    parser.add_argument('--dataset', type=str, help='Dataset.', default=None, required=True)
    parser.add_argument('--per_device_train_batch_size', type=int, help='Train Batch size', default=32, required=True)
    parser.add_argument('--per_device_eval_batch_size', type=int, help='Train Eval size', default=32, required=True)
    parser.add_argument('--num_train_epochs', type=int, help='Train Eval size', default=5, required=True)
    parser.add_argument('--run_name', type=str, help='Where to report', default='wandb', required=True)
    parser.add_argument('--output_dir', type=str, help='Where to save', required=True)
    args = parser.parse_args()

    dataset = args.dataset
    if dataset == 'WEBEmo':
        train_dataset = WEBEmoDataset(phase='train')
        validation_dataset = WEBEmoDataset(phase='validation')
        test_dataset = WEBEmoDataset(phase='test')
        train_path = f'/home/cgutierrez/shared/features_{args.vision_encoder_path.split("/")[-1]}/{args.dataset}/train/'
        validation_path = f'/home/cgutierrez/shared/features_{args.vision_encoder_path.split("/")[-1]}/{args.dataset}/validation/'
        test_path = f'/home/cgutierrez/shared/features_{args.vision_encoder_path.split("/")[-1]}/{args.dataset}/test/'
    elif dataset == 'Emotion6':
        dataset = Emotion6Dataset()
        test_size = int(0.3 * len(dataset)) # same images as with EmotionROI
        train_size = len(dataset) - test_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        validation_dataset = test_dataset
        train_path = f'/home/cgutierrez/shared/features_{args.vision_encoder_path.split("/")[-1]}/{args.dataset}/images/train/'
        validation_path = f'/home/cgutierrez/shared/features_{args.vision_encoder_path.split("/")[-1]}/{args.dataset}/images/validation/'
        test_path = f'/home/cgutierrez/shared/features_{args.vision_encoder_path.split("/")[-1]}/{args.dataset}/images/test/'
    elif dataset == 'EmoROI':
        train_dataset = EmotionROIDataset(phase='train')
        validation_dataset = EmotionROIDataset(phase='validation')
        test_dataset = EmotionROIDataset(phase='test')
        train_path = f'/home/cgutierrez/shared/features_{args.vision_encoder_path.split("/")[-1]}/{args.dataset}/train/'
        validation_path = f'/home/cgutierrez/shared/features_{args.vision_encoder_path.split("/")[-1]}/{args.dataset}/validation/'
        test_path = f'/home/cgutierrez/shared/features_{args.vision_encoder_path.split("/")[-1]}/{args.dataset}/test/'
    elif dataset == "EmoSet":
        train_dataset = EmosetDataset(phase='train')
        validation_dataset = EmosetDataset(phase='val')
        test_dataset = EmosetDataset(phase='test')
        train_path = f'/home/cgutierrez/shared/features_{args.vision_encoder_path.split("/")[-1]}/{args.dataset}/train/'
        validation_path = f'/home/cgutierrez/shared/features_{args.vision_encoder_path.split("/")[-1]}/{args.dataset}/validation/'
        test_path = f'/home/cgutierrez/shared/features_{args.vision_encoder_path.split("/")[-1]}/{args.dataset}/test/'
    elif dataset == "FI":
        dataset = FI_Dataset()
        test_size = int(0.2 * len(dataset)) # same images as with EmotionROI
        train_size = len(dataset) - test_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        validation_dataset = test_dataset
        train_path = f'/home/cgutierrez/shared/features_{args.vision_encoder_path.split("/")[-1]}/{args.dataset}/train/'
        validation_path = f'/home/cgutierrez/shared/features_{args.vision_encoder_path.split("/")[-1]}/{args.dataset}/validation/'
        test_path = f'/home/cgutierrez/shared/features_{args.vision_encoder_path.split("/")[-1]}/{args.dataset}/test/'
    elif dataset == "UnbiasedEmo":
        dataset = UnbiasedEmo_Dataset()
        test_dataset = dataset
        test_path = f'/home/cgutierrez/shared/features_{args.vision_encoder_path.split("/")[-1]}/{args.dataset}/test/'
    elif dataset == "Joint":
        train_dataset_25 = WEBEmoDataset(phase='train')
        train_dataset_8 = EmosetDataset(phase='train')
        train_dataset_6 = Emotion6Dataset()
        test_size = int(0.3 * len(train_dataset_6)) # same images as with EmotionROI
        train_size = len(train_dataset_6) - test_size
        train_dataset_6, _ = random_split(train_dataset_6, [train_size, test_size])
        train_dataset = ConcatDataset([train_dataset_25, train_dataset_8, train_dataset_6])
        validation_dataset = EmosetDataset(phase='test')
        test_dataset = EmosetDataset(phase='test')
    else:
        raise NotImplementedError(f'Dataset: {dataset}')
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True, drop_last=False, num_workers=4)
    val_dataloader = DataLoader(validation_dataset, batch_size=args.per_device_eval_batch_size, shuffle=True, drop_last=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.per_device_eval_batch_size, shuffle=True, drop_last=False, num_workers=4)

    model = Model(vision_encoder_path=args.vision_encoder_path, model_path=args.model_path, tokenizer_path=args.tokenizer_path, num_epochs=args.num_train_epochs, len_dataloader=len(test_dataloader), dataset=dataset, run_name=args.run_name, output_dir=args.output_dir)

    print(model.projector)
    precompute_features(model, train_dataloader, train_path)
    precompute_features(model, val_dataloader, validation_path)
    precompute_features(model, test_dataloader, test_path)