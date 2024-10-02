import argparse
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from huggingface_hub import hf_hub_download
from pytorch_metric_learning import losses

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


def get_model_config(projector_path):
    configs = {
        "ggcristian/TinyEmo-CLIP-OpenELM-270M-Syn": {
            "vision_encoder_path": "openai/clip-vit-large-patch14",
            "model_path": "apple/OpenELM-270M-Instruct",
            "loss_name": "infoNCE",
            "weights_file": "TinyEmo-CLIP-OpenELM-270M-Syn.bin"
        },
        "ggcristian/TinyEmo-CLIP-OpenELM-450M": {
            "vision_encoder_path": "openai/clip-vit-large-patch14",
            "model_path": "apple/OpenELM-450M-Instruct",
            "loss_name": "CosineEmbedding",
            "weights_file": "TinyEmo-CLIP-OpenELM-450M.bin"
        },
        "ggcristian/TinyEmo-CLIP-TinyLlama-1_1-Syn": {
            "vision_encoder_path": "openai/clip-vit-large-patch14",
            "model_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "loss_name": "CosineEmbedding",
            "weights_file": "TinyEmo-CLIP-TinyLlama-1_1-Syn.bin"
        },
        "ggcristian/TinyEmo-CLIP-Phi-2": {
            "vision_encoder_path": "openai/clip-vit-large-patch14",
            "model_path": "microsoft/Phi-2",
            "loss_name": "CosineEmbedding",
            "weights_file": "TinyEmo-CLIP-Phi-2.bin"
        }
    }
    return configs.get(projector_path, {})

def download_weights(repo_id, filename):
    return hf_hub_download(repo_id=repo_id, filename=filename)

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
    parser.add_argument('--vision_encoder_path', type=str, help='Path to the vision encoder model.', required=False)
    parser.add_argument('--model_path', type=str, help='Huggingface LLM path.', required=False)
    parser.add_argument('--tokenizer_path', type=str, help='Path to the tokenizer model.', default=None, required=False)
    parser.add_argument('--dataset', type=str, help='Dataset.', required=True)
    parser.add_argument('--projector_path', type=str, help='Path to the projector model on HuggingFace.', required=True)
    parser.add_argument('--per_device_eval_batch_size', type=int, help='Eval batch size', default=16, required=True)
    parser.add_argument('--num_train_epochs', type=int, help='Train Eval size', default=5, required=False)
    parser.add_argument('--loss_name', type=str, help='Loss to use during training', default='infoNCE', required=False)
    parser.add_argument('--run_name', type=str, help='Where to report', default='', required=True)
    parser.add_argument('--precomputed', type=str, help='Use precomputed dataloaders.', default=False)
    args = parser.parse_args()

    config = get_model_config(args.projector_path)
    if not config:
        print(f"{args.projector_path} is not a HuggingFace model, loading custom model")
        config['vision_encoder_path'] = args.vision_encoder_path
        config['model_path'] = args.model_path
        config['weights_file'] = args.projector_path
        config['loss_name'] = args.loss_name
        weights_path = args.projector_path
    else:
        weights_path = download_weights(args.projector_path, config['weights_file'])

    dataset = args.dataset
    if dataset == 'WEBEmo':
        train_dataset = WEBEmoDataset_precompute(config['vision_encoder_path'], phase='train')
        validation_dataset = WEBEmoDataset_precompute(config['vision_encoder_path'], phase='validation')
        test_dataset = WEBEmoDataset_precompute(config['vision_encoder_path'], phase='test')
    elif dataset == 'Emotion6':
        train_dataset = Emotion6Dataset_precompute(config['vision_encoder_path'], phase='train')
        validation_dataset = Emotion6Dataset_precompute(config['vision_encoder_path'], phase='validation')
        test_dataset = Emotion6Dataset_precompute(config['vision_encoder_path'], phase='test')
    elif dataset == 'EmoROI':
        train_dataset = EmoROIDataset_precompute(config['vision_encoder_path'], phase='train')
        validation_dataset = EmoROIDataset_precompute(config['vision_encoder_path'], phase='validation')
        test_dataset = EmoROIDataset_precompute(config['vision_encoder_path'], phase='test')
    elif dataset == "EmoSet":
        train_dataset = EmosetDataset_precompute(config['vision_encoder_path'], phase='train')
        validation_dataset = EmosetDataset_precompute(config['vision_encoder_path'], phase='validation')
        test_dataset = EmosetDataset_precompute(config['vision_encoder_path'], phase='test')
    elif dataset == "FI":
        train_dataset = FIDataset_precompute(config['vision_encoder_path'], phase='train')
        validation_dataset = FIDataset_precompute(config['vision_encoder_path'], phase='validation')
        test_dataset = FIDataset_precompute(config['vision_encoder_path'], phase='test')
    elif dataset == "UnbiasedEmo":
        test_dataset = UnbiasedEmo_Dataset()
    elif dataset == 'Abstract':
        test_dataset = AbstractDataset()
    elif dataset == 'ArtPhoto':
        test_dataset = ArtPhotoDataset()
    else:
        raise NotImplementedError(f'Dataset: {dataset}')

    test_dataloader = DataLoader(test_dataset, batch_size=args.per_device_eval_batch_size, shuffle=True, drop_last=False, num_workers=4)
 
    model = Model(
        vision_encoder_path=config['vision_encoder_path'],
        model_path=config['model_path'],
        tokenizer_path="meta-llama/Llama-2-7b-hf",
        num_epochs=args.num_train_epochs,
        dataset=dataset,
        run_name=args.run_name,
        precomputed=args.precomputed
    )
    model.loss_name = config['loss_name']
    if config['loss_name'] == 'CosineEmbedding':
        print(f"Using loss {config['loss_name']}")
        model.loss_func = torch.nn.CosineEmbeddingLoss(margin=0.5)
    elif config['loss_name'] == 'NTXent':
        print(f"Using loss {config['loss_name']}")
        model.loss_func = losses.NTXentLoss()

    print(model.projector)
    test(model, test_dataloader, projector_weights_path=weights_path)
