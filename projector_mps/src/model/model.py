import torch
import tqdm
import numpy as np
import wandb
import os

from src.model.vision_encoder import VisionEncoder
from src.model.llm import LLM
from src.model.projector import AlignmentMLP
from src.model.utils import pos_neg_infoNCE, check_gradients

class Model:
    def __init__(
        self, 
        vision_encoder_path: str,
        model_path: str,
        tokenizer_path: str,
        report_to: str = None,
        dataset: str = None,
        run_name: str = None,
        output_dir: str = '/tmp/',
        num_epochs: int = 1,
        len_dataloader: int = 1000,
        precomputed: bool = False,
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.precomputed = precomputed
        if not self.precomputed:
            self.vision_encoder = VisionEncoder(model_path=vision_encoder_path)
        else:
            self.vision_encoder = VisionEncoder(model_path=vision_encoder_path)
            print("Skipping loading the Vision Encoder.")
            print("Using precomputed features.")
        self.open_elm = LLM(model_path=model_path, tokenizer_path=tokenizer_path)

        if 'openai/clip' in vision_encoder_path:
            if 'openai/clip-vit-base-patch16' == vision_encoder_path:
                vision_dims = 512
            else:
                vision_dims = 768
        elif 'google/siglip-so400m-patch14-384' in vision_encoder_path:
            vision_dims = 1152
        elif 'facebook/dinov2' in vision_encoder_path:
            vision_dims = 1024

        if 'microsoft/Phi-2' in model_path: # phi-2
            hidden_dim = 2560
        elif 'microsoft/Phi-3-mini-128k-instruct' in model_path: # phi-3
            hidden_dim = 3072
        elif 'TinyLlama' in model_path: # TinyLlama/TinyLlama-1.1B-Chat-v1.0
            hidden_dim = 2048

        self.projector = AlignmentMLP(input_dim=vision_dims, hidden_dim= self.open_elm.model.config.model_dim if hasattr(self.open_elm.model.config, 'model_dim') else hidden_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(self.projector.parameters(), lr=2e-5)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.0005, total_steps=len_dataloader*num_epochs, pct_start=0.005)
        
        self.num_epochs = num_epochs
        self.report_to = report_to
        self.output_dir = output_dir
        self.dataset = dataset
        self.precomputed = precomputed
        self.loss_name = 'infoNCE' # do not change this, if we choose another LOSS from the scripts, we will modify it externally


    def step(self, images, labels, sentiments, return_image_embeds=False):
        # Encode images and project them to LLM
        if self.precomputed:
            features = images.to(self.device)
        else:
            features = self.vision_encoder.encode_images(images).to(self.device)
        image_embds = self.projector(features).to(self.device)

        # Compute Embeddings
        image_embds = self.open_elm.compute_embeddings_images(image_embds).to(self.device)
        # text_embds = torch.vstack([self.open_elm.model.get_input_embeddings()(torch.tensor(self.open_elm.tokenizer.encode(sentiment)).to(self.device)).mean(dim=0) for sentiment in list(sentiments)]).to(self.device)
        text_embds = self.open_elm.get_embds(list(sentiments)).to(self.device)
        
        # L2 Normalize to boost cosine similarity metrics
        image_embds = torch.nn.functional.normalize(image_embds, p=2, dim=1)
        text_embds = torch.nn.functional.normalize(text_embds, p=2, dim=1)

        # Compute metrics loss
        if self.loss_name == 'infoNCE':
            loss = pos_neg_infoNCE(image_embds, text_embds, labels, self.device)
        elif self.loss_name == 'CosineEmbedding':
            similarity_labels = torch.ones(labels.size(0)).to(self.device)
            loss = self.loss_func(image_embds, text_embds, similarity_labels)
        elif self.loss_name == 'NTXent':
            loss = self.loss_func(image_embds, labels, ref_emb=text_embds, ref_labels=labels)

        if return_image_embeds == True:
            return loss, image_embds
        return loss
    
        
    def train(self, dataloader, epoch):
        self.projector.train()
        running_loss = []
        for batch_idx, (images, labels, sentiments) in enumerate(tqdm.tqdm(dataloader)):
            self.optimizer.zero_grad()
            labels = labels.to(self.device)
            sentiments = list(sentiments)

            loss = self.step(images, labels, sentiments)
            if loss.item() != -1: 
                loss.backward()
                running_loss.append(loss.item())
            self.optimizer.step()
            self.scheduler.step()

            if batch_idx % 100 == 0:
                # Check for gradients
                gradients = check_gradients(self.projector)
                tqdm.tqdm.write(f'Epoch [{str(epoch).zfill(1)}/{str(self.num_epochs).zfill(1)}] | Step [{str(batch_idx).zfill(4)}/{str(len(dataloader)).zfill(4)}] | Step Loss {loss.item():.5} | Avg Running Loss {torch.Tensor(running_loss).mean():.5} | MLP Gradient Norms {gradients[0]:.3} {gradients[1]:.3} {gradients[2]:.3}')
        # Save model
        os.makedirs(self.output_dir, exist_ok=True)
        connector_output_path = os.path.join(self.output_dir, f'pytorch_model_{epoch}.bin')
        torch.save(self.projector.state_dict(), connector_output_path)

        mean_loss = torch.Tensor(running_loss).mean()

        return mean_loss
    

    def eval(self, dataloader, epoch=None):
        self.projector.eval()
        running_loss = []
        embeds = []
        with torch.inference_mode():
            for batch_idx, (images, labels, sentiments) in enumerate(tqdm.tqdm(dataloader)):
                labels = labels.to(self.device)
                sentiments = list(sentiments)
                loss = self.step(images, labels, sentiments)
                if loss.item() != -1: 
                    running_loss.append(loss.item())
        mean_loss = torch.Tensor(running_loss).mean()
        return mean_loss