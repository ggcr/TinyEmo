import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLM:
    def __init__(self, model_path, tokenizer_path=None):
        print(f"Loading AutoModelForCausalLM with model_path: {model_path}")
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        # If we use Apple's OpenELM we require to use Meta's LLaMA tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if 'apple' in model_path else model_path, 
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
        ).to(self.device)

        self.model.eval()
        print(self.tokenizer.name_or_path)
        print(next(self.model.parameters()).dtype)

    def compute_embeddings_images(self, aligned_features):
        return aligned_features
    
    def get_embds(self, text):
        with torch.no_grad():
            # text = ['cheerful' if t == 'joy' else t for t in text]
            # text = ['affection' if t == 'love' else t for t in text]
            # text = [f"an image expressing {t} emotion" for t in text]
            # text = [f"{t} emotion" for t in text]
            # text = [f"an image expressing {emo} emotion" for emo in text]
            # text = [f"a {emotion} scene" for emotion in text]
            # text = [f"a scene evoking {emotion} to viewers" for emotion in text]
            # text = [f"a scene evoking {emotion}" for emotion in text]
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            p_token = self.tokenizer(
                text,
                add_special_tokens=True,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512, # BERT-like
            )
            input_ids = p_token['input_ids'].to(self.device)
            attention_mask = p_token['attention_mask'].to(self.device)
            embeds = self.model.get_input_embeddings()(input_ids) * attention_mask.unsqueeze(-1)
            return embeds.mean(dim=1)