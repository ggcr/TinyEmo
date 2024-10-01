import torch
import torch.nn as nn
import sys

class AlignmentMLP(nn.Module):
    """
    MLP that acts as the alignment between ViT and LLM.

    :param input_dim: Vision encoder output size
    :param hidden_dim: Hidden layer size in MLP
    :param output_dim: Output projection size == LLM embedding dims
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        print(f"Loading Projector with {input_dim} input dims and {hidden_dim} hidden dims")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Checking the norm
        x = x.float()
        return self.model(x)