import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenReceiver(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.receiver = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        received_token = self.receiver(token)  # [batch, seq_len, output_dim]

        return received_token
    
class MultimodalPooler(nn.Module):
    def __init__(self):
        super(MultimodalPooler, self).__init__()

    def forward(self, token: torch.Tensor, modality: str):
        """
        For text, we use the mean of all tokens.
        For img and audio, please refer to the according repositories.
        """
        if modality == 'text':
            feature = token.mean(dim=1)
        elif modality == 'img':
            feature = token[:, 0]
        elif modality == 'audio':
            feature = (token[:, 0] + token[:, 1]) / 2
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        
        return feature