import torch
import torch.nn as nn

import torch
import torch.nn as nn

class ChannelSimulator(nn.Module):
    def __init__(self, snr_db):
        super().__init__()
        self.snr_db = snr_db
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, x.ndim))
        signal_power = torch.mean(x ** 2, dim=dims, keepdim=True)
        
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr_linear
        
        noise = torch.randn_like(x) * torch.sqrt(torch.clamp(noise_power, min=1e-12))
        return x + noise
    