import os
import torch
import torch.nn as nn

from PIL import Image
import soundfile as sf
from typing import Optional
from datasets import load_dataset
from torch.autograd import Function


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

class text_transmitter(nn.Module):
    def __init__(self, model, text_tokenizer, if_train_encoder: bool):
        super().__init__()
        self.tokenizer = text_tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.embedding = model.model.get_input_embeddings().to(device)

        for param in self.embedding.parameters():
            param.requires_grad = if_train_encoder

    def forward(self, inputs):
        if isinstance(inputs, (str, list)):
            tokenized = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = tokenized.input_ids.to(device)
        elif isinstance(inputs, dict):
            input_ids = inputs["input_ids"].to(device)
        else:
            input_ids = inputs.to(device)
        
        text_tokens = self.embedding(input_ids)
        text_features = text_tokens.mean(dim=1)  # [batch, dim]
        return text_tokens, text_features  # [batch, seq_len, dim], [batch, dim]


class TokenSelector(nn.Module):
    def __init__(self, dim, target_len):
        super().__init__()
        self.target_len = target_len

    def forward(self, output_tokens):  # [batch, seq_len, dim]
        batch_size, seq_len, dim = output_tokens.shape
        
        indices = torch.linspace(0, seq_len - 1, steps=self.target_len, device=output_tokens.device).long()
        indices = indices.unsqueeze(0).expand(batch_size, -1)

        selected_tokens = torch.gather(output_tokens, 1, indices.unsqueeze(-1).expand(-1, -1, dim))
        
        return selected_tokens  # [batch, target_len, dim]
    

class img_transmitter(nn.Module):
    def __init__(self, img_encoder, img_processor, img_token_len, output_dim, if_train_encoder: bool):
        super().__init__()
        self.encoder = img_encoder
        self.processor = img_processor

        self.target_len = img_token_len
        self.hidden_size = img_encoder.config.hidden_size
        
        self.tokenSelect = TokenSelector(self.hidden_size, self.target_len)
        self.projection = nn.Linear(self.hidden_size, output_dim).to(device)

        for param in self.encoder.parameters():
            param.requires_grad = if_train_encoder

    @staticmethod
    def load_vivit_frames(frame_dir, num_frames=32):
        frame_files = sorted(os.listdir(frame_dir))[:num_frames]
        frames = [Image.open(os.path.join(frame_dir, f)) for f in frame_files]
        frames += [frames[-1]] * (num_frames - len(frames)) if len(frames) < num_frames else []
        return frames

    def forward(self, img_file):
        frames = img_transmitter.load_vivit_frames(img_file, num_frames=32)
        inputs = self.processor(frames, return_tensors="pt").to(device)
        outputs = self.encoder(**inputs)
        output_tokens = outputs.last_hidden_state
        output_features = outputs.pooler_output
        
        target_tokens = self.tokenSelect(output_tokens)
        img_tokens = self.projection(target_tokens)
        img_features = self.projection(output_features)

        return img_tokens, img_features  # [batch, target_len, dim], [batch, dim]

class audio_transmitter(nn.Module):
    def __init__(self, audio_encoder, audio_processor, audio_token_len, output_dim, if_train_encoder: bool):
        super().__init__()
        self.encoder = audio_encoder
        self.processor = audio_processor

        self.target_len = audio_token_len
        self.hidden_size = audio_encoder.config.hidden_size

        self.tokenSelect = TokenSelector(self.hidden_size, self.target_len)
        self.projection = nn.Linear(self.hidden_size, output_dim).to(device)
        

        for param in self.encoder.parameters():
            param.requires_grad = if_train_encoder

    def forward(self, audio_file):
        if not os.path.exists(audio_file):
            print(f"Audio file {audio_file} not found")
            return None
        audio, sample_rate = sf.read(audio_file)
        inputs = self.processor(audio, return_tensors="pt", sampling_rate=sample_rate).to(device)
        outputs = self.encoder(**inputs)
        output_tokens = outputs.last_hidden_state
        output_features = outputs.pooler_output

        target_tokens = self.tokenSelect(output_tokens)
        audio_tokens = self.projection(target_tokens)
        audio_features = self.projection(output_features)

        return audio_tokens, audio_features