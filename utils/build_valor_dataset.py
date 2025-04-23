import os
import pandas as pd
from torch.utils.data import Dataset
import torch

data_path = "/datadisk/datasets/VALOR"
dataset_name = "desc_train"

class ValorDataset(Dataset):
    def __init__(self, data_path, dataset_name):
        self.json_data, self.audio_dir, self.img_dir = self.build_dataset(data_path, dataset_name)

    def build_dataset(self, data_path, dataset_name):
        json_path = os.path.join(data_path, f"annotation/{dataset_name}.json")
        audio_dir = os.path.join(data_path, "extraction/ast_audio")
        img_dir = os.path.join(data_path, "extraction/vivit_frames")
        json_data = pd.read_json(json_path).to_dict(orient="records")
        return json_data, audio_dir, img_dir

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        item = self.json_data[idx]
        return {
            "video_id": item["video_id"],
            "desc": item["desc"],
            "audio_path": os.path.join(self.audio_dir, f"{item['video_id']}.wav"),
            "img_path": os.path.join(self.img_dir, item["video_id"])
        }
    
class ValorCollactor:
    def __init__(self, tokenizer, text_transmitter, image_transmitter, audio_transmitter):
        self.tokenizer = tokenizer
        self.text_tx = text_transmitter
        self.img_tx = image_transmitter
        self.audio_tx = audio_transmitter

    @staticmethod
    def pad_to_length(embeds, max_length):
        padded_embeds = []
        for embed in embeds:
            if embed.size(0) > max_length:
                padded_embeds.append(embed[:max_length])
            else:
                pad_size = max_length - embed.size(1)
                padding = torch.zeros(embed.size(0), pad_size, embed.size(2), device=embed.device)
                padded_embeds.append(torch.cat([embed, padding], dim=1))
        return torch.stack(padded_embeds)

    def __call__(self, batch):
        video_ids = [item["video_id"] for item in batch]
        descs = [item["desc"] for item in batch]
        audio_paths = [item["audio_path"] for item in batch]
        img_paths = [item["img_path"] for item in batch]

        text_outputs = [self.text_tx(q) for q in descs]
        audio_outputs = [self.audio_tx(p) for p in audio_paths]
        img_outputs = [self.img_tx(p) for p in img_paths]

        text_embeds, text_features = zip(*text_outputs)
        audio_embeds, audio_features = zip(*audio_outputs)
        img_embeds, img_features = zip(*img_outputs)

        audio_embeds = torch.stack(audio_embeds, dim=0).squeeze(1)
        img_embeds = torch.stack(img_embeds, dim=0).squeeze(1)

        text_embeds = ValorCollactor.pad_to_length(list(text_embeds), max_length=256).squeeze(1)

        text_features = torch.stack(text_features).squeeze(1)
        audio_features = torch.stack(audio_features).squeeze(1)
        img_features = torch.stack(img_features).squeeze(1)


        for i, embed in enumerate(text_embeds):
            text_length = (text_embeds.ne(0)).sum(dim=1)

        return {
            "text_embeds": text_embeds,            # torch.Size([B, 128, D])
            "text_length": text_length,          # torch.Size([B])
            "audio_embeds": audio_embeds,          # torch.Size([B, L, D])
            "img_embeds": img_embeds,               # torch.Size([B, L, D])
            "text_features": text_features,          # torch.Size([B, D])
            "audio_features": audio_features,        # torch.Size([B, D])
            "img_features": img_features,            # torch.Size([B, D])
        }
