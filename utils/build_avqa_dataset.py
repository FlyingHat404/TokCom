import os
import json
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
import re


data_path = "datasets/MUSIC-AVQA"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

class AVQADataset(Dataset):
    def __init__(self, data_path, dataset_name="avqa-train"):
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
        question = item["question_content"]

        # if there is a templated value in the question, replace it
        if re.search(r"<[^>]+>", question):
            templ_values = json.loads(item["templ_values"]) if isinstance(item["templ_values"], str) else item["templ_values"]
            
            placeholders = re.findall(r"<[^>]+>", question)
            
            for i, placeholder in enumerate(placeholders):
                if i < len(templ_values):
                    question = question.replace(placeholder, templ_values[i], 1)

        return {
            "question": question,
            "answer": item["anser"],
            "audio_path": os.path.join(self.audio_dir, f"{item['video_id']}.wav"),
            "img_path": os.path.join(self.img_dir, item["video_id"])
        }
    
class AVQACollactor:
    def __init__(self, tokenizer, text_tx, image_tx, audio_tx):
        self.tokenizer = tokenizer
        self.text_tx = text_tx
        self.img_tx = image_tx
        self.audio_tx = audio_tx

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
        texts = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]
        audio_paths = [item["audio_path"] for item in batch]
        img_paths = [item["img_path"] for item in batch]

        audio_embeds = torch.stack([self.audio_tx(p)[0] for p in audio_paths], dim=0).squeeze(1)
        img_embeds = torch.stack([self.img_tx(p)[0] for p in img_paths], dim=0).squeeze(1)
        audio_mask = torch.ones(audio_embeds.shape[0], audio_embeds.shape[1], device=audio_embeds.device)
        img_mask = torch.ones(img_embeds.shape[0], img_embeds.shape[1], device=img_embeds.device)


        text_embeds = [self.text_tx(q)[0] for q in texts]
        
        # pad to same length
        text_embeds = AVQACollactor.pad_to_length(text_embeds, max_length=32).squeeze(1)

        # Generate text_mask and set padding parts to 0
        text_mask = torch.zeros(text_embeds.shape[0], text_embeds.shape[1], device=text_embeds.device)
        for i, embed in enumerate(text_embeds):
            actual_length = (embed.sum(dim=1) != 0).sum().item()  # Count non-padding tokens
            text_mask[i, :actual_length] = 1
        

        # there is only one token in the label
        label_ids = [self.tokenizer(ans, return_tensors="pt")["input_ids"][0][0] for ans in answers]
        labels = torch.tensor(label_ids).unsqueeze(1).to(device)  # shape: [batch_size, 1]
        
        # Concatenate masks
        attention_mask = torch.cat([audio_mask, img_mask, text_mask], dim=1)

        return {
            "text_embeds": text_embeds,
            "audio_embeds": audio_embeds,
            "img_embeds": img_embeds,
            "labels": labels,
            "attention_mask": attention_mask 
        }