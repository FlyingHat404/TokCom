import torch
import random
import matplotlib
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoProcessor, ASTModel, VivitImageProcessor, VivitModel
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from modules.foundation_model import Qwen2ForVQA
from modules.token_transmitter import text_transmitter, img_transmitter, audio_transmitter
from modules.channel import ChannelSimulator
from modules.token_receiver import TokenReceiver, MultimodalPooler
from utils.build_valor_dataset import ValorDataset, ValorCollactor
from torch.optim.lr_scheduler import CosineAnnealingLR

Train = False # training or using checkpoint to generate t-SNE directly

SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_path = "/mnt/Qwen/Qwen2.5-1.5B"
model = Qwen2ForVQA(model_path).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_path)
img_encoder = VivitModel.from_pretrained("/datadisk/google/vivit-b-16x2").to(device)
img_processor = VivitImageProcessor.from_pretrained("/datadisk/google/vivit-b-16x2")
audio_encoder = ASTModel.from_pretrained("/datadisk/MIT/ast-finetuned-audioset-10-10-0.4593").to(device)
audio_processor = AutoProcessor.from_pretrained("/datadisk/MIT/ast-finetuned-audioset-10-10-0.4593")

hidden_size = model.model.config.hidden_size
audio_token_len = 128
img_token_len = 128

text_tx = text_transmitter(model, tokenizer, if_train_encoder=False).to(device)
img_tx = img_transmitter(img_encoder, img_processor, img_token_len, hidden_size, if_train_encoder=False).to(device)
audio_tx = audio_transmitter(audio_encoder, audio_processor, audio_token_len, hidden_size, if_train_encoder=False).to(device)
text_receiver = TokenReceiver(hidden_size, hidden_size).to(device)
img_receiver = TokenReceiver(hidden_size, hidden_size).to(device)
audio_receiver = TokenReceiver(hidden_size, hidden_size).to(device)
channel = ChannelSimulator(snr_db=10.0)
pooler = MultimodalPooler().to(device)

def contrastive_loss(modality_features, text_features, temperature=0.01):
    modality_features = F.normalize(modality_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    similarity_matrix = torch.matmul(modality_features, text_features.T) / temperature
    positives = torch.diag(similarity_matrix)
    denominator = torch.sum(torch.exp(similarity_matrix), dim=1)
    return -torch.log(torch.exp(positives) / denominator).mean()

def save_losses_to_excel(loss_records, filename="alignment_losses.xlsx"):
    df = pd.DataFrame(loss_records)
    df.to_excel(filename, index=False, float_format="%.8f", engine="openpyxl")

def save_tsne_to_excel(features, labels, filename):
    tsne = TSNE(n_components=2, random_state=SEED)
    reduced = tsne.fit_transform(features)

    label_names = {0: 'Text', 1: 'Audio', 2: 'Image'}
    modalities = [label_names[int(l)] for l in labels]

    df = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "label": labels,
        "modality": modalities
    })

    df.to_excel(filename, index=False, float_format="%.6f", engine="openpyxl")

def plot_tsne(features, labels, filename, color_map):
    matplotlib.rcParams['font.family'] = 'serif'
    tsne = TSNE(n_components=2, random_state=SEED)
    reduced = tsne.fit_transform(features)
    plt.figure(figsize=(10, 8))

    label_names = {0: 'Text', 1: 'Audio', 2: 'Image'}

    for label, color in color_map.items():
        mask = labels == label
        plt.scatter(reduced[mask, 0], reduced[mask, 1], c=color, label=label_names[label], alpha=0.5)

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    legend = plt.legend(loc='upper right', fontsize=12, frameon=True)
    legend.get_frame().set_linewidth(2)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_facecolor('white') 

    plt.grid(False)
    plt.xticks(fontname='serif')
    plt.yticks(fontname='serif')
    plt.savefig(filename, dpi=1200, bbox_inches='tight')
    plt.close()

def generate_tsne(train_loader, channel, text_receiver, audio_receiver, img_receiver, device, num_batch=50):
    orig_text_feats, orig_audio_feats, orig_img_feats = [], [], []
    aligned_text_feats, aligned_audio_feats, aligned_img_feats = [], [], []

    with torch.no_grad():
        for batch in train_loader:
            if len(orig_text_feats) >= num_batch:
                break

            # original features
            orig_text_feats.append(batch['text_features'].to(device))
            orig_audio_feats.append(batch['audio_features'].to(device))
            orig_img_feats.append(batch['img_features'].to(device))

            # get token
            text_token = batch['text_embeds'].to(device)
            audio_token = batch['audio_embeds'].to(device)
            img_token = batch['img_embeds'].to(device)

            # get features token -> channel -> receiver -> pool
            aligned_text = text_receiver(channel(text_token))
            aligned_audio = audio_receiver(channel(audio_token))
            aligned_img = img_receiver(channel(img_token))

            aligned_text_feat = pooler(aligned_text, 'text')
            aligned_audio_feat = pooler(aligned_audio, 'audio')
            aligned_img_feat = pooler(aligned_img, 'img')

            aligned_text_feats.append(aligned_text_feat)
            aligned_audio_feats.append(aligned_audio_feat)
            aligned_img_feats.append(aligned_img_feat)

            print(f"Processed Batch {len(orig_text_feats)}")
            torch.cuda.empty_cache()

    orig_text = torch.cat(orig_text_feats, dim=0).detach().cpu().numpy()
    orig_audio = torch.cat(orig_audio_feats, dim=0).detach().cpu().numpy()
    orig_img = torch.cat(orig_img_feats, dim=0).detach().cpu().numpy()

    aligned_text = torch.cat(aligned_text_feats, dim=0).detach().cpu().numpy()
    aligned_audio = torch.cat(aligned_audio_feats, dim=0).detach().cpu().numpy()
    aligned_img = torch.cat(aligned_img_feats, dim=0).detach().cpu().numpy()

    torch.cuda.empty_cache()

    labels_before = np.concatenate([
        np.zeros(orig_text.shape[0]), 
        np.ones(orig_audio.shape[0]), 
        np.full(orig_img.shape[0], 2)
    ])

    labels_after = np.concatenate([
        np.zeros(aligned_text.shape[0]), 
        np.ones(aligned_audio.shape[0]),
        np.full(aligned_img.shape[0], 2)
    ])

    custom_colors = {
        0: '#12507B',  # text
        1: '#FAC03D',  # audio
        2: '#5DA39D'   # image
    }

    # plot_tsne(np.concatenate([orig_text, orig_audio, orig_img], axis=0), 
    #           labels_before, 
    #           "tsne_before_alignment.png",
    #           custom_colors)

    # plot_tsne(np.concatenate([aligned_text, aligned_audio, aligned_img], axis=0), 
    #           labels_after, 
    #           "tsne_after_alignment.png",
    #           custom_colors)

    save_tsne_to_excel(np.concatenate([orig_text, orig_audio, orig_img], axis=0), 
                   labels_before, 
                   "tsne_before_alignment.xlsx")

    save_tsne_to_excel(np.concatenate([aligned_text, aligned_audio, aligned_img], axis=0), 
                    labels_after, 
                    "tsne_after_alignment.xlsx")
    
def save_model_checkpoint(filepath, img_tx, audio_tx, text_receiver, img_receiver, audio_receiver):
    torch.save({
        'img_tx': img_tx.state_dict(),
        'audio_tx': audio_tx.state_dict(),
        'text_receiver': text_receiver.state_dict(),
        'img_receiver': img_receiver.state_dict(),
        'audio_receiver': audio_receiver.state_dict()
    }, filepath)

if __name__ == '__main__':
    train_dataset = ValorDataset("/datadisk/datasets/VALOR", dataset_name="desc_train")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=ValorCollactor(tokenizer, text_tx, img_tx, audio_tx))

    optimizer = torch.optim.AdamW(
        list(img_tx.parameters()) + list(audio_tx.parameters()) +
        list(text_receiver.parameters()) + list(img_receiver.parameters()) +
        list(audio_receiver.parameters()), lr=1e-4
    )
    if Train:
        max_iters = 500
        loss_records = []
        scheduler = CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-6)

        for iter_idx, batch in enumerate(train_loader):
            if iter_idx >= max_iters:
                break
            
            # embeds after transmitter
            text_embeds = batch["text_embeds"].to(device)
            audio_embeds = batch["audio_embeds"].to(device)
            img_embeds = batch["img_embeds"].to(device)

            text_transmitted = channel(text_embeds)
            audio_transmitted = channel(audio_embeds)
            img_transmitted = channel(img_embeds)

            # embeds after receiver
            text_received = text_receiver(text_transmitted)
            audio_received = audio_receiver(audio_transmitted)
            img_received = img_receiver(img_transmitted)

            text_features = pooler(text_received, 'text')
            audio_features = pooler(audio_received, 'audio')
            img_features = pooler(img_received, 'img')

            # loss
            text_loss = F.mse_loss(text_embeds, text_received)
            audio_loss = contrastive_loss(audio_features, text_features)
            img_loss = contrastive_loss(img_features, text_features)

            optimizer.zero_grad()
            text_loss.backward(retain_graph=True)
            audio_loss.backward(retain_graph=True)
            img_loss.backward()
            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()

            loss_records.append({'Iteration': iter_idx, 'Text Reconstruction': text_loss.item(), 'Audio Contrastive': audio_loss.item(), 'Image Contrastive': img_loss.item()})
            print(f"Iteration {iter_idx}: Text Loss: {text_loss.item():.4f}, Audio Loss: {audio_loss.item():.4f}, Image Loss: {img_loss.item():.4f}")

        save_model_checkpoint('/mnt/checkpoints/007-multimodal_transceivers.pth', img_tx, audio_tx, text_receiver, img_receiver, audio_receiver)
        save_losses_to_excel(loss_records)
    else:
        checkpoint = torch.load('/mnt/checkpoints/013-multimodal_transceivers.pth', map_location='cpu')
        img_tx.load_state_dict(checkpoint['img_tx'])
        audio_tx.load_state_dict(checkpoint['audio_tx'])
        text_receiver.load_state_dict(checkpoint['text_receiver'])
        img_receiver.load_state_dict(checkpoint['img_receiver'])
        audio_receiver.load_state_dict(checkpoint['audio_receiver'])
    generate_tsne(train_loader, channel, text_receiver, audio_receiver, img_receiver, device)