# Token Communication-Driven Multimodal Large Models in Resource-Constrained Multiuser Networks

  The code implementation of our proposed token communication paradigm for distributed multimodal large model deployment [(arXiv paper)](https://arxiv.org/abs/2505.07841). By performing cross-modal alignment and task-oriented fine-tuning, out method enables efficient, task-oriented token transmission.

  <!-- ⭐ **The simple and straightforward implementation is suitable to be taken as a baseline :)** -->


---

## 📁 Dataset Preparation
The entire training process consists of two stages: **cross-modal alignment** and **task-oriented fine-tuning**.

For the cross-modal alignment stage, we use the [VALOR-32K](https://casia-iva-group.github.io/projects/VALOR/data.html) dataset.
For the task-oriented fine-tuning stage, we use the [MUSIC-AVQA](https://gewu-lab.github.io/MUSIC-AVQA/) dataset.

After downloading the raw data into the corresponding directory, please run ```utils/raw_video_preprocess.py``` to split each video into image frames and extract the associated audio.
Using the VALOR dataset as an example, the folder structure of the processed data should be organized as follows:

```
VALOR/
├── annotations/
│   ├── desc_train
│   ├── desc_val
│   └── desc_test
└── extraction/
    ├── ast_audio/
    │   ├── xxxxx0.wav
    │   ├── xxxxx1.wav
    └── vivit_frames/
        ├── xxxxx0/
        │   ├── frame_0001.png
        │   ├── frame_0002.png
        └── xxxxx1/
            ├── frame_0001.png
            ├── frame_0002.png
```

---

## 🧠 Model Preparation

- **Foundation model**: [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)
- **Text**: [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)
- **Audio**: [AST (Audio Spectrogram Transformer)](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
- **Visual**: [ViViT (Video Vision Transformer)](https://huggingface.co/google/vivit-b-16x2)

Models can be downloaded from HuggingFace or via `utils/hf-download.py`.

---

## 🚀 Training

### Cross-Modal Alignment  
```bash
python task_align.py
```
- Checkpoints will be saved in `checkpoint/xxx.pth` (create the folder beforehand).

We also provide **Rounds vs. Loss curves** and **t-SNE plots** corresponding to different contrastive temperatures during the alignment stage.
#### Rounds vs. Loss
<p align="center">
  <img src="imgs/Temp.png" width="400"/>
</p>

#### t-SNE Plots with Different Contrastive Temperatures τ
<p align="center">
  <img src="imgs/000.png" width="200"/>
  <img src="imgs/003.png" width="200"/>
  <img src="imgs/007.png" width="200"/>
  <img src="imgs/013.png" width="200"/>
</p>
From left to right: without alignment, τ = 0.03, τ = 0.07, τ = 0.13.

### Task-Oriented Fine-Tuning  
```bash
python task_avqa.py
```

It’s noteworthy that we evaluate the autoregressive generation capability of the foundation model rather than using a classification head as the origin implementation does [(MUSIC-AVQA)](https://github.com/GeWu-Lab/MUSIC-AVQA). Specifically, the output space of our model spans the entire vocabulary instead of a limited set of label indices.  See `modules/foundation_model.py` for details.

---

For beginners, it is recommended to set breakpoints to check the shape of tensors at each step of the process :)

---

## 📖 Citation

If you find our work helpful, please consider citing:

```bibtex
@misc{junhe2025tokcom,
  title={Token Communication-Driven Multimodal Large Models in Resource-Constrained Multiuser Networks},
  author={Junhe Zhang and Wanli Ni and Pengwei Wang and Dongyu Wang},
  year={2025},
  eprint={2505.07841},
  archivePrefix={arXiv},
  primaryClass={cs.NI},
  url={https://arxiv.org/abs/2505.07841}
}
```

---

## 🎥 Acknowledgment

Thanks to this excellent tutorial on building your own LLaVA model:  
👉 [Bilibili Video](https://space.bilibili.com/45156039/lists/3213902)

---
