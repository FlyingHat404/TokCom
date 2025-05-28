# Token Communication-Driven Multimodal Large Model

This repository provides the code implementation for our proposed **token communication paradigm** for distributed multimodal large model deployment. By performing cross-modal alignment and task-oriented fine-tuning, our method enables efficient, task-oriented token transmission.

You can also view this as a fine-tuning baseline for multimodal large models. We unify visual, audio, and textual modalities into a shared latent space and input them into a foundation model for **audio-visual question answering (AVQA)**.

---

## 📁 Dataset Preparation

The training process is divided into **two stages**:
1. **Cross-modal alignment**
2. **Task-oriented fine-tuning**

### Stage 1: Cross-Modal Alignment  
- Dataset: [VALOR-32K](#) (originally for audio-visual captioning)
- Preprocessing:
  - Run `utils/raw_video_preprocess.py` to:
    - Extract frames from video
    - Extract audio from video

Processed directory structure (example for VALOR):
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

### Stage 2: Task-Oriented Fine-Tuning  
- Dataset: [MUSIC-AVQA](#)

---

## 🧠 Model Preparation

- **Text**: [Qwen2.5-1.5B](#)
- **Audio**: [AST (Audio Spectrogram Transformer)](#)
- **Visual**: [ViViT (Video Vision Transformer)](#)

Models can be downloaded from HuggingFace or via `utils/hf-download.py`.

---

## 🚀 Training

### Cross-Modal Alignment  
Run:
```bash
python task_align.py
```
- Checkpoints will be saved in `checkpoint/xxx.pth` (create the folder beforehand).
- Visualization:
  - Temperature comparison  
    ![](imgs/Temp.png)
  - T-SNE plots at different contrastive temperatures  
    ![](imgs/000.png)  
    ![](imgs/003.png)  
    ![](imgs/007.png)  
    ![](imgs/013.png)

### Task-Oriented Fine-Tuning  
Run:
```bash
python task_avqa.py
```
- Unlike original AVQA implementations, our model **generates autoregressively** rather than using a classification head.
- The output spans the entire vocabulary.
- See `modules/foundation_model.py` for details.

---

## 💡 Notes for Beginners
- Set breakpoints to inspect tensor shapes during each stage — it's a great way to understand the data flow.

---

## 📖 Citation

If you find our work helpful, please consider citing:

```bibtex
@misc{zhang2025tokencommunicationdrivenmultimodallarge,
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

Feel free to star ⭐ this repository and contribute back!
