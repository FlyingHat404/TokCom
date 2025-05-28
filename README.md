# Token Communication-Driven Multimodal Large Models in Resource-Constrained Multiuser Networks

  The code implementation of our propose **token communication** paradigm for distributed multimodal large model deployment. By performing **cross-modal alignment** and **task-oriented fine-tuning**, out method enables efficient, task-oriented token transmission.

  Moreover, you can also simply view it as a fine-tuning implementation of a multimodal large model. We unify the visual, audio, and textual modality data into the same latent space and input it to the foundation model to perform an audio-visual question answering task. 

  The simple and straightforward implementation is suitable to be taken as a baseline :)


---

## 📁 Dataset Preparation

The training process is divided into **two stages**:
1. **Cross-modal alignment**
2. **Task-oriented fine-tuning**

### Stage 1: Cross-Modal Alignment  
- Dataset: [VALOR-32K](https://casia-iva-group.github.io/projects/VALOR/data.html)
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
- Dataset: [MUSIC-AVQA](https://gewu-lab.github.io/MUSIC-AVQA/)

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
Run:
```bash
python task_align.py
```
- Checkpoints will be saved in `checkpoint/xxx.pth` (create the folder beforehand). We also provide the Rounds vs. Loss and T-SNE pictures of different contrastive temperature for the alignment stage.
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
- It’s noteworthy that we evaluate the autoregressive generation capability of the foundation model rather than using a classification head as the origin implementation does [MUSIC-AVQA](https://github.com/GeWu-Lab/MUSIC-AVQA). Specifically, the output space of our model spans the entire vocabulary instead of a limited set of label indices.  See `modules/foundation_model.py` for details.

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

Feel free to star ⭐ this repository and contribute back!
