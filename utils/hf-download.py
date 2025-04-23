#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# hf-mirror for China users

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download
import huggingface_hub
huggingface_hub.login("******")

model_name = "Qwen/Qwen2.5-1.5B"

# while True to handle network error
while True:
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir_use_symlinks=True,
            ignore_patterns=["*.bin"],  # ignore .bin files
            local_dir="/mnt/Qwen/Qwen2.5-1.5B",  # local directory
            token="******",   # huggingface token
            resume_download=True
        )
        break
    except Exception as e:
        print(f"Download failed: {e}")