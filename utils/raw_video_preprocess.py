import os
import os.path as P
import ffmpeg
import json
import tqdm
import numpy as np
import threading
import time
import multiprocessing
from multiprocessing import Pool
import subprocess



### change different datasets
input_path = 'datasets/MUSIC-AVQA/raw_videos_1'
output_path = 'datasets/MUSIC-AVQA/extraction_1'
data_list = os.listdir(input_path)



def execCmd(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text

def pipline(video_path, video_probe, output_dir, fps, sr):
    video_name = os.path.basename(video_path)
    video_basename = video_name.replace(".mp4", "")
    
    # video extraction, scale to 224x224
    vivit_frame_dir = P.join(output_dir, "vivit_frames", video_basename)
    os.makedirs(vivit_frame_dir, exist_ok=True)
    cmd = (
        "ffmpeg -loglevel error -i {} -vsync 0 -f image2 "
        "-vf \"fps=fps={:.02f},scale=224:224\" -qscale:v 2 {}/frame_%04d.jpg"
    ).format(video_path, fps, vivit_frame_dir)
    subprocess.call(cmd, shell=True)
    
    # audio extraction
    ast_audio_dir = P.join(output_dir, "ast_audio")
    os.makedirs(ast_audio_dir, exist_ok=True)
    audio_file_path = P.join(ast_audio_dir, video_basename + ".wav")
    cmd = (
        "ffmpeg -i {} -loglevel error -f wav -vn -ac 1 -ab 16k -ar {} -y {}"
    ).format(video_path, sr, audio_file_path)
    subprocess.call(cmd, shell=True)

def extract_thread(video_id):
    video_file = os.path.join(input_path, video_id)
    if not os.path.exists(video_file):
        return
    probe = ffmpeg.probe(video_file)
    # fps = 0.6, sr = 16000
    pipline(video_file, probe, output_path, fps=0.6, sr=16000)

def extract_all(video_ids, thread_num, start):
    print("Total video:", len(video_ids))
    for video_id in tqdm.tqdm(video_ids):
        extract_thread(video_id)  

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    thread_num = 50
    start = 0

    print("To be processed", len(data_list))
    extract_all(data_list, thread_num, start)