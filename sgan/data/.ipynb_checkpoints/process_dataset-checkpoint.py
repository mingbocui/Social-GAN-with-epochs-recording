import os
import trajnettools
import numpy as np
import torch
data_dir = 'datasets/dataset_aux_sgan/train/train'

files_path = []
all_files = os.listdir(data_dir)
for file in all_files:
    if os.path.splitext(file)[1] == '.ndjson':
        files_path.append(file)
        
all_files = [os.path.join(data_dir, _path) for _path in files_path]
seq_list = []
seq_start_end = []
for path in all_files:
    train_scenes = list(trajnettools.load_all(path, scene_type=None))
    seq_list_prev = [scene for _, scene in train_scenes]
    seq_start_end.append(len(seq_list_prev))
    seq_list_prev = np.concatenate(seq_list_prev, axis=1)
    seq_list_prev = np.asarray(seq_list_prev)
    seq_list.append(seq_list_prev)