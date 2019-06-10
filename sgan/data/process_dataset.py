import os
import trajnettools
import numpy as np
import torch
import pandas as pd

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
    
    

def remove_nan(train_scenes):
    list_final = []
    len_train_scenes = len(train_scenes)
    for i in range(len_train_scenes):
        scene_list = train_scenes[i].tolist()
        conc_scene = np.concatenate(scene_list, axis=1)
        conc_scene_df = pd.DataFrame(conc_scene)
        conc_scene_df_no_nan = conc_scene_df.dropna(axis=0,how='any')
        l = np.asarray(conc_scene_df_no_nan)
        k = np.hsplit(l,21)
        o = np.asarray(k)
        list_final.append(o)
    return list_final