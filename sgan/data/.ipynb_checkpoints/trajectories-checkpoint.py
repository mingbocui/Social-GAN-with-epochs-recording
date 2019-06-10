import logging
import os
import math

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def seq_collate(data):#注意这里的data是一个list，包含了所有的TrajDataset
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list) = zip(*data)
    
    #print(len(data)) 64
    #print("split")
    #print(len(obs_seq_list)) 64
    #print("finished")
    #print(len(obs_seq_list)) 64
    #print("split")
    #print(np.shape(obs_seq_list[0])) [3,2,8]
    #print("finished")

    _len = [len(seq) for seq in obs_seq_list]#_len = [10,2,3,...]
    cum_start_idx = [0] + np.cumsum(_len).tolist()# =[0, 10, 12, 15]
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)#
    #print(np.shape(obs_traj)) [8,759,2]
    
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end
    ]
    #print(np.shape(loss_mask))
    #print("finished one seq_collate")

    return tuple(out)


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0
    
def add_nan(frame):
    length = len(frame)
    ll = np.asarray(frame)
    ll = np.expand_dims(ll,1)
    kk = np.full((length,2),np.nan)
    ff = np.concatenate([ll,ll,kk.tolist()], axis=1)
    return ff


def handle_nan(sequence):
    df = pd.DataFrame(sequence)
    c = df.interpolate(limit_direction='both') #replace nan with interpolate values
    return c.values 


class TrajectoryDataset_train(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t'
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset_train, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len # frames, observe 8 frames and predict 8 frames
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))#for all files we have 4110 sequences(take hotel for example), but we have filtered some of them

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                #print(curr_seq_data.shape)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    #print(np.isnan(curr_ped_seq))
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    
                    if(pad_end - pad_front == self.seq_len):
                        curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                 
                    if pad_end - pad_front != self.seq_len:
                        #continue
                        frame_tmp = frames[idx:idx + self.seq_len]
                        curr_ped_seq_tmp = add_nan(frame_tmp)
                        #print(curr_ped_seq_tmp)
                        for i in range(pad_end - pad_front):
                            frame_id_tmp = frame_tmp.index((curr_ped_seq[i, 0]))
                            #print(type(frame_id_tmp))
                            #print(curr_ped_seq_tmp[1,:])
                            curr_ped_seq_tmp[frame_id_tmp,:] = curr_ped_seq[i,:]
                        pad_front = frames.index(curr_ped_seq_tmp[0, 0]) - idx
                        pad_end = frames.index(curr_ped_seq_tmp[-1, 0]) - idx + 1
                        #print(type(curr_ped_seq_tmp[:,2:]))
                        #print(curr_ped_seq_tmp[:,2:].shape)
                        #print(curr_ped_seq_tmp[:,2:])
                        curr_ped_seq = handle_nan(curr_ped_seq_tmp[:,2:])
                        #print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
                        #print(curr_ped_seq)
                        #print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
                        curr_ped_seq = np.transpose(curr_ped_seq)

                    #print(curr_ped_seq)
                    #print("finished")
                    #if(pad_end - pad_front == self.seq_len):
                        #curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    #print(curr_ped_seq.shape)
                    #curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq#if some pedestrians do not appear in all the first and end of the frames, they will not be added to the surr_seq
                    
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq 
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    #print(np.shape(curr_ped_seq))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1 #loss_mask was used to filter pedestrians who did not appear in the first and last frame
                    num_peds_considered += 1
                    #print(curr_loss_mask)


                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    #print(curr_seq[:num_peds_considered].shape) Nx2x16
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    #print(seq_list)
                    #print(len(seq_list))
                    #print(loss_mask_list)
                #print(len(non_linear_ped))
                #print(non_linear_ped)
        #print(len(seq_list))
        self.num_seq = len(seq_list)# 2930
        seq_list = np.concatenate(seq_list, axis=0)#(33866, 2, 16)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        # shape torch.Size([33866, 2, 8])
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float) # tensor([1., 0., 0.,  ..., 0., 1., 1.]), torch.Size([33866])
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:]) #len: 2930, [(0, 2), (2, 4), (4, 6),...,(33860, 33866)]
        ]
        
        #print(np.shape(self.obs_traj)) torch.Size([33866, 2, 8])
        

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :]
        ]
        #print(self.obs_traj)
        return out
    
    
    
class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t'
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len # frames, observe 8 frames and predict 8 frames
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))#for all files we have 4110 sequences(take hotel for example), but we have filtered some of them

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                #print(curr_seq_data.shape)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                 
                    if pad_end - pad_front != self.seq_len:
                        continue
                    #print(curr_ped_seq)
                    #print("finished")
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    #curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq#if some pedestrians do not appear in all the first and end of the frames, they will not be added to the surr_seq
                    
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq 
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    #print(np.shape(curr_ped_seq))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1 #loss_mask was used to filter pedestrians who did not appear in the first and last frame
                    num_peds_considered += 1
                    #print(curr_loss_mask)


                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    #print(curr_seq[:num_peds_considered].shape) Nx2x16
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    #print(seq_list)
                    #print(len(seq_list))
                    #print(loss_mask_list)
                #print(len(non_linear_ped))
                #print(non_linear_ped)
        #print(len(seq_list))
        self.num_seq = len(seq_list)# 2930
        seq_list = np.concatenate(seq_list, axis=0)#(33866, 2, 16)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        # shape torch.Size([33866, 2, 8])
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float) # tensor([1., 0., 0.,  ..., 0., 1., 1.]), torch.Size([33866])
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:]) #len: 2930, [(0, 2), (2, 4), (4, 6),...,(33860, 33866)]
        ]
        
        #print(np.shape(self.obs_traj)) torch.Size([33866, 2, 8])

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :]
        ]
        #print(self.obs_traj)
        return out
