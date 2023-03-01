import os
import glob
import numpy as np
from tqdm import tqdm
from utils.haggling_utils import normalize

import sys
sys.path.append("/home/user/social_motion")
from social_motion.datasets.haggling import get_haggling_train_sequences

import torch
import torch.utils.data as data

class HagglingDataset(data.Dataset):
    def __init__(self, config, data_aug=True):
        super(HagglingDataset, self).__init__()

        self._data_aug = data_aug
        self._root_dir = config.root_dir

        self.motion_input_length =  config.motion.input_length
        self.motion_target_length =  config.motion.target_length

        self.frame_rate = config.frame_rate
        self.motion_dim = config.motion.dim
        self._all_motion_poses = self._load_all()

    def __len__(self):
        return len(self._all_motion_poses)

    def _preprocess(self, motion_feats):
        bam_motion = torch.from_numpy(normalize(motion_feats, self.motion_input_length - 1)).reshape(motion_feats.shape[0], -1)
        return bam_motion

    # seqs = N x 3 x 57
    def _load_all(self):
        sequences = get_haggling_train_sequences()

        all_motion_poses = []
        req_len_seq = self.motion_target_length + self.motion_input_length
        for sequence in sequences:
            poses_3person = sequence.Seq.reshape(-1, 3, 19, 3)
            N = len(poses_3person)
            if N < req_len_seq:
                continue
            
            frame_rate = self.frame_rate
            sample_rate = int(frame_rate // 25)
            sampled_index = np.arange(0, N, sample_rate)
            poses_3person = poses_3person[sampled_index]

            if (len(sampled_index) > req_len_seq):
                ids = np.arange(0, len(sampled_index) - req_len_seq, req_len_seq)
                for i in ids:
                    all_motion_poses.append(poses_3person[i:i+req_len_seq, 0])
                    all_motion_poses.append(poses_3person[i:i+req_len_seq, 1])
                    all_motion_poses.append(poses_3person[i:i+req_len_seq, 2])
        
        return all_motion_poses

    def __getitem__(self, index):
        motion_poses = self._all_motion_poses[index]
        motion = self._preprocess(motion_poses)
        if motion is None:
            while motion is None:
                index = np.random.randint(self.__len__())
                motion_poses = self._all_motion_poses[index]
                motion = torch.from_numpy(motion_poses)

        if self._data_aug:
            if np.random.rand() > .5:
                idx = [i for i in range(motion.size(0)-1, -1, -1)]
                idx = torch.LongTensor(idx)
                motion = motion[idx]

        motion_input = motion[:self.motion_input_length].float()
        motion_target = motion[-self.motion_target_length:].float()
        return motion_input, motion_target