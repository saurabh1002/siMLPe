import os
import glob
import numpy as np
from tqdm import tqdm

import sys
sys.path.append("/home/user/social_motion")
from social_motion.datasets.haggling import get_haggling_test_sequences

import torch
import torch.utils.data as data

class HagglingEvalDataset(data.Dataset):
    def __init__(self, config):
        super(HagglingEvalDataset, self).__init__()

        self.motion_input_length =  config.motion.input_length
        self.motion_target_length =  config.motion.target_length

        self._all_motion_poses = self._load_all()

    def __len__(self):
        return len(self._all_motion_poses)

    # seqs = N x 3 x 57
    def _load_all(self):
        sequences = get_haggling_test_sequences()

        all_motion_poses = []
        req_len_seq = self.motion_target_length + self.motion_input_length
        for sequence in sequences:
            poses_3person = sequence.Seq.reshape(-1, 3, 19, 3)
            N = len(poses_3person)
            if N < req_len_seq:
                continue

            all_motion_poses.append(poses_3person[:178, 0])
            all_motion_poses.append(poses_3person[:178, 1])
            all_motion_poses.append(poses_3person[:178, 2])
        return all_motion_poses

    def __getitem__(self, index):
        motion_poses = self._all_motion_poses[index]
        return motion_poses