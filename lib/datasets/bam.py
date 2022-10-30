import os
import glob
import numpy as np
from tqdm import tqdm
from utils.bam_utils import unknown_pose_shape_to_known_shape, normalize

import torch
import torch.utils.data as data

class BAMDataset(data.Dataset):
    def __init__(self, config, split_name, data_aug=True):
        super(BAMDataset, self).__init__()
        self._split_name = split_name
        self._data_aug = data_aug
        self._root_dir = config.root_dir

        self._bam_anno_dir = config.bam_anno_dir

        self._bam_file_names = self._get_bam_names(config)
        self.bam_motion_input_length =  config.motion.bam_input_length
        self.bam_motion_target_length =  config.motion.bam_target_length

        self.frame_rate = config.frame_rate
        self.motion_dim = config.motion.dim
        self._all_bam_motion_poses = self._load_all()
        self._file_length = len(self._all_bam_motion_poses)

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._all_bam_motion_poses)

    def _get_bam_names(self, config):
        # create list
        seq_names = []
        if self._split_name == 'train' :
            seq_names = config.train_data
        else :
            seq_names = config.eval_data
        
        file_list = []
        for dataset in seq_names:
            files = glob.glob(self._bam_anno_dir + dataset + '/poses_pid*.npy')
            file_list.extend(files)
        return file_list

    def _preprocess(self, bam_motion_feats):
        if bam_motion_feats is None:
            return None
        bam_seq_len = bam_motion_feats.shape[0]

        if self.bam_motion_input_length + self.bam_motion_target_length < bam_seq_len:
            start = np.random.randint(bam_seq_len - self.bam_motion_input_length  - self.bam_motion_target_length + 1)
            end = start + self.bam_motion_input_length
        else:
            return None
        bam_motion_input = bam_motion_feats[start:end]

        bam_motion_target = bam_motion_feats[end:end+self.bam_motion_target_length]

        bam_motion = np.concatenate((bam_motion_input, bam_motion_target), axis=0)
        bam_motion = torch.from_numpy(normalize(bam_motion, self.bam_motion_input_length - 1)).reshape(bam_motion.shape[0], -1)
        return bam_motion

    def _load_all(self):
        all_bam_motion_poses = []
        for bam_motion_name in tqdm(self._bam_file_names):
            bam_motion_poses = unknown_pose_shape_to_known_shape(np.load(bam_motion_name)) # (N x 18 x 3)
            bam_motion_poses = bam_motion_poses[:, :17]
            N = len(bam_motion_poses)
            if N < self.bam_motion_target_length + self.bam_motion_input_length:
                continue
            
            frame_rate = self.frame_rate
            sample_rate = int(frame_rate // 25)
            sampled_index = np.arange(0, N, sample_rate)
            bam_motion_poses = bam_motion_poses[sampled_index]

            all_bam_motion_poses.append(bam_motion_poses)
            if self._split_name == "train":
                if (len(sampled_index) > self.bam_motion_target_length + self.bam_motion_input_length):
                    ids = np.arange(len(sampled_index) - self.bam_motion_target_length + self.bam_motion_input_length)
                    splits = np.random.choice(ids, len(ids) // 2, replace=False)
                    for split in splits:
                        all_bam_motion_poses.append(bam_motion_poses[split:])
        return all_bam_motion_poses

    def __getitem__(self, index):
        bam_motion_poses = self._all_bam_motion_poses[index]
        bam_motion = self._preprocess(bam_motion_poses)
        if bam_motion is None:
            while bam_motion is None:
                index = np.random.randint(self._file_length)
                bam_motion_poses = self._all_bam_motion_poses[index]
                bam_motion = self._preprocess(bam_motion_poses)

        if self._data_aug:
            if np.random.rand() > .5:
                idx = [i for i in range(bam_motion.size(0)-1, -1, -1)]
                idx = torch.LongTensor(idx)
                bam_motion = bam_motion[idx]

        bam_motion_input = bam_motion[:self.bam_motion_input_length].float()
        bam_motion_target = bam_motion[-self.bam_motion_target_length:].float()
        return bam_motion_input, bam_motion_target