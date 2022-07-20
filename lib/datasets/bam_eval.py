import os
import glob
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from utils.angle_to_joint import ang2joint

import torch
import torch.utils.data as data

class BAMEval(data.Dataset):
    def __init__(self, config, split_name, paired=True):
        super(BAMEval, self).__init__()
        self._split_name = split_name
        self._bam_anno_dir = config.bam_anno_dir
        self._root_dir = config.root_dir

        self._amass_file_names = self._get_bam_names()

        self.bam_motion_input_length =  config.motion.bam_input_length
        self.bam_motion_target_length =  config.motion.bam_target_length

        self.motion_dim = config.motion.dim
        self.shift_step = config.shift_step

        self._load_skeleton()
        self._collect_all()
        self._file_length = len(self.data_idx)

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._bam_file_names)

    def _get_bam_names(self):

        # create list
        seq_names = []
        assert self._split_name == 'test'

        seq_names += open(
            os.path.join(self._bam_anno_dir, "bam_test.txt"), 'r'
            ).readlines()

        file_list = []
        for dataset in seq_names:
            dataset = dataset.strip()
            subjects = glob.glob(self._bam_anno_dir + '/' + dataset + '/*')
            for subject in subjects:
                if os.path.isdir(subject):
                    files = glob.glob(subject + '/*poses.npz')
                    file_list.extend(files)
        #file_list = file_list[:10]
        return file_list

    def _load_skeleton(self):

        skeleton_info = np.load(
                os.path.join(self._root_dir, 'body_models', 'smpl_skeleton.npz')
                )
        self.p3d0 = torch.from_numpy(skeleton_info['p3d0']).float()
        parents = skeleton_info['parents']
        self.parent = {}
        for i in range(len(parents)):
            self.parent[i] = parents[i]

    def _collect_all(self):
        # Keep align with HisRep dataloader
        self.bam_seqs = []
        self.data_idx = []
        idx = 0
        for bam_seq_name in tqdm(self._bam_file_names):
            bam_info = np.load(bam_seq_name)
            bam_motion_poses = bam_info['poses'] # 156 joints(all joints of SMPL)
            N = len(bam_motion_poses)
            if N < self.bam_motion_target_length + self.bam_motion_input_length:
                continue

            frame_rate = bam_info['mocap_framerate']
            sample_rate = int(frame_rate // 25)
            sampled_index = np.arange(0, N, sample_rate)
            bam_motion_poses = bam_motion_poses[sampled_index]

            T = bam_motion_poses.shape[0]
            bam_motion_poses = R.from_rotvec(bam_motion_poses.reshape(-1, 3)).as_rotvec()
            bam_motion_poses = bam_motion_poses.reshape(T, 52, 3)
            bam_motion_poses[:, 0] = 0

            p3d0_tmp = self.p3d0.repeat([bam_motion_poses.shape[0], 1, 1])
            bam_motion_poses = ang2joint(p3d0_tmp, torch.tensor(bam_motion_poses).float(), self.parent)
            bam_motion_poses = bam_motion_poses.reshape(-1, 52, 3)[:, 4:22].reshape(T, 54)

            self.bam_seqs.append(bam_motion_poses)
            valid_frames = np.arange(0, T - self.bam_motion_input_length - self.bam_motion_target_length + 1, self.shift_step)

            self.data_idx.extend(zip([idx] * len(valid_frames), valid_frames.tolist()))
            idx += 1

    def __getitem__(self, index):
        idx, start_frame = self.data_idx[index]
        frame_indexes = np.arange(start_frame, start_frame + self.bam_motion_input_length + self.bam_motion_target_length)
        motion = self.bam_seqs[idx][frame_indexes]
        bam_motion_input = motion[:self.bam_motion_input_length]
        bam_motion_target = motion[self.bam_motion_input_length:]
        return bam_motion_input, bam_motion_target

