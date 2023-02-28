import argparse
import os, sys
from scipy.spatial.transform import Rotation as R

import numpy as np
from sklearn import model_selection
from tqdm import tqdm
from config  import config
from model import siMLPe as Model
from utils.misc import rotmat2xyz_torch, rotmat2euler_torch
from datasets.bam import BAMDataset

import torch
from torch.utils.data import DataLoader

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for i in np.arange(N):
        for j in np.arange(N):
            w = np.sqrt(2 / N)
            if i == 0:
                w = np.sqrt(1 / N)
            dct_m[i, j] = w * np.cos(np.pi * (j + 1 / 2) * i / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

dct_m,idct_m = get_dct_matrix(config.motion.bam_input_length)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

def regress_pred(pbar, num_samples, m_p3d_h36):
    motion = []
    motion_gt_ = []
    for (motion_input, motion_target) in pbar:
        motion_input = motion_input.cuda()
        b, n, c = motion_input.shape
        b, t, c = motion_target.shape
        num_samples += b

        motion_input = motion_input.reshape(b, n, 18, 3)
        motion_target = motion_target.reshape(b, t, 18, 3)
        motion_gt_.append(np.concatenate((motion_input.cpu().detach().numpy(), motion_target.cpu().detach().numpy()), 1))
        motion_input = motion_input.reshape(b, n, -1)
        motion_target = motion_target.reshape(b, t, -1)
        outputs = []
        step = config.motion.bam_target_length_train
        if step == 25:
            num_step = 1
        else:
            num_step = 25 // step + 1
        for idx in range(num_step):
            with torch.no_grad():
                if config.deriv_input:
                    motion_input_ = motion_input.clone()
                    motion_input_ = torch.matmul(dct_m, motion_input_.cuda())
                    motion_input_ = motion_input_[:, -config.motion.bam_input_length:]
                else:
                    motion_input_ = motion_input.clone()
                output = model(motion_input_)
                output = torch.matmul(idct_m, output)[:, :step, :]
                if config.deriv_output:
                    output = output + motion_input[:, -1:, :].repeat(1, step, 1)

            output = output.reshape(-1, 18*3)
            output = output.reshape(b, step, -1)
            outputs.append(output)
            # motion_input = torch.cat([motion_input[:, step:], output], axis=1)

        motion_pred = torch.cat(outputs, axis=1)[:, :25]

        motion_target = motion_target.detach().reshape(b, t, 18, 3)
        motion_gt = motion_target.clone()

        motion_pred = motion_pred.detach().cpu()
        motion_pred = motion_pred.reshape(b, t, 18, 3)

        motion.append(np.concatenate((motion_input.reshape(b, n, 18, 3).cpu().detach().numpy(), motion_pred.numpy()), 1))

        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(motion_pred - motion_gt, dim=3), dim=2), dim=0)
        m_p3d_h36 += mpjpe_p3d_h36.cpu().numpy()
    motion_gt_ = np.concatenate(motion_gt_, 0)
    motion = np.concatenate(motion, 0)

    np.save("motion_pred.npy", motion)
    np.save("motion_gt.npy", motion_gt_)
    m_p3d_h36 = m_p3d_h36 / num_samples
    return m_p3d_h36

results_keys = ['#2', '#4', '#8', '#10', '#14', '#18', '#22', '#25']

def test(model, dataloader) :

    m_p3d_h36 = np.zeros([config.motion.bam_target_length])
    titles = np.array(range(config.motion.bam_target_length)) + 1
    num_samples = 0

    pbar = tqdm(dataloader)
    m_p3d_h36 = regress_pred(pbar, num_samples, m_p3d_h36)

    ret = {}
    for j in range(config.motion.bam_target_length):
        ret["#{:d}".format(titles[j])] = [m_p3d_h36[j], m_p3d_h36[j]]
    print([round(ret[key][0], 6) for key in results_keys])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model-pth', type=str, default=None, help='=encoder path')
    args = parser.parse_args()

    model = Model(config)

    state_dict = torch.load(args.model_pth)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()

    config.motion.bam_target_length = config.motion.bam_target_length_eval
    dataset = BAMDataset(config, 'test')

    shuffle = False
    sampler = None
    train_sampler = None
    dataloader = DataLoader(dataset, batch_size=2,
                            num_workers=1, drop_last=False,
                            sampler=sampler, shuffle=shuffle, pin_memory=True)

    test(model, dataloader)

