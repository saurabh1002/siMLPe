import os, sys

import numpy as np
from tqdm import tqdm
from config  import config
from model import siMLPe as Model

from datasets.haggling_eval import HagglingEvalDataset
from utils.haggling_utils import normalize, undo_normalization_to_seq

import torch

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


if __name__ == "__main__":
    model = Model(config)

    state_dict = torch.load("log-haggling/snapshot-haggling/model-iter-115000.pth")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()

    dct_m,idct_m = get_dct_matrix(config.motion.input_length)
    dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
    idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

    config.motion.target_length = config.motion.target_length_eval
    sequences_in = HagglingEvalDataset(config)

    sequences_out = []

    n_seqs = len(sequences_in) // 3

    for i in tqdm(range(n_seqs)):
        sequence_out_3_persons = []
        for p in range(3):
            sequence = sequences_in[(3 * i) + p]   # 178 x 19 x 3
            normalized_seq, normalization_params = normalize(sequence, -1, return_transform=True)
            for n_iter in range(1870 // config.motion.target_length + 1):
                seq_in_torch = torch.from_numpy(normalized_seq[-config.motion.input_length:].reshape(config.motion.input_length, -1)).unsqueeze(0).cuda()
                seq_out_torch = (torch.matmul(idct_m, model(torch.matmul(dct_m, seq_in_torch))) + seq_in_torch[:, -1])[0].cpu().detach()[:config.motion.target_length]
                seq_out = undo_normalization_to_seq(seq_out_torch.numpy().reshape(-1, 19, 3), normalization_params[0], normalization_params[1])
                sequence = np.concatenate((sequence, seq_out), 0)
                normalized_seq, normalization_params = normalize(sequence, -1, return_transform=True)
            sequence_out_3_persons.append(sequence)
        sequences_out.append(np.stack(sequence_out_3_persons, 1))

    sequences_out = np.array(sequences_out)[:, :2048]
    print(sequences_out.shape)
    np.save("haggling_eval_simlpe.npy", sequences_out)

