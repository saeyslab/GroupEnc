
import os
import numpy as np
import time
import sys

import argparse

parser = argparse.ArgumentParser(description='Create GroupEnc embedding')
parser.add_argument('-f', '--fpath_data',      type=str,                help='path to input (high-dim numpy array saved as .npy)')
parser.add_argument('-d', '--dataset',         type=str,                help='dataset name')
parser.add_argument('-r', '--fpath_res',       type=str,                help='path to results folder')
parser.add_argument('-k', '--k_group',         type=int,                help='group size (at least 3; 0 ~ use VAE instead)')
parser.add_argument('-e', '--epochs',          type=int,                help='number of training epochs')
parser.add_argument('-l', '--latent_dim',      type=int,                help='latent space dimensionality')
parser.add_argument('-i', '--run',             type=int,                help='index of embedding run to score')
parser.add_argument('-s', '--fpath_groupenc',  type=str,                help='path to the GroupEnc package')
args = parser.parse_args()

sys.path[0] = args.fpath_groupenc
from GroupEnc import VAE
from GroupEnc import GroupEnc

fpath_data = args.fpath_data
dataset    = args.dataset
fpath_res  = args.fpath_res
k          = args.k_group
e          = args.epochs
i          = args.run
l          = args.latent_dim

## Check if results exist already ----

id = f'{dataset}_k{k}l{l}e{e}i{i}'
skip = os.path.exists(f'{fpath_res}/{id}_proj.npy') and os.path.exists(f'{fpath_res}/{id}_time.npy')

if not skip:
        
    ## Load HD data ----
    hd = np.load(fpath_data, allow_pickle=True)

    ## Make embedding ----
    print(f'Embedding {id}')
    if k==0:
        model = VAE(full_dim=hd.shape[1], latent_dim=l)
    else:
        model = GroupEnc(full_dim=hd.shape[1], latent_dim=l)
    start_time = time.time()
    model.fit(
        X=hd,
        n_epochs=e,
        seed=i
    )
    t = time.time() - start_time
    proj = model.transform(hd)
    np.save(f'{fpath_res}/{id}_proj.npy', proj)
    np.save(f'{fpath_res}/{id}_time.npy', t)

    print(f'Done embedding {id}')
else:
    print(f'Skipped embedding {id}')

## ----
