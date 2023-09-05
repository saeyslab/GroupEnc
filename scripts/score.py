
import numpy as np
import os
from nxcurve import quality_curve

import argparse

parser = argparse.ArgumentParser(description='Score GroupEnc embedding')
parser.add_argument('-f', '--fpath_data',      type=str,                help='path to input (high-dim numpy array saved as .npy)')
parser.add_argument('-d', '--dataset',         type=str,                help='dataset name')
parser.add_argument('-r', '--fpath_res',       type=str,                help='path to results folder')
parser.add_argument('-k', '--k_group',         type=int,                help='group size (at least 3; 0 ~ use VAE instead)')
parser.add_argument('-e', '--epochs',          type=int,                help='number of training epochs')
parser.add_argument('-l', '--latent_dim',      type=int,                help='latent space dimensionality')
parser.add_argument('-i', '--run',             type=int,                help='index of embedding run to score')
args = parser.parse_args()

fpath_data = args.fpath_data
dataset    = args.dataset
fpath_res  = args.fpath_res
k          = args.k_group
e          = args.epochs
i          = args.run
l          = args.latent_dim

## Check if results exist already ----


id = f'{dataset}_k{k}l{l}e{e}i{i}'
skip = os.path.exists(f'{fpath_res}/{id}_curve.npy') and os.path.exists(f'{fpath_res}/{id}_auclog.npy')

if not skip:
        
    ## Load HD data ----
    hd = np.load(fpath_data, allow_pickle=True)

    ## Score ----
    print(f'Scoring {id}')
    proj = np.load(f'{fpath_res}/{id}_proj.npy', allow_pickle=True)
    s = quality_curve(X=hd, X_r=proj, n_neighbors=50, opt='r', graph=False)
    np.save(f'{fpath_res}/{id}_curve.npy', s[0], allow_pickle=True)
    np.save(f'{fpath_res}/{id}_auclog.npy', s[1], allow_pickle=True)

    print(f'Done scoring {id}')
else:
    print(f'Skipped scoring {id}')

## ----

