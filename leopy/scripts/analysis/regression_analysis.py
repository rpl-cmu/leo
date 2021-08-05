#!/usr/bin/env python

#%%

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import glob
import math

import matplotlib.pyplot as plt

import sys
sys.path.append('../examples')
from regression import plot_energy_landscape, EnergyNetRFF

# %% helper functions

def compute_mse_loss(Enet, x_vec, y_vec):

    E_mat = torch.zeros((x_vec.shape[0], y_vec.shape[0]))
    for x_idx, x_val in enumerate(x_vec):
        for y_idx, y_val in enumerate(y_vec):
            E_mat[x_idx, y_idx] = torch.square(Enet(x_val.view(1,-1), y_val.view(1,-1)))

    min_val, min_ind = torch.min(E_mat, dim=1)
    y_pred = y_vec[min_ind]

    y_gt = x_vec * torch.sin(x_vec)

    loss = (y_pred - y_gt) ** 2

    return loss

def plot_Enet(Enet, x_vec):
    y_gt = x_vec * torch.sin(x_vec)
    plot_energy_landscape(x_vec, y_gt, Enet)
    plt.show()


#%% load different model files

plt.ion()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_printoptions(precision=2, sci_mode=False)

BASE_PATH = '~/code/fair_ws/dcem/'
model_type = 'leo_gn'

srcdir_model = f'{BASE_PATH}/local/regression/models/{model_type}/'
modelfiles = sorted(glob.glob(f"{srcdir_model}/*.pt"))

print(f"Modelfiles: {modelfiles}")

#%% mse loss across x for each model

n_res_x, n_res_y = 20, 100
x_vec = torch.linspace(0., 2.*math.pi, n_res_x)
y_vec = torch.linspace(-6., 6., n_res_y)

loss_mat = None
for midx, modelfile in enumerate(modelfiles):

    Enet = EnergyNetRFF(1, 1, 128, 1, 128)
    Enet.load_state_dict(torch.load(modelfile))
    Enet.eval()

    # debug
    # plot_Enet(Enet, x_vec)

    loss_vec = compute_mse_loss(Enet, x_vec, y_vec)
    loss_mat = torch.cat((loss_mat, loss_vec.view(1, -1)),
                         dim=0) if (loss_mat is not None) else loss_vec.view(1, -1)

#%% mean variance plots

mean = torch.mean(loss_mat, dim=0).detach().cpu().numpy()
std = torch.std(loss_mat, dim=0).detach().cpu().numpy()

scale = 1.
plt.plot(x_vec, mean, color='tab:orange', linewidth=2)
plt.fill_between(x_vec, mean - scale * std, mean + scale * std, color='tab:orange', alpha=0.2)

plt.ylim([-20, 80])
