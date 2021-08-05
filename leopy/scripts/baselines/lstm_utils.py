#!/usr/bin/env python

import math
import json

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

import pytorch_lightning as pl

def wrap_to_pi(arr):
    arr_wrap = (arr + math.pi) % (2 * math.pi) - math.pi
    return arr_wrap

def vecToRot(v):
  return torch.tensor(([ [np.cos(v[2]), -np.sin(v[2]), 0], \
                         [np.sin(v[2]),  np.cos(v[2]), 0], \
                         [0,0,1]] ))

def vecToTransform(v):
  return torch.tensor(([ [np.cos(v[2]), -np.sin(v[2]), v[0]], \
                         [np.sin(v[2]),  np.cos(v[2]), v[1]], \
                         [0,0,1]] ))

def transformToVec(T):
  return torch.tensor(([ T[0,2], T[1,2], torch.atan2(T[1,0], T[0,0])]))

def read_file_json(filename, verbose=False):
    data = None
    with open(filename) as f:
        data = json.load(f)

    if verbose:
        print("Loaded file: {0}". format(filename))

    return data

class SequenceDataset(Dataset):
  def __init__(self, seq_len, device='cpu'):
    self.seq_len = seq_len

  def __len__(self):
    # Account for seq length
    return self.x.shape[0]-self.seq_len+1

  def __getitem__(self, idx):
    end_idx = idx+self.seq_len
    return self.x[idx:end_idx,:], self.y[idx:end_idx,:]

  def get_input_dim(self):
    return self.x.shape[1]
  
  def get_output_dim(self):
    return self.y.shape[1]

  def get_input_sequence(self):
    return self.x

  def get_output_sequence(self):
    return self.y[:self.x.shape[0], :]

class Nav2dFixDataset(SequenceDataset):
  def __init__(self, filename, seq_len, device='cpu'):
    super().__init__(seq_len)

    data = read_file_json(filename)

    # TODO: Check 1-off errors
    poses = torch.tensor(data["poses_gt"][1::], device=device)
    poses[:,2] = wrap_to_pi(poses[:,2])

    factors = data["factor_meas"]
    gps_meas = torch.tensor(factors[0:len(factors)-1:2], device=device)
    odom_meas = torch.tensor(factors[1::2], device=device)

    # T0_inv = torch.inverse(vecToTransform(poses[0]))
    # for i in range(len(poses)):
    #   p = poses[i,:]
    #   T = vecToTransform(p)
    #   T_new =  torch.matmul(T0_inv,T)
    #   poses[i,:] = transformToVec(T_new)
    # for i in range(len(gps_meas)):
    #   m = gps_meas[i,:]
    #   Tm = torch.matmul(T0_inv, vecToTransform(m))
    #   gps_meas[i,:] = transformToVec(Tm)

    self.x = torch.cat((gps_meas, odom_meas), dim=1)
    self.y = poses

    print(self.x.shape, self.y.shape)

class Push2dDataset(SequenceDataset):
  def __init__(self, filename, seq_len, device='cpu'):
    super().__init__(seq_len)

    data = read_file_json(filename)

    # TODO: Check 1-off errors
    poses = torch.tensor(data["obj_poses_gt"], device=device)
    poses[:,2] = wrap_to_pi(poses[:,2])

    ee_meas = torch.tensor(data["meas_ee_prior"], device=device)
    tactile_fts = torch.tensor(data["meas_tactile_img_feats"], device=device)

    T0_inv = torch.inverse(vecToTransform(poses[0]))
    for i in range(len(poses)):
      p = poses[i,:]
      T = vecToTransform(p)
      T_new =  torch.matmul(T0_inv,T)
      poses[i,:] = transformToVec(T_new)
    for i in range(len(ee_meas)):
      m = ee_meas[i,:]
      Tm = torch.matmul(T0_inv, vecToTransform(m))
      ee_meas[i,:] = transformToVec(Tm)

    self.x = torch.cat((ee_meas, tactile_fts), dim=1)
    self.y = poses

    print(self.x.shape, self.y.shape)


class Pose2dLSTM(nn.Module):
  def __init__(self, input_size, hidden_size_lstm, \
        num_layers_lstm, hidden_size_mlp):
    super(Pose2dLSTM, self).__init__()

    # Parameters
    self.hidden_size_lstm = hidden_size_lstm
    self.num_layers_lstm = num_layers_lstm
    self.hidden_size_mlp = hidden_size_mlp

    # Model layers
    self.lstm = torch.nn.LSTM(input_size, hidden_size_lstm, num_layers_lstm, batch_first=True)
    self.fc1 = nn.Linear(hidden_size_lstm, hidden_size_mlp)
    self.fc2 = nn.Linear(hidden_size_mlp, hidden_size_mlp)
    self.fc3 = nn.Linear(hidden_size_mlp, 4)

    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()

  def forward(self, x):
    batch_size = x.shape[0]

    # LSTM
    h0 = torch.zeros(self.num_layers_lstm, batch_size, self.hidden_size_lstm)
    c0 = torch.zeros(self.num_layers_lstm, batch_size, self.hidden_size_lstm)
    lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

    # MLP output
    tmp1 = self.relu(self.fc1(lstm_out))
    tmp2 = self.relu(self.fc2(tmp1))
    tmp3 = self.fc3(tmp2)
    out = torch.zeros((tmp3.shape[0], tmp3.shape[1], 3))
    out[:,:,:2] = tmp3[:,:,:2]
    # tmp4 = self.tanh(tmp3[:,:,2:])
    tmp4 = tmp3[:,:,2:]
    out[:,:,2] = torch.atan2(tmp4[:,:,0], tmp4[:,:,1])

    return out
    

class LSTMPoseSeqNet(pl.LightningModule):
    def custom_loss(self, output, target, w=100.0):
        t1 = output[:,:,:2]
        t2 = target[:,:,:2]
        r1 = output[:,:,2]
        r2 = target[:,:,2]
        loss_t = torch.sum((t1 - t2)**2)
        # loss_r = torch.sum(torch.minimum((r1 - r2)**2, (r1 - r2 - 2*np.pi)**2))
        loss_r = torch.sum(1.0 - torch.cos(r1-r2))
        loss = loss_t/t1.numel() + w*loss_r/r1.numel()
        return loss

    def __init__(self, params, input_size, tb_writer=None):
        super().__init__()

        # init config
        self.params = params
        self.learning_rate = self.params.train.learning_rate
        self.tb_writer = tb_writer

        # init model
        self.model = Pose2dLSTM(input_size=input_size, hidden_size_lstm=params.network.hidden_size_lstm,
                          num_layers_lstm=params.network.num_layers_lstm, hidden_size_mlp=params.network.hidden_size_mlp)

        # init loss
        self.loss = lambda x, y: self.custom_loss(x, y)
        # self.loss = lambda x, y: nn.MSELoss()(x, y)

        # For loading model?
        self.save_hyperparameters('params')
        self.save_hyperparameters('input_size')

    def forward(self, x, *args):
        return self.model(x, *args)

    def configure_optimizers(self):
        weight_decay = self.params.train.reg_weight if (self.params.train.reg_type == "l2") else 0.0
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):        
        meas, poses_gt = batch

        pred = self.forward(meas)
        actual = poses_gt

        loss = self.loss(pred, actual)

        return {'loss': loss}

    def training_epoch_end(self, train_step_outputs):
        avg_train_loss = torch.Tensor([x['loss'] for x in train_step_outputs]).mean()
        self.tb_writer.add_scalar("train/loss/epoch", avg_train_loss, self.trainer.current_epoch)

    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        return results

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.Tensor([x['loss'] for x in val_step_outputs]).mean()
        self.tb_writer.add_scalar("val/loss/epoch", avg_val_loss, self.trainer.current_epoch)

        self.log('val_loss', avg_val_loss)