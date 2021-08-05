#!/usr/bin/env python

import numpy as np
import os

import hydra
from datetime import datetime

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from lstm_utils import Nav2dFixDataset, Push2dDataset, Pose2dLSTM, LSTMPoseSeqNet

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
CONFIG_PATH = os.path.join(BASE_PATH, "python/config/baselines/lstm_net_train.yaml")

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

def data_loader(dataset_name, dataset_type, datatype, seq_len, params):

    dataset_dir_global = f"{BASE_PATH}/local/datasets/{dataset_name}/{datatype}"
    dataset_files = os.listdir(dataset_dir_global)

    datasets = []
    for filename in dataset_files:
        if dataset_type == "sim":
            datasets.append(Nav2dFixDataset(f"{dataset_dir_global}/{filename}", seq_len=seq_len, device=device))
        elif dataset_type == "real":
            datasets.append(Push2dDataset(f"{dataset_dir_global}/{filename}", seq_len=seq_len, device=device))

    dataset = ConcatDataset(datasets)    
    dataloader = DataLoader(dataset, batch_size=params.batch_size,
                            shuffle=params.shuffle, num_workers=params.num_workers)

    return dataloader, dataset, datasets[0].get_input_dim()

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, dataset_type, params):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.params = params

    # def setup(self, stage: Optional[str] = None):
    #     pass

    def get_input_size(self):
        # TODO: Hacky way to get input size
        _, _, input_size = data_loader(dataset_name=self.dataset_name, dataset_type=self.dataset_type, datatype="train", seq_len=1, params=self.params)
        return input_size

    def train_dataloader(self):
        seq_len = self.trainer.current_epoch + 2
        # seq_len = self.params.seq_len
        train_dataloader, train_dataset, input_size, = data_loader(dataset_name=self.dataset_name, dataset_type=self.dataset_type, datatype="train", seq_len=seq_len, params=self.params)
        return train_dataloader

    def val_dataloader(self):
        # seq_len = self.trainer.current_epoch + 2
        val_dataloader, _, _ = data_loader(dataset_name=self.dataset_name, dataset_type=self.dataset_type, datatype="test", seq_len=self.params.seq_len, params=self.params)
        return val_dataloader

    # def test_dataloader(self):
        # pass
        # return DataLoader(self.mnist_test, batch_size=self.batch_size)

    # def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        # ...

def train_model(cfg):
    
    # save config
    prefix = cfg.prefix + datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    print(cfg.pretty())
    with open("{0}/local/logs/{1}.txt".format(BASE_PATH, prefix), "w") as f:
        print(cfg.pretty(), file=f)
    
    # init tensorboard writer
    tb_writer = SummaryWriter("{0}/local/logs/{1}".format(BASE_PATH, prefix))
    
    # init trainer
    partitioned_string = cfg.dataset_name.partition('/')
    dataset_type = partitioned_string[0]
    checkpoint_callback = ModelCheckpoint(dirpath=f"{BASE_PATH}/local/checkpoints/{cfg.dataset_name}/",
                                          filename="{0}_{1}_{2}".format(prefix, cfg.network.model, "{epoch:02d}"))

    gpus = None if (device == "cpu") else 1 
    trainer = pl.Trainer(gpus=gpus, max_epochs=cfg.train.epochs, callbacks=[checkpoint_callback], reload_dataloaders_every_epoch=True)

    # init dataloaders
    # train_dataloader, train_dataset, input_size, = data_loader(dataset_name=cfg.dataset_name, dataset_type=dataset_type, datatype="train", params=cfg.dataloader)
    # val_dataloader, _, _ = data_loader(dataset_name=cfg.dataset_name, dataset_type=dataset_type, datatype="test", params=cfg.dataloader)
    dataloader = CustomDataModule(dataset_name=cfg.dataset_name, dataset_type=dataset_type, params=cfg.dataloader)

    # init model
    net = LSTMPoseSeqNet(cfg, dataloader.get_input_size(), tb_writer=tb_writer)

    # run training loop
    # trainer.fit(net, train_dataloader, val_dataloader)
    trainer.fit(net, datamodule=dataloader)

@hydra.main(config_path=CONFIG_PATH)
def main(cfg):

    train_model(cfg)

if __name__ == '__main__':
    main()
