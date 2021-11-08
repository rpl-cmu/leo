#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

# Example adapted from: https://github.com/facebookresearch/dcem/blob/main/exps/regression.py

import copy

import numpy as np
import numpy.random as npr

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable

import higher
import csv
import os
import datetime

import pickle as pkl
import rff

from leopy.thirdparty.dcem import dcem

import hydra

import matplotlib.pyplot as plt
from matplotlib import cm, colorbar

from setproctitle import setproctitle
setproctitle('regression')

plt.ion()
plt.rcParams.update({'font.size': 20})

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CONFIG_PATH = os.path.join(BASE_PATH, "config/examples/regression.yaml")

@hydra.main(config_path=CONFIG_PATH)
def main(cfg):
    import sys
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(mode='Verbose',
        color_scheme='Linux', call_pdb=1)
    print('Current dir: ', os.getcwd())

    exp = RegressionExp(cfg)
    
    if (cfg.model.tag == 'leo_gn' or cfg.model.tag == 'leo_cem'):
        exp.run_leo()
    else:
        exp.run()

def to_np(tensor_arr):
    np_arr = tensor_arr.detach().cpu().numpy()
    return np_arr

def plot_energy_landscape(x_train, y_train, Enet=None, pred_model=None, ax=None, norm=True, show_cbar=False, y_samples=None):
    x = np.linspace(0., 2.*np.pi, num=500)
    y = np.linspace(-7., 7., num=500)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6,4))
    else:
        fig = ax.get_figure()
    
    # plt.axis('off')
    ax.plot(to_np(x_train), to_np(y_train), color='darkgreen', linewidth=2)

    if Enet is not None:
        X, Y = np.meshgrid(x, y)
        Xflat = torch.from_numpy(X.reshape(-1)).float().to(x_train.device).unsqueeze(1)
        Yflat = torch.from_numpy(Y.reshape(-1)).float().to(x_train.device).unsqueeze(1)
        Zflat = to_np(torch.square(Enet(Xflat, Yflat)))
        Z = Zflat.reshape(X.shape)
        
        if norm:
            Zmin = Z.min(axis=0)
            Zmax = Z.max(axis=0)
#             Zmax = np.quantile(Z, 0.75, axis=0)
            Zrange = Zmax-Zmin
            Z = (Z - np.expand_dims(Zmin, 0))/np.expand_dims(Zrange, 0)
            Z[Z > 1.0] = 1.0
            Z = np.log(Z+1e-6)
            Z = np.clip(Z, -10., 0.)
            CS = ax.contourf(X, Y, Z, cmap=cm.Blues, levels=10, vmin=-10., vmax=0., extend='min', alpha=0.8)
#             CS = ax.contourf(X, Y, Z, cmap=cm.Blues, levels=10, vmin=0., vmax=1., extend='max', alpha=0.8)
        else:
            CS = ax.contourf(X, Y, Z, cmap=cm.Blues, levels=10)

        if show_cbar:
            fig.colorbar(CS, ax=ax)
        
    if pred_model is not None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        ypreds = pred_model(x_train.unsqueeze(1)).squeeze()
        ax.plot(to_np(x_train), to_np(ypreds), color=colors[6], linestyle='--')

    if y_samples is not None:
        x_samples = x_train.repeat((y_samples.shape[0],1))
        ax.scatter(to_np(x_samples.flatten()), to_np(y_samples.flatten()), color='#e6ab02',s=2,alpha=0.25)

    ax.set_xlim(0, 2.*np.pi)
    ax.set_ylim(-7, 7)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
       
    return fig, ax, Z

class RegressionExp():
    def __init__(self, cfg):
        self.cfg = cfg

        self.exp_dir = os.getcwd()
        self.model_dir = os.path.join(self.exp_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)

        torch.manual_seed(cfg.seed)
        npr.seed(cfg.seed)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # self.Enet = EnergyNet(n_in=1, n_out=1, n_hidden=cfg.n_hidden).to(self.device)
        self.Enet = hydra.utils.instantiate(cfg.enet, n_in=1, n_out=1).to(self.device)
        self.model = hydra.utils.instantiate(cfg.model, self.Enet)

        self.load_data()

        self.fig, self.ax = plt.subplots(1, 1, figsize=(6,4))

    def dump(self, tag='latest'):
        fname = os.path.join(self.exp_dir, f'{tag}.pkl')
        pkl.dump(self, open(fname, 'wb'))

    def __getstate__(self):
        d = copy.copy(self.__dict__)
        del d['x_train']
        del d['y_train']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.load_data()

    def load_data(self):
        self.x_train = torch.linspace(0., 2.*np.pi, steps=self.cfg.n_samples).to(self.device)
        self.y_train = self.x_train*torch.sin(self.x_train)

    def run(self):
        # opt = optim.SGD(self.Enet.parameters(), lr=1e-1)
        opt = optim.Adam(self.Enet.parameters(), lr=1e-3)
        lr_sched = ReduceLROnPlateau(opt, 'min', patience=20, factor=0.5, verbose=True)

        fieldnames = ['iter', 'loss']
        f = open(os.path.join(self.exp_dir, 'loss.csv'), 'w')
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        directory = f"{BASE_PATH}/local/regression/plots/{self.cfg.model.tag}/{dt}/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.join(directory, "config.txt"), "w") as config_file:
            print(self.cfg.pretty(), file=config_file)
            config_file.close()

        model_directory = f"{BASE_PATH}/local/regression/models/{self.cfg.model.tag}/"
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        step = 0
        while step < self.cfg.n_update:
            if (step in list(range(100)) or step % 10000 == 0):
                self.dump(f'{step:07d}')

            j = npr.randint(self.cfg.n_samples)
            for i in range(self.cfg.n_inner_update):
                y_preds = self.model(self.x_train[j].view(1)).squeeze()
                loss = F.mse_loss(input=y_preds, target=self.y_train[j])
                opt.zero_grad()
                loss.backward(retain_graph=True)
                if self.cfg.clip_norm:
                    nn.utils.clip_grad_norm_(self.Enet.parameters(), 1.0)
                opt.step()

            if step % self.cfg.n_disp_step == 0:
                y_preds = self.model(self.x_train.view(-1, 1)).squeeze()
                loss = F.mse_loss(input=y_preds, target=self.y_train)
                lr_sched.step(loss)
                print(f'Iteration {step}: Loss {loss:.2f}')
                writer.writerow({
                    'iter': step,
                    'loss': loss.item(),
                })
                f.flush()
                exp_dir = os.getcwd()
                fieldnames = ['iter', 'loss', 'lr']
                self.dump('latest')
            
                plt.cla()
                plot_energy_landscape(self.x_train, self.y_train, Enet=self.Enet, ax=self.ax, show_cbar=False)

                if self.cfg.show_plot:
                    plt.show()
                    plt.pause(1e-2)

                if self.cfg.save_plot:
                    savepath = os.path.join(directory, f"{step:06d}.png")
                    plt.savefig(savepath)

                if self.cfg.save_model:
                    modelpath = os.path.join(model_directory, "model.pt")
                    # modelpath = os.path.join(model_directory, f"model_inner_iter_{self.cfg.model.params.n_inner_iter:03d}.pt")
                    print(f"Saving model to {modelpath}")
                    torch.save(self.Enet.state_dict(), modelpath)

            step += 1

    def run_leo(self):
        # opt = optim.SGD(self.Enet.parameters(), lr=1e-1)
        opt = optim.Adam(self.Enet.parameters(), lr=self.cfg.lr)
        lr_sched = ReduceLROnPlateau(opt, 'min', patience=20, factor=0.5, verbose=True)

        fieldnames = ['iter', 'loss']
        f = open(os.path.join(self.exp_dir, 'loss.csv'), 'w')
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        directory = f"{BASE_PATH}/local/regression/plots/{self.cfg.model.tag}/{dt}/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.join(directory, "config.txt"), "w") as config_file:
            print(self.cfg.pretty(), file=config_file)
            config_file.close()

        model_directory = f"{BASE_PATH}/local/regression/models/{self.cfg.model.tag}/"
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        step = 0
        while step < self.cfg.n_update:
            if (step in list(range(100)) or step % 10000 == 0):
                self.dump(f'{step:07d}')

            j = npr.randint(self.cfg.n_samples)
            for i in range(self.cfg.n_inner_update):

                y_mean, y_samples = self.model(self.x_train[j].view(1))
                n_samples = y_samples.shape[0]

                y_samples = y_samples.squeeze()
                y_mean = y_mean.squeeze()

                energy_samples = self.Enet((self.x_train[j].repeat(n_samples)).view(-1, 1), y_samples.view(-1, 1))
                energy_samples = torch.square(energy_samples)
                energy_samples = torch.mean(energy_samples)

                gt_samples = self.Enet(self.x_train[j].view(-1, 1), self.y_train[j].view(-1, 1))
                gt_samples = torch.square(gt_samples)
                loss = gt_samples - energy_samples

                opt.zero_grad()
                loss.backward(retain_graph=True)

                # Just for reference
                tracking_error = F.mse_loss(input=y_mean, target=self.y_train[j])

                if self.cfg.clip_norm:
                    nn.utils.clip_grad_norm_(self.Enet.parameters(), 1.0)
                opt.step()

            if step % self.cfg.n_disp_step == 0:
                y_mean, y_samples = self.model(self.x_train.view(-1, 1))
                y_mean = y_mean.squeeze()

                loss = torch.square(self.Enet(self.x_train.view(-1, 1), self.y_train.view(-1, 1))) - torch.square(self.Enet(
                    self.x_train.view(-1, 1), y_mean.view(-1, 1)))
                loss = torch.mean(loss)

                tracking_error = F.mse_loss(input=y_mean, target=self.y_train)

                lr_sched.step(loss)
                print(f'Iteration {step}: Loss {loss.item()}, Tracking Error: {tracking_error.item()}')
                writer.writerow({
                    'iter': step,
                    'loss': loss.item(),
                })
                f.flush()
                exp_dir = os.getcwd()
                fieldnames = ['iter', 'loss', 'lr']
                self.dump('latest')

                plt.cla()
                plot_energy_landscape(self.x_train, self.y_train,  Enet=self.Enet, ax=self.ax, show_cbar=False, y_samples=y_samples)

                if self.cfg.show_plot:
                    plt.show()
                    plt.pause(1e-2)

                if self.cfg.save_plot:
                    savepath = os.path.join(directory, f"{step:06d}.png")
                    plt.savefig(savepath)

                if self.cfg.save_model:
                    modelpath = os.path.join(model_directory, "model.pt")
                    # modelpath = os.path.join(model_directory, f"model_inner_iter_{self.cfg.model.params.n_inner_iter:03d}.pt")
                    print(f"Saving model to {modelpath}")
                    torch.save(self.Enet.state_dict(), modelpath)
            
            step += 1

class EnergyNetBasic(nn.Module):
    def __init__(self, n_in: int, n_out: int, n_hidden):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.E_net = nn.Sequential(
            nn.Linear(n_in+n_out, n_hidden),
            nn.Softplus(),
            nn.Linear(n_hidden, n_hidden),
            nn.Softplus(),
            nn.Linear(n_hidden, n_hidden),
            nn.Softplus(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x, y):
        x = x / (2.0*np.pi) # normalize
        y = (y + 7.0)/14.0
        z = torch.cat((x, y), dim=-1)
        E = self.E_net(z)
        return E

class EnergyNetRFF(nn.Module):
    def __init__(self, n_in: int, n_out: int, n_hidden, sigma, encoded_size):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.encoding = rff.layers.GaussianEncoding(sigma=sigma, input_size=n_in+n_out, encoded_size=encoded_size)
        encoding_out = encoded_size * (n_in+n_out)

        self.E_net = nn.Sequential(
            nn.Linear(encoding_out, n_hidden),
            nn.Softplus(),
            nn.Linear(n_hidden, n_hidden),
            nn.Softplus(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x, y):
        x = x / (2.0*np.pi)
        y = (y + 7.0)/14.0
        z = torch.cat((x, y), dim=-1)
        encoded = self.encoding(z)
        E = self.E_net(encoded)
        return E

class UnrollGD(nn.Module):
    def __init__(self, Enet, n_inner_iter, inner_lr, init_scheme='zero'):
        super().__init__()
        self.Enet = Enet
        self.n_inner_iter = n_inner_iter
        self.inner_lr = inner_lr
        self.init_scheme = init_scheme
        
        print("Instantiated UnrollGD class")

    def forward(self, x):
        b = x.ndimension() > 1
        if not b:
            x = x.unsqueeze(0)
        assert x.ndimension() == 2
        nbatch = x.size(0)

        y_init = torch.zeros(nbatch, self.Enet.n_out, device=x.device, requires_grad=True)
        if (self.init_scheme == 'gt'): y_init = (x*torch.sin(x)).clone()

        y = Variable(y_init.data, requires_grad=True)
        inner_opt = higher.get_diff_optim(
            torch.optim.SGD([y], lr=self.inner_lr),
            [y], device=x.device
        )

        for _ in range(self.n_inner_iter):
            E = self.Enet(x, y)
            E = torch.square(E)
            y, = inner_opt.step(E.sum(), params=[y])
        
        return y

class UnrollGN(nn.Module):
    ''' 1-D Gauss Newton '''
    def __init__(self, Enet, n_inner_iter, inner_lr, init_scheme='zero'):
        super().__init__()
        self.Enet = Enet
        self.n_inner_iter = n_inner_iter
        self.inner_lr = inner_lr
        self.init_scheme = init_scheme

        print("Instantiated UnrollGN class")

    def forward(self, x, y0=None):
        b = x.ndimension() > 1
        if not b:
            x = x.unsqueeze(0)
        assert x.ndimension() == 2
        nbatch = x.size(0)

        y_init = torch.zeros(nbatch, self.Enet.n_out, device=x.device, requires_grad=True)
        if (self.init_scheme == 'gt'): y_init = (x*torch.sin(x)).clone()

        y = [Variable(y_init.data, requires_grad=True)]
        # y = [-7+14*torch.rand(nbatch, self.Enet.n_out, device=x.device, requires_grad=True)]
        y[-1].retain_grad()

        for iter in range(self.n_inner_iter):
            y_tmp = torch.zeros(nbatch, self.Enet.n_out, device=x.device, requires_grad=True)
            y_new = y_tmp.clone()
            for b in range(x.shape[0]):
                yb = y[-1][b]
                E = self.Enet(x[b], yb)

                # Gauss-Newton
                grad_yb = torch.autograd.grad(E, yb, retain_graph=True, create_graph=True)
                J_inv = 1.0/(grad_yb[0][0])
                step = J_inv*E
                y_new[b] = yb - self.inner_lr * step

                # Newton's Method
                # grad_yb = torch.autograd.grad(torch.square(E), yb, retain_graph=True, create_graph=True)
                # print(H)
                # print(grad_yb)
                # step = (1.0/H[0][0]) * grad_yb[0][0]
                # print(step, gn_step)
                # y_new[b] = yb - self.inner_lr * step

            y.append(y_new)
            y[-1].retain_grad()

        return y[-1]

class UnrollCEM(nn.Module):
    def __init__(self, Enet, n_sample, n_elite,
                 n_iter, init_sigma, temp, normalize):
        super().__init__()
        self.Enet = Enet
        self.n_sample = n_sample
        self.n_elite = n_elite
        self.n_iter = n_iter
        self.init_sigma = init_sigma
        self.temp = temp
        self.normalize = normalize

        print("Instantiated UnrollCEM class")

    def forward(self, x):
        b = x.ndimension() > 1
        if not b:
            x = x.unsqueeze(0)
        assert x.ndimension() == 2
        nbatch = x.size(0)

        def f(y):
            _x = x.unsqueeze(1).repeat(1, y.size(1), 1)
            Es = self.Enet(_x.view(-1, 1), y.view(-1, 1)).view(y.size(0), y.size(1))
            Es = torch.square(Es)
            return Es

        yhat, ycov = dcem(
            f,
            n_batch=nbatch,
            nx=1,
            n_sample=self.n_sample,
            n_elite=self.n_elite,
            n_iter=self.n_iter,
            init_sigma=self.init_sigma,
            temp=self.temp,
            device=x.device,
            normalize=self.temp,
        )

        return yhat

class LEOGN(nn.Module):
    def __init__(self, Enet, n_sample, temp, min_cov, max_cov, n_inner_iter, init_scheme='zero'):
        super().__init__()
        self.Enet = Enet
        self.n_sample = n_sample
        self.temp = temp
        self.min_cov = min_cov
        self.max_cov = max_cov
        self.n_inner_iter = n_inner_iter

        self.init_scheme = init_scheme

        print("Instantiated LEOGN class")

    def gauss_newton(self, x, y_init):
        # Assuming nonlinear least squares min ||f||^2
        nbatch = y_init.size(0)
        ydim = y_init.size(1)

        # Initialize outputs
        yhat = y_init.clone()
        ycov = torch.zeros(nbatch, ydim, ydim, device=y_init.device, requires_grad=False)

        err_tol = 1e-7
        max_iter = self.n_inner_iter
        for b in range(yhat.size(0)):
            err_diff = 1e9
            alpha = 1.
            iter = 0

            def f(y):
                _x = x[b:b+1,...].unsqueeze(1).repeat(1, y.size(1), 1)
                Es = self.Enet(_x.view(-1, 1), y.view(-1, 1)).view(y.size(0), y.size(1))
                return Es

            while (iter < max_iter and err_diff > err_tol):
                e = f(yhat[b:b+1,...])
                e_tot = torch.sum(torch.square(e))

                jacobian = torch.autograd.functional.jacobian(f, yhat[b:b+1,...])
                # Diagonal trick from here: https://discuss.pytorch.org/t/jacobian-functional-api-batch-respecting-jacobian/84571/2
                jacobian = torch.diagonal(jacobian, dim1=0, dim2=2).permute(2, 0, 1)
                A = jacobian[0,...]
                AtA = torch.matmul(A.t(), A)
                augmented_H = AtA + alpha*torch.diag(torch.diag(AtA))
                d = torch.matmul(A.t(), e)
                ycov[b,...] = torch.inverse(AtA) # Not efficient in each iter
                dy = torch.matmul(torch.inverse(augmented_H), d)
                potential_y = yhat[b,...] - dy

                eb_new = f(potential_y.unsqueeze(0))
                eb_new_tot = torch.sum(torch.square(eb_new))
                # print(eb_new_tot, e_tot)
                if eb_new_tot < e_tot:
                    yhat[b,...] = potential_y
                    alpha /= 10

                    err_diff = abs(eb_new_tot - e_tot)
                else:
                    alpha *= 10

                iter += 1

        return yhat, ycov

    def forward(self, x):
        b = x.ndimension() > 1
        if not b:
            x = x.unsqueeze(0)
        assert x.ndimension() == 2
        nbatch = x.size(0)

        y_init = torch.zeros(nbatch, self.Enet.n_out, device=x.device, requires_grad=True)
        if (self.init_scheme == 'gt'): y_init = (x*torch.sin(x)).clone()

        yhat, ycov = self.gauss_newton(x, y_init)
        yhat = yhat.clone().detach().requires_grad_(True).flatten()
        ycov = ycov.clone().detach().requires_grad_(True).flatten()

        if self.temp > 0:
            ycov_clamped = torch.clamp(ycov/self.temp, self.min_cov, self.max_cov)
            ydist = torch.distributions.multivariate_normal.MultivariateNormal(yhat, torch.diag(ycov_clamped))
            ysamples = ydist.sample((self.n_sample,)) 
        else:
            ysamples = 20 * torch.rand(self.n_sample, nbatch) - 10 

        return yhat, ysamples

class LEOCEM(nn.Module):
    def __init__(self, Enet, n_sample, temp, min_cov, max_cov, cem_n_sample, cem_n_elite,
                 cem_n_iter, cem_init_sigma, cem_temp, cem_normalize):
        super().__init__()
        self.Enet = Enet
        self.n_sample = n_sample
        self.temp = temp
        self.min_cov = min_cov
        self.max_cov = max_cov
        self.cem_n_sample = cem_n_sample
        self.cem_n_elite = cem_n_elite
        self.cem_n_iter = cem_n_iter
        self.cem_init_sigma = cem_init_sigma
        self.cem_temp = cem_temp
        self.cem_normalize = cem_normalize

        print("Instantiated LEOCEM class")

    def forward(self, x):
        b = x.ndimension() > 1
        if not b:
            x = x.unsqueeze(0)
        assert x.ndimension() == 2
        nbatch = x.size(0)

        if self.temp <= 0:
            ysamples = 20 * torch.rand(self.n_sample, nbatch) - 10 
            yhat = torch.rand(1, nbatch)
            return yhat, ysamples


        def f(y):
            _x = x.unsqueeze(1).repeat(1, y.size(1), 1)
            Es = self.Enet(_x.view(-1, 1), y.view(-1, 1)).view(y.size(0), y.size(1))
            Es = torch.square(Es)
            return Es


        yhat, ycov = dcem(
            f,
            n_batch=nbatch,
            nx=1,
            n_sample=self.cem_n_sample,
            n_elite=self.cem_n_elite,
            n_iter=self.cem_n_iter,
            init_sigma=self.cem_init_sigma,
            temp=self.cem_temp,
            device=x.device,
            normalize=self.cem_normalize,
        )
        
        yhat = yhat.clone().detach().requires_grad_(True).flatten()
        ycov = ycov.clone().detach().requires_grad_(True).flatten()

        ycov_clamped = torch.clamp(ycov/self.temp, self.min_cov, self.max_cov)
        ydist = torch.distributions.multivariate_normal.MultivariateNormal(yhat, torch.diag(ycov_clamped))
        ysamples = ydist.sample((self.n_sample,)) 

        return yhat, ysamples


if __name__ == '__main__':
    import sys
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(mode='Verbose',
        color_scheme='Linux', call_pdb=1)
    main()
