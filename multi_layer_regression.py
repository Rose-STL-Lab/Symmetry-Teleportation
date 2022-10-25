"""Applying symmetry teleportation to optimize a multi-layer neural network. """

import numpy as np
from matplotlib import pyplot as plt
import time
import torch
from torch import nn
import os

from teleportation import teleport_MLP

sigma = nn.LeakyReLU(0.1)
sigma_inv = nn.LeakyReLU(10)

# list of dimensions of weight matrices
# example: [4, 5, 6, 7, 8] -> X: 5x4, W1:6x5, W2:7x6, W3:8x7, Y:8x4
dim = [4, 5, 6, 7, 8] 

def init_param(dim, seed=12345):
    torch.manual_seed(seed)
    W_list = []
    for i in range(len(dim) - 2):
        W_list.append(torch.rand(dim[i+2], dim[i+1], requires_grad=True))
    X = torch.rand(dim[1], dim[0], requires_grad=True)
    Y = torch.rand(dim[-1], dim[0], requires_grad=True)
    return W_list, X, Y

def loss_multi_layer(W_list, X, Y):
    h = X
    for i in range(len(W_list)-1):
        h = sigma(torch.matmul(W_list[i], h))
    return 0.5 * torch.norm(Y - torch.matmul(W_list[-1], h)) ** 2

def train_epoch_SGD(W_list, X, Y, lr):
    L = loss_multi_layer(W_list, X, Y)
    dL_dW_list = torch.autograd.grad(L, inputs=W_list)
    W_list_updated = []
    dL_dt = 0
    for i in range(len(W_list)):
        W_list_updated.append(W_list[i] - lr * dL_dW_list[i])
        dL_dt += torch.norm(dL_dW_list[i])**2 
    return W_list_updated, L, dL_dt

def train_step_AdaGrad(W_list, X, Y, lr, G_list, eps=1e-10):
    L = loss_multi_layer(W_list, X, Y)
    dL_dW_list = torch.autograd.grad(L, inputs=W_list)
    W_list_updated = []
    dL_dt = 0
    for i in range(len(W_list)):
        G_list[i] = G_list[i] + dL_dW_list[i] * dL_dW_list[i]
        W_list_updated.append(W_list[i] - lr * torch.div(dL_dW_list[i], torch.sqrt(G_list[i]) + eps))
        dL_dt += torch.norm(dL_dW_list[i])**2 
    return W_list_updated, L, dL_dt, G_list


# do some random things first so that wall-clock time comparison is fair
for n in range(5):
    W_list, X, Y = init_param(dim, seed=n*n*12345)
    lr = 1e-4
    for epoch in range(300):
        W_list, loss, dL_dt = train_epoch_SGD(W_list, X, Y, lr)

# training with GD
print("SGD")
time_arr_SGD_n = []
loss_arr_SGD_n = []
dL_dt_arr_SGD_n = []
for n in range(5):
    W_list, X, Y = init_param(dim, seed=n*n*12345)
    lr = 1e-4
    time_arr_SGD = []
    loss_arr_SGD = []
    dL_dt_arr_SGD = []

    t0 = time.time()
    for epoch in range(300):
        W_list, loss, dL_dt = train_epoch_SGD(W_list, X, Y, lr)
        t1 = time.time()
        time_arr_SGD.append(t1 - t0)
        loss_arr_SGD.append(loss.detach().numpy())
        dL_dt_arr_SGD.append(dL_dt.detach().numpy())

    time_arr_SGD_n.append(time_arr_SGD)
    loss_arr_SGD_n.append(loss_arr_SGD)
    dL_dt_arr_SGD_n.append(dL_dt_arr_SGD)


# training with teleportation
print("SGD and teleportation")
time_arr_teleport_n = []
loss_arr_teleport_n = []
dL_dt_arr_teleport_n = []
for n in range(5):
    W_list, X, Y = init_param(dim, seed=n*n*12345)
    lr = 1e-4
    lr_teleport = 1e-7
    time_arr_teleport = []
    loss_arr_teleport = []
    dL_dt_arr_teleport = []

    t0 = time.time()
    for epoch in range(300):
        if epoch == 5:
            W_list = teleport_MLP(W_list, X, Y, lr_teleport, dim, loss_multi_layer, 8)

        W_list, loss, dL_dt = train_epoch_SGD(W_list, X, Y, lr)
        t1 = time.time()
        time_arr_teleport.append(t1 - t0)
        loss_arr_teleport.append(loss.detach().numpy())
        dL_dt_arr_teleport.append(dL_dt.detach().numpy())

    time_arr_teleport_n.append(time_arr_teleport)
    loss_arr_teleport_n.append(loss_arr_teleport)
    dL_dt_arr_teleport_n.append(dL_dt_arr_teleport)


# training with AdaGrad
print("AdaGrad")
time_arr_AdaGrad_n = []
loss_arr_AdaGrad_n = []
dL_dt_arr_AdaGrad_n = []
for n in range(5):
    W_list, X, Y = init_param(dim, seed=n*n*12345)
    lr = 1e-1
    time_arr_AdaGrad = []
    loss_arr_AdaGrad = []
    dL_dt_arr_AdaGrad = []

    G_list = []
    for i in range(len(W_list)):
        G_list.append(torch.zeros_like(W_list[i]))

    t0 = time.time()
    for epoch in range(300):
        W_list, loss, dL_dt, G_list = train_step_AdaGrad(W_list, X, Y, lr, G_list)
        t1 = time.time()
        time_arr_AdaGrad.append(t1 - t0)
        loss_arr_AdaGrad.append(loss.detach().numpy())
        dL_dt_arr_AdaGrad.append(dL_dt.detach().numpy())

    time_arr_AdaGrad_n.append(time_arr_AdaGrad)
    loss_arr_AdaGrad_n.append(loss_arr_AdaGrad)
    dL_dt_arr_AdaGrad_n.append(dL_dt_arr_AdaGrad)


# training with AdaGrad and teleportation
print("AdaGrad and teleportation")
time_arr_AdaGrad_teleport_n = []
loss_arr_AdaGrad_teleport_n = []
dL_dt_arr_AdaGrad_teleport_n = []
for n in range(5):
    W_list, X, Y = init_param(dim, seed=n*n*12345)
    lr = 1e-1
    lr_teleport = 1e-5
    time_arr_AdaGrad_teleport = []
    loss_arr_AdaGrad_teleport = []
    dL_dt_arr_AdaGrad_teleport = []

    G_list = []
    for i in range(len(W_list)):
        G_list.append(torch.zeros_like(W_list[i]))

    t0 = time.time()
    for epoch in range(300):
        if epoch == 5:
            W_list = teleport_MLP(W_list, X, Y, lr_teleport, dim, loss_multi_layer, 2)

        W_list, loss, dL_dt, G_list = train_step_AdaGrad(W_list, X, Y, lr, G_list)
        t1 = time.time()
        time_arr_AdaGrad_teleport.append(t1 - t0)
        loss_arr_AdaGrad_teleport.append(loss.detach().numpy())
        dL_dt_arr_AdaGrad_teleport.append(dL_dt.detach().numpy())

    time_arr_AdaGrad_teleport_n.append(time_arr_AdaGrad_teleport)
    loss_arr_AdaGrad_teleport_n.append(loss_arr_AdaGrad_teleport)
    dL_dt_arr_AdaGrad_teleport_n.append(dL_dt_arr_AdaGrad_teleport)

loss_arr_SGD_n = np.array(loss_arr_SGD_n)
dL_dt_arr_SGD_n = np.array(dL_dt_arr_SGD_n)
loss_arr_teleport_n = np.array(loss_arr_teleport_n)
dL_dt_arr_teleport_n = np.array(dL_dt_arr_teleport_n)
loss_arr_AdaGrad_n = np.array(loss_arr_AdaGrad_n)
dL_dt_arr_AdaGrad_n = np.array(dL_dt_arr_AdaGrad_n)
loss_arr_AdaGrad_teleport_n = np.array(loss_arr_AdaGrad_teleport_n)
dL_dt_arr_AdaGrad_teleport_n = np.array(dL_dt_arr_AdaGrad_teleport_n)

if not os.path.exists('figures'):
    os.mkdir('figures')
if not os.path.exists('figures/mlp'):
    os.mkdir('figures/mlp')

# plot loss vs epoch
plt.figure()
mean_SGD = np.mean(loss_arr_SGD_n, axis=0)
std_SGD = np.std(loss_arr_SGD_n, axis=0)
mean_SGD_teleport = np.mean(loss_arr_teleport_n, axis=0)
std_SGD_teleport = np.std(loss_arr_teleport_n, axis=0)
mean_AdaGrad = np.mean(loss_arr_AdaGrad_n, axis=0)
std_AdaGrad = np.std(loss_arr_AdaGrad_n, axis=0)
mean_AdaGrad_teleport = np.mean(loss_arr_AdaGrad_teleport_n, axis=0)
std_AdaGrad_teleport = np.std(loss_arr_AdaGrad_teleport_n, axis=0)

plt.plot(mean_SGD, linewidth=3, label='GD')
plt.plot(mean_SGD_teleport, linewidth=3, label='GD+teleport')
plt.plot(mean_AdaGrad, linewidth=3, label='AdaGrad')
plt.plot(mean_AdaGrad_teleport, linewidth=3, label='AdaGrad+teleport')
plt.gca().set_prop_cycle(None)
plt.fill_between(np.arange(300), mean_SGD-std_SGD, mean_SGD+std_SGD, alpha=0.5)
plt.fill_between(np.arange(300), mean_SGD_teleport-std_SGD_teleport, mean_SGD_teleport+std_SGD_teleport, alpha=0.5)
plt.fill_between(np.arange(300), mean_AdaGrad-std_AdaGrad, mean_AdaGrad+std_AdaGrad, alpha=0.5)
plt.fill_between(np.arange(300), mean_AdaGrad_teleport-std_AdaGrad_teleport, mean_AdaGrad_teleport+std_AdaGrad_teleport, alpha=0.5)
plt.xlabel('Epoch', fontsize=26)
plt.ylabel('Loss', fontsize=26)
plt.yscale('log')
plt.xticks([0, 100, 200, 300], fontsize= 20)
plt.yticks(fontsize= 20)
plt.legend(fontsize=17)
plt.savefig('figures/mlp/multi_layer_loss.pdf', bbox_inches='tight')

# plot loss vs wall-clock time
plt.figure()
n = 0
g_idx = np.arange(10) * 10 + 9
g_idx.astype(int)

plt.plot(np.mean(time_arr_SGD_n, axis=0), mean_SGD, linewidth=3, label='GD')
plt.plot(np.mean(time_arr_teleport_n, axis=0), mean_SGD_teleport, linewidth=3, label='GD+teleport')
plt.plot(np.mean(time_arr_AdaGrad_n, axis=0), mean_AdaGrad, linewidth=3, label='AdaGrad')
plt.plot(np.mean(time_arr_AdaGrad_teleport_n, axis=0), mean_AdaGrad_teleport, linewidth=3, label='AdaGrad+teleport')
plt.fill_between(np.mean(time_arr_SGD_n, axis=0), mean_SGD-std_SGD, mean_SGD+std_SGD, alpha=0.5)
plt.fill_between(np.mean(time_arr_teleport_n, axis=0), mean_SGD_teleport-std_SGD_teleport, mean_SGD_teleport+std_SGD_teleport, alpha=0.5)
plt.fill_between(np.mean(time_arr_AdaGrad_n, axis=0), mean_AdaGrad-std_AdaGrad, mean_AdaGrad+std_AdaGrad, alpha=0.5)
plt.fill_between(np.mean(time_arr_AdaGrad_teleport_n, axis=0), mean_AdaGrad_teleport-std_AdaGrad_teleport, mean_AdaGrad_teleport+std_AdaGrad_teleport, alpha=0.5)
plt.xlabel('time (s)', fontsize=26)
plt.ylabel('Loss', fontsize=26)
max_t = np.max(time_arr_teleport_n)
interval = np.round(max_t * 0.3, 2)
plt.xticks([0, interval, interval * 2, interval * 3], fontsize= 20)
plt.yticks(fontsize= 20)
plt.yscale('log')
plt.legend(fontsize=17)
plt.savefig('figures/mlp/multi_layer_loss_vs_time.pdf', bbox_inches='tight')

# plot dL/dt vs epoch
plt.figure()
plt.plot(dL_dt_arr_SGD, linewidth=3, label='GD')
plt.plot(dL_dt_arr_teleport, linewidth=3, label='GD+teleport')
plt.plot(dL_dt_arr_AdaGrad, linewidth=3, label='AdaGrad')
plt.plot(dL_dt_arr_AdaGrad_teleport, linewidth=3, label='AdaGrad+teleport')
plt.xlabel('Epoch', fontsize=26)
plt.ylabel('dL/dt', fontsize=26)
plt.yscale('log')
plt.xticks([0, 100, 200, 300], fontsize= 20)
plt.yticks([1e1, 1e3, 1e5, 1e7], fontsize= 20)
plt.legend(fontsize=17)
plt.savefig('figures/mlp/multi_layer_loss_gradient.pdf', bbox_inches='tight')


# plot loss vs dL/dt
plt.figure()
n = 0
g_idx = np.arange(10) * 10 + 9
g_idx.astype(int)
plt.plot(loss_arr_SGD_n[n], dL_dt_arr_SGD_n[n], linewidth=3, label='GD')
plt.plot(loss_arr_teleport_n[n], dL_dt_arr_teleport_n[n], linewidth=3, label='GD+teleport')
plt.plot(loss_arr_AdaGrad_n[n], dL_dt_arr_AdaGrad_n[n], linewidth=3, label='AdaGrad')
plt.plot(loss_arr_AdaGrad_teleport_n[n], dL_dt_arr_AdaGrad_teleport_n[n], linewidth=3, label='AdaGrad+teleport')
plt.xlabel('Loss', fontsize=26)
plt.ylabel('dL/dt', fontsize=26)
plt.yscale('log')
plt.xscale('log')
plt.xticks(fontsize= 20)
plt.yticks([1e1, 1e3, 1e5, 1e7], fontsize= 20)
plt.legend(fontsize=17)
plt.savefig('figures/mlp/multi_layer_loss_vs_gradient.pdf', bbox_inches='tight')
