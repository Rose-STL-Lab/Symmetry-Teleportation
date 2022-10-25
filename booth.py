"""Applying symmetry teleportation to optimize the Booth function. """

import numpy as np
from matplotlib import pyplot as plt
import torch
import os

from teleportation import teleport_SO2


def booth(x, y):
    return (x + 2*y - 7)**2 + (2*x + y - 5)**2

def xy_to_uv(x, y):
    return x + 2*y - 7, 2*x + y - 5

def uv_to_xy(u, v):
    return -1/3*u + 2/3*v + 1, 2/3*u - 1/3*v + 3

def plot_level_sets():
    L = np.array([1e0, np.sqrt(10)*1e0, 1e1, np.sqrt(10)*1e1, 1e2, np.sqrt(10)*1e2, 1e3, np.sqrt(10)*1e3, 1e4] )
    t = np.linspace(0, 2*np.pi, 1000)
    for loss in L:
        u = np.sqrt(loss) * np.cos(t)
        v = np.sqrt(loss) * np.sin(t)
        x, y = uv_to_xy(u, v)
        plt.plot(x, y, color='gray')
    plt.xlim(-10.0, 10.0)
    plt.ylim(-10.0, 10.0)

def train_epoch_SGD(x, y, lr):
    L = booth(x, y)
    dL_dx, dL_dy = torch.autograd.grad(L, inputs=[x, y])
    x_updated = x - lr * dL_dx
    y_updated = y - lr * dL_dy
    dL_dt = dL_dx**2 + dL_dy**2
    return x_updated, y_updated, L, dL_dt


x0, y0 = 5.0, -5.0 # initialization of parameters
lr = 0.08

# gradient descent with no teleportation
x = torch.tensor(x0, requires_grad=True)
y = torch.tensor(y0, requires_grad=True)
x_arr = [x0] # save trajectory for plotting
y_arr = [y0]
loss_arr_SGD = []
dL_dt_arr_SGD = []
for epoch in range(10):
    x, y, loss, dL_dt = train_epoch_SGD(x, y, lr)
    x_arr.append(x.detach().numpy())
    y_arr.append(y.detach().numpy())
    loss_arr_SGD.append(loss.detach().numpy())
    dL_dt_arr_SGD.append(dL_dt.detach().numpy())

# gradient descent with teleportation
lr_theta = 0.001 # for gradient ascent on theta (optimizing for the group element to improve dL/dt)
x = torch.tensor(x0, requires_grad=True)
y = torch.tensor(y0, requires_grad=True)
x_arr_teleport = [x0]
y_arr_teleport = [y0]
loss_arr_teleport = []
dL_dt_arr_teleport = []

for epoch in range(10):
    if epoch == 4:
        x, y = teleport_SO2(x, y, xy_to_uv, uv_to_xy, booth, lr_theta)

    x, y, loss, dL_dt = train_epoch_SGD(x, y, lr)
    x_arr_teleport.append(x.detach().numpy())
    y_arr_teleport.append(y.detach().numpy())
    loss_arr_teleport.append(loss.detach().numpy())
    dL_dt_arr_teleport.append(dL_dt.detach().numpy())


# make figures
if not os.path.exists('figures'):
    os.mkdir('figures')
if not os.path.exists('figures/booth'):
    os.mkdir('figures/booth')

# visualization of GD without teleportation
plt.figure()
plot_level_sets()
plt.scatter(x_arr, y_arr)
plt.plot(x_arr, y_arr)
plt.scatter(x_arr[-1], y_arr[-1], marker="*", s=60, color='#1f77b4')
plt.scatter(1, 3, marker="*", s=40, color='#2ca02c')
plt.savefig('figures/booth/Booth_level_set_GD.pdf', bbox_inches='tight')

# visualization of GD with teleportation
x_arr_teleport = np.array(x_arr_teleport)
y_arr_teleport = np.array(y_arr_teleport)
loss_arr_teleport = np.array(loss_arr_teleport)
dL_dt_arr_teleport = np.array(dL_dt_arr_teleport)
loss_arr_SGD = np.array(loss_arr_SGD)
dL_dt_arr_SGD = np.array(dL_dt_arr_SGD)

plt.figure()
plot_level_sets()
plt.scatter(x_arr_teleport, y_arr_teleport, s=20)
plt.plot(x_arr_teleport, y_arr_teleport)
g_idx = np.arange(1) * 5 + 4
plt.scatter(x_arr_teleport[g_idx], y_arr_teleport[g_idx], s=20) # orange dots
for idx in g_idx:
    plt.plot(x_arr_teleport[idx:idx+2], y_arr_teleport[idx:idx+2], color='#ff7f0e') # orange line
plt.scatter(x_arr_teleport[-1], y_arr_teleport[-1], marker="*", s=60, color='#1f77b4')   # initial point (blue dot)
plt.scatter(1, 3, marker="*", s=40, color='#2ca02c') # target point (green star)
plt.savefig('figures/booth/Booth_level_set_teleport.pdf', bbox_inches='tight')

# plot loss vs epoch
plt.figure()
plt.plot(loss_arr_SGD, label='GD', linewidth=3.5, zorder=3)
plt.plot(loss_arr_teleport, label='GD with teleport', linewidth=3.5, zorder=2)
plt.xlabel('Epoch', fontsize=26)
plt.ylabel('Loss', fontsize=26)
plt.yscale('log')
plt.xticks(fontsize= 20)
plt.yticks(fontsize= 20)
plt.legend(fontsize=17)
plt.savefig('figures/booth/Booth_loss.pdf', bbox_inches='tight')

# plot dL/dt vs epoch
plt.figure()
plt.plot(dL_dt_arr_SGD, label='GD', linewidth=3.5, zorder=3)
plt.plot(dL_dt_arr_teleport, label='GD with teleport', linewidth=3.5, zorder=2)
plt.xlabel('Epoch', fontsize=26)
plt.ylabel('dL/dt', fontsize=26)
plt.yscale('log')
plt.xticks(fontsize= 20)
plt.yticks(fontsize= 20)
plt.legend(fontsize=17)
plt.savefig('figures/booth/Booth_loss_gradient.pdf', bbox_inches='tight')

# plot loss vs dL/dt
plt.figure()
plt.scatter(loss_arr_SGD[g_idx], dL_dt_arr_SGD[g_idx], s=60)
plt.scatter(loss_arr_teleport[g_idx], dL_dt_arr_teleport[g_idx], s=60, label='teleportation point')
plt.plot(loss_arr_SGD[0:], dL_dt_arr_SGD[0:], label='GD', linewidth=3.5, zorder=3)
plt.plot(loss_arr_teleport[0:], dL_dt_arr_teleport[0:], label='GD with teleport', linewidth=3.5, zorder=2)
plt.xlabel('Loss', fontsize=26)
plt.ylabel('dL/dt', fontsize=26)
plt.yscale('log')
plt.xscale('log')
plt.xticks(fontsize= 20)
plt.yticks(fontsize= 20)
plt.legend(fontsize=17)
plt.savefig('figures/booth/Booth_loss_vs_gradient.pdf', bbox_inches='tight')
