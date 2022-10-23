import numpy as np
import torch
from torch import nn

def group_action_SO2(x, y, g, xy_to_uv, uv_to_xy):
    u, v = xy_to_uv(x, y)
    u_v = torch.einsum('ij,j->i', g, torch.stack((u, v)))
    u = u_v[0]
    v = u_v[1]
    return uv_to_xy(u, v)

def teleport_SO2(x, y, xy_to_uv, uv_to_xy, loss_func, lr_theta):
    theta = torch.tensor(np.random.random() * np.pi, requires_grad=True)
    for theta_step in range(10):
        g = torch.vstack(( \
            torch.cat(( (torch.cos(theta)).view(1), (-torch.sin(theta)).view(1) )), \
            torch.cat(( (torch.sin(theta)).view(1), (torch.cos(theta)).view(1))) \
            ))
        gx, gy = group_action_SO2(x, y, g, xy_to_uv, uv_to_xy)

        L = loss_func(gx, gy)
        dL_dgW = torch.autograd.grad(L, inputs=[gx, gy], create_graph=True)
        dL_dt = torch.square(dL_dgW[0]) + torch.square(dL_dgW[1])
        dLdt_dtheta = torch.autograd.grad(dL_dt, inputs=[theta])[0]

        theta = theta + lr_theta * dLdt_dtheta

    x = torch.tensor(gx.detach().numpy(), requires_grad=True)
    y = torch.tensor(gy.detach().numpy(), requires_grad=True)
    return x, y

def group_action_MLP(U, V, X, X_inv, T, sigma=nn.LeakyReLU(0.1), sigma_inv=nn.LeakyReLU(10)):
    # U = W_m, V = W_{m-1}, X = h_{m-2}
    k = list(T.size())[0]
    I = torch.eye(k)
    U_out = torch.matmul(U, (I-T))
    V_out = sigma(torch.matmul(V, X))
    V_out = torch.matmul((I+T), V_out)
    V_out = sigma_inv(V_out)
    V_out = torch.matmul(V_out, X_inv)
    return U_out, V_out

def teleport_MLP(W_list, X, Y, lr_teleport, dim, loss_func, step=10, sigma=nn.LeakyReLU(0.1)):
    X_inv = torch.linalg.pinv(X)

    for teleport_step in range(step):
        gW_list = W_list.copy()
        T = []
        h = X
        h_inv = X_inv
        for m in range(0, len(gW_list)-1):
            T.append(torch.zeros(dim[m+2], dim[m+2], requires_grad=True))
            gW_list[m+1], gW_list[m] = group_action_MLP(gW_list[m+1], gW_list[m], h, h_inv, T[m])
            h = sigma(torch.matmul(gW_list[m], h))
            h_inv = torch.linalg.pinv(h)

        L = loss_func(gW_list, X, Y)

        dL_dW_list = torch.autograd.grad(L, inputs=gW_list, create_graph=True)
        dL_dt = 0
        for i in range(len(gW_list)):
            dL_dt += torch.norm(dL_dW_list[i])**2 
        dLdt_dT_list = torch.autograd.grad(dL_dt, inputs=T)
        for i in range(len(T)):
            T[i] = T[i] + lr_teleport * dLdt_dT_list[i]

        h = X
        h_inv = X_inv
        h_inv_list = [h_inv]
        for m in range(0, len(W_list)-1):
            W_list[m+1], W_list[m] = group_action_MLP(W_list[m+1], W_list[m], h, h_inv, T[m])
            
            h = sigma(torch.matmul(gW_list[m], h))
            h_inv = torch.linalg.pinv(h)
            h_inv_list.append(h_inv)

    return W_list