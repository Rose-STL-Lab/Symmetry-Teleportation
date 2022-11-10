"""Group actions and teleportation algorithms for SO(2) and GL_n(R). """

import numpy as np
import torch
from torch import nn

def group_action_SO2(x, y, g, xy_to_uv, uv_to_xy):
    """SO(2) group action on a 2D vector.

    Args:
        x: A scalar. First element of the vector acted on.
        y: A scalar. Second element of the vector acted on.
        g: A 2x2 rotation matrix. Group element that acts on the vector.
        xy_to_uv: A function that takes input x, y and outputs u, v, where SO(2) acts 
          linearly on vector [u, v].
        uv_to_xy: A function that takes input u, v and outputs x, y. Inverse function
          xy_to_uv.

    Returns:
        Two numbers corresponding to g.[x,y], where . denotes the action map.
    """
    u, v = xy_to_uv(x, y)
    u_v = torch.einsum('ij,j->i', g, torch.stack((u, v)))
    u = u_v[0]
    v = u_v[1]
    return uv_to_xy(u, v)


def teleport_SO2(x, y, xy_to_uv, uv_to_xy, loss_func, lr_theta):
    """Teleportation on a function with SO(2) symmetry.

    Args:
        x: A scalar. First element of the vector acted on.
        y: A scalar. Second element of the vector acted on.
        xy_to_uv: A function that takes input x, y and outputs u, v, where SO(2) acts 
          linearly on vector [u, v].
        uv_to_xy: A function that takes input u, v and outputs x, y. Inverse function
          xy_to_uv.
        loss_func: Loss function in the optimization problem.
        lr_theta: A scalar. Learning rate used in optimizing the group element.

    Returns:
        Two numbers corresponding to g.[x,y], where g is the optimal group element found.
    """
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


def group_action_MLP_two_layer(U, V, X, X_inv, T, sigma=nn.LeakyReLU(0.1), sigma_inv=nn.LeakyReLU(10)):
    """GL(R) group actions on a pair of matrices.

    Performs the group action in equation (8) in https://arxiv.org/pdf/2205.10637.pdf.
    U = W_m, V = W_{m-1}, X = h_{m-2}

    Args:
        U: Matrix with dimension m x k. Weight acting on sigma(VX)
        V: Matrix with dimension k x n. Weight acting on X.
        X: Matrix with dimension m x n. Output from the previous layer.
        X_inv: Matrix with dimension n x m. Inverse of X.
        T: Matrix with dimension k x k. Element in the Lie algebra of GL_k(R)
        sigma: Element-wise activation function.
        sigma_inv: Inverse of sigma.

    Returns:
        U_out: Result of g acting on U. Same dimension as U. 
        V_out: Result of g acting on V. Same dimension as V. 
    """
    k = list(T.size())[0]
    I = torch.eye(k)
    U_out = torch.matmul(U, (I-T))
    V_out = sigma(torch.matmul(V, X))
    V_out = torch.matmul((I+T), V_out)
    V_out = sigma_inv(V_out)
    V_out = torch.matmul(V_out, X_inv)
    return U_out, V_out


def group_action_MLP(W_list, X, X_inv, T, sigma=nn.LeakyReLU(0.1), sigma_inv=nn.LeakyReLU(10)):
    """ GL(R) group actions on all layers in an MLP.

    Args:
        W_list: list of weight matrices.
        X: Data matrix, with dimension a x b. 
        X_inv: Matrix with dimension n x m. Inverse of X.
        T: list of Lie algebra elements used to transform the weight matrices

    Returns:
        W_list: Teleported weights. Same shapes as the input W_list.
    """
    gW_list = W_list.copy()
    h = X
    h_inv = X_inv
    h_inv_list = [h_inv]
    for m in range(0, len(W_list)-1):
        W_list[m+1], W_list[m] = group_action_MLP_two_layer(W_list[m+1], W_list[m], h, h_inv, T[m])
        
        h = sigma(torch.matmul(gW_list[m], h))
        h_inv = torch.linalg.pinv(h)
        h_inv_list.append(h_inv)

    return W_list


def teleport_MLP(W_list, X, Y, lr_teleport, dim, loss_func, step=10, sigma=nn.LeakyReLU(0.1)):
    """Teleportation on weight matrices in a multi-layer neural network, using gradient ascent.

    Args:
        W_list: list of weight matrices.
        X: Data matrix, with dimension a x b. 
        Y: Label matrix, with dimension c x b.
        lr_teleport: A scalar. Learning rate used in optimizing the group element.
        dim: list of dimensions of weight matrices. Example: [4, 5, 6, 7, 8] -> 
          X: 5x4, W1:6x5, W2:7x6, W3:8x7, Y:8x4
        loss_func: Loss function in the optimization problem.
        step: An integer. Number of gradient ascent steps used to optimize the group
          element.
        sigma: Element-wise activation function.

    Returns:
        W_list: Teleported weights. Same shapes as the input W_list.
    """
    X_inv = torch.linalg.pinv(X)

    for teleport_step in range(step):
        # populate gW_list with (I+T).W, where T=0
        gW_list = W_list.copy()
        T = []
        h = X
        h_inv = X_inv
        for m in range(0, len(gW_list)-1):
            T.append(torch.zeros(dim[m+2], dim[m+2], requires_grad=True))
            gW_list[m+1], gW_list[m] = group_action_MLP_two_layer(gW_list[m+1], gW_list[m], h, h_inv, T[m])
            h = sigma(torch.matmul(gW_list[m], h))
            h_inv = torch.linalg.pinv(h)

        # compute L(T.W) and dL/d(T.W)
        L = loss_func(gW_list, X, Y)
        dL_dW_list = torch.autograd.grad(L, inputs=gW_list, create_graph=True)

        # compute dL/dt=||dL/d(T.W)||^2 and d/dT dL/dt
        dL_dt = 0
        for i in range(len(gW_list)):
            dL_dt += torch.norm(dL_dW_list[i])**2 
        dLdt_dT_list = torch.autograd.grad(dL_dt, inputs=T)

        # gradient ascent step on T, in the direction of d/dT dL/dt
        for i in range(len(T)):
            T[i] = T[i] + lr_teleport * dLdt_dT_list[i]

        # replace original W's with T.W, using the new T's
        W_list = group_action_MLP(W_list, X, X_inv, T)

    return W_list