## Paper: 
Bo Zhao, Nima Dehmamy, Robin Walters, Rose Yu. [Symmetry Teleportation for Accelerated Optimization](https://arxiv.org/abs/2205.10637). *Advances in Neural Information Processing Systems (NeurIPS)*, 2022.

## Abstract:
Existing gradient-based optimization methods update parameters locally, in a direction that minimizes the loss function. We study a different approach, symmetry teleportation, that allows parameters to travel a large distance on the loss level set, in order to improve the convergence speed in subsequent steps. Teleportation exploits symmetries in the loss landscape of optimization problems. We derive loss-invariant group actions for test functions in optimization and multi-layer neural networks, and prove a necessary condition for teleportation to improve convergence rate. We also show that our algorithm is closely related to second order methods. Experimentally, we show that teleportation improves the convergence speed of gradient descent and AdaGrad for several optimization problems including test functions, multi-layer regressions, and MNIST classification.


## Requirement 
* [PyTorch](https://pytorch.org/)
* [Matplotlib](https://matplotlib.org/)


## Optimizing with teleportation
Rosenbrock function:

```
python rosenbrock.py
```

Booth function:

```
python booth.py
```

Multi-layer neural network:

```
python multi_layer_regression.py
```

Training curves and visualizations are saved in the directory `figures`.

## Cite
```
@article{zhao2022symmetry,
  title={Symmetry Teleportation for Accelerated Optimization},
  author={Bo Zhao and Nima Dehmamy and Robin Walters and Rose Yu},
  journal={Neural Information Processing Systems},
  year = {2022}
}
```
