import numpy as np
from typing import Union, Optional, Tuple, List
from ..core.node import Node, to_node, Variable
from ..core.ops import Operation
from .module import Parameter, Module

class BatchNormOp(Operation):
    def __init__(self, eps: float = 1e-5):
        self.eps = eps
        # store for backward
        self.x_centered = None
        self.std_inv = None
        self.x_hat = None
        self.batch_size = None
        self.spatial_dims = None

    def forward(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        # just simplify input shapes for now
        # x shape: (N, C) or (N, C, H, W)
        # gamma, beta shape: (C,)
        
        self.batch_size = x.shape[0]
        if x.ndim == 4:
            # Conv2d case: (N, C, H, W)
            # Normalise over (N, H, W)
            self.spatial_dims = (0, 2, 3)
            gamma_reshaped = gamma.reshape(1, -1, 1, 1)
            beta_reshaped = beta.reshape(1, -1, 1, 1)
        else:
            # Linear case: (N, C)
            self.spatial_dims = (0,)
            gamma_reshaped = gamma
            beta_reshaped = beta

        mean = x.mean(axis=self.spatial_dims, keepdims=True)
        var = x.var(axis=self.spatial_dims, keepdims=True)
        
        self.x_centered = x - mean
        self.std_inv = 1.0 / np.sqrt(var + self.eps)
        self.x_hat = self.x_centered * self.std_inv
        
        return gamma_reshaped * self.x_hat + beta_reshaped

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        # parents: [x, gamma, beta]
        gamma = node.parents[1].value
        
        if output_grad.ndim == 4:
            gamma_reshaped = gamma.reshape(1, -1, 1, 1)
            # m is N * H * W
            m = self.batch_size * output_grad.shape[2] * output_grad.shape[3]
            axis = (0, 2, 3)
        else:
            gamma_reshaped = gamma
            m = self.batch_size
            axis = (0,)

        d_beta = output_grad.sum(axis=axis)
        d_gamma = (output_grad * self.x_hat).sum(axis=axis)
        dx_hat = output_grad * gamma_reshaped
        
        # d_var = sum(dx_hat * (x - mean) * -0.5 * (var + eps)^-1.5)
        #       = sum(dx_hat * x_centered * -0.5 * std_inv^3)
        dvar = (dx_hat * self.x_centered * -0.5 * (self.std_inv**3)).sum(axis=axis, keepdims=True)
        
        # d_mean = sum(dx_hat * -std_inv) + dvar * sum(-2 * (x - mean)) / m
        # since sum(x - mean) = 0, d_mean = sum(dx_hat * -std_inv)
        dmean = (dx_hat * -self.std_inv).sum(axis=axis, keepdims=True)
        
        # dx = dx_hat * std_inv + dvar * 2 * (x - mean) / m + dmean / m
        dx = dx_hat * self.std_inv + dvar * 2 * self.x_centered / m + dmean / m
        
        return [dx, d_gamma, d_beta]

class BatchNorm(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = True
        
        self.gamma = Parameter(np.ones(num_features), name="bn_gamma")
        self.beta = Parameter(np.zeros(num_features), name="bn_beta")
        
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x: Node) -> Node:
        # x: (N, C) or (N, C, H, W)
        if self.training:
            op = BatchNormOp(self.eps)
            out_val = op.forward(x.value, self.gamma.value, self.beta.value)
            
            # Update running stats
            if x.value.ndim == 4:
                axis = (0, 2, 3)
            else:
                axis = (0,)
                
            batch_mean = x.value.mean(axis=axis)
            batch_var = x.value.var(axis=axis)
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            return Node(out_val, parents=[x, self.gamma, self.beta], op=op)
        else:
            # Inference mode
            if x.value.ndim == 4:
                rm = self.running_mean.reshape(1, -1, 1, 1)
                rv = self.running_var.reshape(1, -1, 1, 1)
                g = self.gamma.value.reshape(1, -1, 1, 1)
                b = self.beta.value.reshape(1, -1, 1, 1)
            else:
                rm = self.running_mean
                rv = self.running_var
                g = self.gamma.value
                b = self.beta.value
                
            out_val = g * (x.value - rm) / np.sqrt(rv + self.eps) + b
            return Node(out_val, parents=[], op=None) # Leaf node in inference usually

