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
        # Generalize to any dimensionality N >= 2
        # Normalise over all dimensions except the second (index 1, channel/feature dim)
        self.stats_axes = (0,) + tuple(range(2, x.ndim))
        self.batch_size = x.shape[0]
        
        # Reshape gamma and beta for broadcasting: (1, C, 1, 1, ...)
        reshape_shape = [1] * x.ndim
        reshape_shape[1] = -1
        gamma_reshaped = gamma.reshape(*reshape_shape)
        beta_reshaped = beta.reshape(*reshape_shape)

        mean = x.mean(axis=self.stats_axes, keepdims=True)
        var = x.var(axis=self.stats_axes, keepdims=True)
        
        self.x_centered = x - mean
        self.std_inv = 1.0 / np.sqrt(var + self.eps)
        self.x_hat = self.x_centered * self.std_inv
        
        return gamma_reshaped * self.x_hat + beta_reshaped

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        # parents: [x, gamma, beta]
        gamma = node.parents[1].value
        
        reshape_shape = [1] * output_grad.ndim
        reshape_shape[1] = -1
        gamma_reshaped = gamma.reshape(*reshape_shape)
        
        # Total number of elements in normalization dimensions
        m = np.prod([output_grad.shape[i] for i in self.stats_axes])

        d_beta = output_grad.sum(axis=self.stats_axes)
        d_gamma = (output_grad * self.x_hat).sum(axis=self.stats_axes)
        dx_hat = output_grad * gamma_reshaped
        
        # d_var = sum(dx_hat * x_centered * -0.5 * std_inv^3)
        dvar = (dx_hat * self.x_centered * -0.5 * (self.std_inv**3)).sum(axis=self.stats_axes, keepdims=True)
        
        # d_mean = sum(dx_hat * -std_inv)
        dmean = (dx_hat * -self.std_inv).sum(axis=self.stats_axes, keepdims=True)
        
        # dx = dx_hat * std_inv + dvar * 2 * x_centered / m + dmean / m
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
        # x: (N, C, ...) 
        if self.training:
            op = BatchNormOp(self.eps)
            out_val = op.forward(x.value, self.gamma.value, self.beta.value)
            
            # Use same axis logic for running stats update
            axis = (0,) + tuple(range(2, x.value.ndim))
                
            batch_mean = x.value.mean(axis=axis)
            batch_var = x.value.var(axis=axis)
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            return Node(out_val, parents=[x, self.gamma, self.beta], op=op)
        else:
            # Inference mode
            reshape_shape = [1] * x.value.ndim
            reshape_shape[1] = -1
            
            rm = self.running_mean.reshape(*reshape_shape)
            rv = self.running_var.reshape(*reshape_shape)
            g = self.gamma.value.reshape(*reshape_shape)
            b = self.beta.value.reshape(*reshape_shape)
                
            out_val = g * (x.value - rm) / np.sqrt(rv + self.eps) + b
            return Node(out_val, parents=[], op=None) # Leaf node in inference usually

