import numpy as np
from ..core.node import Node
from .module import Module, Parameter


class Linear(Module):
    """Linear (fully connected) layer.
    
    Applies a linear transformation to the incoming data: $y = xA^T + b$.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """Initialises the linear layer with weights and optional bias.
        
        Args:
            in_features (int): size of each input sample.
            out_features (int): size of each output sample.
            bias (bool, optional): If set to False, the layer will not include a set of bias parameters. Default: True.
        """
        super().__init__()
        # Normal Xavier Initialisation
        self.weight = Parameter(
            np.random.randn(in_features, out_features) *
            np.sqrt(2 / (in_features + out_features)),
            name="weight"
        )

        if bias:
            self.bias = Parameter(np.zeros(out_features), name="bias")
        else:
            self.bias = None

    def forward(self, x: Node) -> Node:
        """Performs the forward pass of the linear layer.

        Args:
            x (Node): The input node containing the input tensor.

        Returns:
            Node: Output node after applying the linear transformation with this
            layer's parameters.
        """
        if self.bias is not None:
            return (x @ self.weight) + self.bias
        else:
            return (x @ self.weight)
