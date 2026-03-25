import numpy as np
from ..core.node import Node
from .module import Module, Parameter


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
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
        if self.bias is not None:
            return (x @ self.weight) + self.bias
        else:
            return (x @ self.weight)
