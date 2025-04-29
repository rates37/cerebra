import numpy as np
import graph
from abc import ABC, abstractmethod
from graph import Node, Operation


class Parameter(Node):
    def __init__(self, value, name=None):
        super().__init__(np.array(value), parents=[], op=None, name=name)


class Module(ABC):
    def __init__(self):
        self._parameters = {}
        self._modules = {}

    def parameters(self) -> list[Parameter]:
        parameters = []
        # loop over all object attributes
        # todo: find a cleaner way to do this if it exists
        for attribute in self.__dict__.values():
            if isinstance(attribute, Parameter):
                parameters.append(attribute)
            elif isinstance(attribute, Module):
                parameters.extend(attribute.parameters())

        return parameters

    def __call__(self, *args, **kwargs):
        args = [graph.to_node(a) for a in args]
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass


class Linear(Module):
    """
    Simple FC Layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = Parameter(
            np.random.randn(in_features, out_features) * np.sqrt(2 / in_features),
        )
        if bias:
            self.bias = Parameter(np.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        if self.bias:
            return (x @ self.weight) + self.bias
        else:
            return x @ self.weight


# Relu operation:
class ReLU(Operation):
    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, grad_output, node):
        x_val = node.parents[0].value
        grad = grad_output * (x_val > 0)
        return [grad]


def relu(x):
    x = graph.to_node(x)
    op = ReLU()
    value = op.forward(x.value)
    return Node(value, parents=[x], op=op)
