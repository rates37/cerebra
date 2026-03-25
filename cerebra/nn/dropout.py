import numpy as np
from typing import Union, List
from ..core.node import Node, to_node
from ..core.ops import Operation
from .module import Module


# Dropout:
class DropoutOp(Operation):
    def __init__(self, p: float = 0.5, training: bool = True):
        # todo: check p is in range [0,1]
        self.p = p
        self.training = training
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training and self.p > 0:
            # create mask of same shape as input 0/1 with probability p
            self.mask = (np.random.rand(*x.shape) <
                         (1.0-self.p)).astype(x.dtype)

            # scale output by 1/(1-p) to preserve E[X]
            return (x * self.mask) / (1.0 - self.p)
        else:
            self.mask = np.ones_like(x, dtype=x.dtype)
            return x

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        if self.training and self.p > 0 and self.mask is not None:
            grad = (output_grad * self.mask) / (1-self.p)
        else:
            grad = output_grad
        return [grad]


def dropout(x: Union[Node, np.ndarray], p: float = 0.5, training: bool = True) -> Node:
    x = to_node(x)
    op = DropoutOp(p=p, training=training)
    out = op.forward(x.value)
    return Node(out, parents=[x], op=op)


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        # todo: check p in [0, 1]
        self.p = p
        self.training = True

    def forward(self, x: Node) -> Node:
        return dropout(x, p=self.p, training=self.training)
