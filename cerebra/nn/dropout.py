import numpy as np
from typing import Union, List
from ..core.node import Node, to_node
from ..core.ops import Operation
from .module import Module


# Dropout:
class DropoutOp(Operation):
    """Dropout operation that randomly zeroes out elements.
    
    During training, randomly zeroes some of the elements of the input tensor with 
    probability p using samples from a Bernoulli distribution.
    """
    def __init__(self, p: float = 0.5, training: bool = True):
        """Initialises the dropout operation.
        
        Args:
            p (float, optional): probability of an element to be zeroed. Default: 0.5
            training (bool, optional): apply dropout if is True. Default: True
        """
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
    """Applies dropout to a node's value.

    Args:
        x (Union[Node, np.ndarray]): The input tensor / node.
        p (float, optional): probability of an element to be zeroed. Default: 0.5
        training (bool, optional): apply dropout if is True. Default: True
        
    Returns:
        Node: A new Node with dropout applied.
    """
    x = to_node(x)
    op = DropoutOp(p=p, training=training)
    out = op.forward(x.value)
    return Node(out, parents=[x], op=op)


class Dropout(Module):
    """Dropout layer.
    
    During training, randomly zeroes some of the elements of the input tensor with 
    probability p. The outputs are scaled by an expected value of 1/(1-p) so that 
    during evaluation the module simply computes an identity function.
    """
    def __init__(self, p: float = 0.5):
        """Initialises the dropout layer.
        
        Args:
            p (float, optional): probability of an element to be zeroed. Default: 0.5
        """
        # todo: check p in [0, 1]
        self.p = p
        self.training = True

    def forward(self, x: Node) -> Node:
        """Applies dropout forward pass.

        Args:
            x (Node): The input node.

        Returns:
            Node: Output node containing the result after applying dropout.
        """
        return dropout(x, p=self.p, training=self.training)
