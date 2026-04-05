import numpy as np
from typing import Union, List
from ..core.node import Node, to_node
from ..core.ops import Operation

#! ========================
#!   Activation Functions
#! ========================


class ReLU(Operation):
    """Rectified Linear Unit (ReLU) activation operation.
    
    Applies the rectified linear unit function element-wise:
    ReLU(x) = max(0, x)
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return np.maximum(x, 0)

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        x_val = node.parents[0].value
        grad = output_grad * (x_val > 0)
        return [grad]


def relu(x: Union[Node, np.ndarray, float, int]) -> Node:
    """Applies the ReLU activation function.

    Args:
        x (Union[Node, np.ndarray, float, int]): The input tensor or node.
        
    Returns:
        Node: A new Node with the ReLU activation applied to the input.
    """
    x = to_node(x)
    op = ReLU()
    val = op.forward(x.value)
    return Node(val, parents=[x], op=op)


class Sigmoid(Operation):
    """Sigmoid activation operation.
    
    Applies the element-wise function:
    Sigmoid(x) = 1 / (1 + exp(-x))
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        out = 1.0 / (1.0 + np.exp(-x))
        self.out = out
        return out

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        grad = output_grad * (self.out * (1.0 - self.out))
        return [grad]


def sigmoid(x: Union[Node, np.ndarray, float, int]) -> Node:
    """Applies the sigmoid activation function.

    Args:
        x (Union[Node, np.ndarray, float, int]): The input tensor or node.
        
    Returns:
        Node: A new Node with the sigmoid activation applied to the input.
    """
    x = to_node(x)
    op = Sigmoid()
    val = op.forward(x.value)
    return Node(val, parents=[x], op=op)


class Tanh(Operation):
    """Hyperbolic tangent (Tanh) activation operation.
    
    Applies the element-wise function:
    Tanh(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        out = np.tanh(x)
        self.out = out
        return out

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        # d/dx(tanh(x)) = 1 - (tanh(x))^2
        grad = output_grad * (1.0 - self.out*self.out)
        return [grad]


def tanh(x: Union[Node, np.ndarray, float, int]) -> Node:
    """Applies the Tanh activation function.

    Args:
        x (Union[Node, np.ndarray, float, int]): The input tensor or node.
        
    Returns:
        Node: A new Node with the Tanh activation applied to the input.
    """
    x = to_node(x)
    op = Tanh()
    val = op.forward(x.value)
    return Node(val, parents=[x], op=op)


class LeakyReLU(Operation):
    """Leaky ReLU activation operation.
    
    Applies the element-wise function:
    LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)
    """
    def __init__(self, negative_slope: float = 0.01) -> None:
        """Initialises the LeakyReLU operation.
        
        Args:
            negative_slope (float, optional): Controls the angle of the negative slope. Defaults to 0.01.
        """
        self.negative_slope = negative_slope

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        out = np.where(self.mask, x, self.negative_slope * x)
        return out

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        grad = output_grad * (self.mask.astype(output_grad.dtype) +
                              (~self.mask).astype(output_grad.dtype)*self.negative_slope)
        return [grad]


def leaky_relu(x: Union[Node, np.ndarray, float, int], negative_slope: float = 0.01) -> Node:
    """Applies the Leaky ReLU activation function.

    Args:
        x (Union[Node, np.ndarray, float, int]): The input tensor or node.
        negative_slope (float, optional): Controls the angle of the negative slope. Defaults to 0.01.
        
    Returns:
        Node: A new Node with the Leaky ReLU activation applied to the input.
    """
    x = to_node(x)
    op = LeakyReLU(negative_slope)
    val = op.forward(x.value)
    return Node(val, parents=[x], op=op)


class ELU(Operation):
    """Exponential Linear Unit (ELU) activation operation.
    
    Applies the element-wise function:
    ELU(x) = max(0, x) + min(0, alpha * (exp(x) - 1))
    """
    def __init__(self, alpha: float = 1.0) -> None:
        """Initialises the ELU operation.
        
        Args:
            alpha (float, optional): The alpha value for the ELU formulation. Defaults to 1.0.
        """
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.positive_mask = x > 0
        self.e_x = np.exp(x)
        out = np.where(self.positive_mask, x, self.alpha * (self.e_x - 1.0))
        return out

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        grad = np.where(self.positive_mask, output_grad,
                        output_grad * (self.alpha * self.e_x))
        return [grad]


def elu(x: Union[Node, np.ndarray, float, int], alpha: float = 1.0) -> Node:
    """Applies the ELU activation function.

    Args:
        x (Union[Node, np.ndarray, float, int]): The input tensor or node.
        alpha (float, optional): The alpha value for the ELU formulation. Defaults to 1.0.
        
    Returns:
        Node: A new Node with the ELU activation applied to the input.
    """
    x = to_node(x)
    op = ELU(alpha)
    val = op.forward(x.value)
    return Node(val, parents=[x], op=op)
