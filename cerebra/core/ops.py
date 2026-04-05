import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional
from .node import Node, to_node, unbroadcast

#! ===================
#!     Operations
#! ===================
class Operation(ABC):
    """Base class for all operations in the computational graph."""
    @abstractmethod
    def forward(self, *inputs: np.ndarray) -> np.ndarray:
        """Executes the forward pass of the operation.

        This method should be implemented by all operations.

        Args:
            *inputs (np.ndarray): The input arrays for the operation.

        Returns:
            np.ndarray: The result of the operation.
        """
        pass

    @abstractmethod
    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        """Given gradient of output and the node (with parents), finds gradients for each parent

        This method should be implemented by all operations.

        Args:
            output_grad (np.ndarray): The gradient of the output of the operation
            node (Node): The output node

        Returns:
            List[np.ndarray]: A list of gradients for each parent of the node
        """
        pass

class Add(Operation):
    """Addition operation node."""
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        a_val, b_val = node.parents[0].value, node.parents[1].value

        return [
            unbroadcast(output_grad, a_val.shape),
            unbroadcast(output_grad, b_val.shape)
        ]

class Multiply(Operation):
    """Element-wise multiplication operation node."""
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        a_val, b_val = node.parents[0].value, node.parents[1].value
        return [
            unbroadcast(output_grad * b_val, a_val.shape),
            unbroadcast(output_grad * a_val, b_val.shape)
        ]


class MatMul(Operation):
    """Matrix multiplication operation node."""
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a @ b

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        a_val = node.parents[0].value
        b_val = node.parents[1].value
        return [output_grad @ b_val.T, a_val.T @ output_grad]


class Sub(Operation):
    """Subtraction operation node."""
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a - b

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        a_val = node.parents[0].value
        b_val = node.parents[1].value
        return [
            unbroadcast(output_grad, a_val.shape),
            unbroadcast(-output_grad, b_val.shape)
        ]


class Neg(Operation):
    """Negation operation node."""
    def forward(self, a: np.ndarray) -> np.ndarray:
        return -a

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        return [unbroadcast(-output_grad, node.parents[0].value.shape)]


# reshape operation:


class Reshape(Operation):
    """Reshape operation node."""
    def __init__(self, shape: Tuple[int, ...]) -> None:
        """Initialises the Reshape operation.
        
        Args:
            shape (Tuple[int, ...]): The target shape to reshape the input to.
        """
        self.shape = shape
        self.original_shape: Optional[Tuple[int, ...]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.original_shape = x.shape
        return x.reshape(self.shape)

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        return [output_grad.reshape(self.original_shape)]


def reshape(x: Union[Node, np.ndarray], shape: Tuple[int, ...]) -> Node:
    """Reshapes a node into a new shape.
    
    This function acts as a wrapper around the Reshape operation.
    
    Args:
        x (Union[Node, np.ndarray]): The input node or array to reshape.
        shape (Tuple[int, ...]): The target shape.

    Returns:
        Node: A new node representing the reshaped tensor.
    """
    x = to_node(x)
    op = Reshape(shape)
    new_val = op.forward(x.value)
    return Node(new_val, parents=[x], op=op)
