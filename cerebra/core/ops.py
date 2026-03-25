import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional
from .node import Node, to_node, unbroadcast

#! ===================
#!     Operations
#! ===================
class Operation(ABC):
    @abstractmethod
    def forward(self, *inputs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        """Given gradient of output and the node (with parents), finds gradients for each parent

        Args:
            output_grad (np.ndarray): The gradient of the output of the operation
            node (Node): The output node

        Returns:
            List[np.ndarray]: A list of gradients for each parent of the node
        """
        pass

class Add(Operation):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        a_val, b_val = node.parents[0].value, node.parents[1].value

        return [
            unbroadcast(output_grad, a_val.shape),
            unbroadcast(output_grad, b_val.shape)
        ]

class Multiply(Operation):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        a_val, b_val = node.parents[0].value, node.parents[1].value
        return [
            unbroadcast(output_grad * b_val, a_val.shape),
            unbroadcast(output_grad * a_val, b_val.shape)
        ]


class MatMul(Operation):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a @ b

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        a_val = node.parents[0].value
        b_val = node.parents[1].value
        return [output_grad @ b_val.T, a_val.T @ output_grad]


class Sub(Operation):
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
    def forward(self, a: np.ndarray) -> np.ndarray:
        return -a

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        return [unbroadcast(-output_grad, node.parents[0].value.shape)]


# reshape operation:


class Reshape(Operation):
    def __init__(self, shape: Tuple[int, ...]) -> None:
        self.shape = shape
        self.original_shape: Optional[Tuple[int, ...]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.original_shape = x.shape
        return x.reshape(self.shape)

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        return [output_grad.reshape(self.original_shape)]


def reshape(x: Union[Node, np.ndarray], shape: Tuple[int, ...]) -> Node:
    x = to_node(x)
    op = Reshape(shape)
    new_val = op.forward(x.value)
    return Node(new_val, parents=[x], op=op)
