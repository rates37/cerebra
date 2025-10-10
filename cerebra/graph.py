from __future__ import annotations
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple

#! =========
#!   Types
#! =========
Array = npt.NDArray[np.float64]
OptionalArray = Optional[Array]


#! ===========
#!   Config
#! ===========
_GRAD_ENABLED = True

#! ===============
#!   Basic Nodes
#! ===============


class Node:
    def __init__(
        self,
        value: Union[np.ndarray, float, int],
        parents: Optional[List[Node]] = None,
        op: Optional[Operation] = None,
        name: Optional[str] = None
    ) -> None:
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        self.value = value
        self.grad = None  # will be computed in backprop
        self.name = name

        if is_grad_enabled():
            self.parents = parents or []
            self.op = op
        else:
            self.parents = []
            self.op = None

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        nodes = self.top_sort_ancestors()
        for node in nodes:
            node.grad = np.zeros_like(node.value)
        if grad is None:
            grad = np.ones_like(self.value)
        self.grad = self.grad + grad
        for node in reversed(nodes):
            if node.op is not None:
                grads = node.op.backward(node.grad, node)
                for parent, g in zip(node.parents, grads):
                    parent.grad = parent.grad + g

    def top_sort_ancestors(self) -> List[Node]:
        visited = set()
        order = []

        def dfs(node: Node) -> None:
            if node not in visited:
                visited.add(node)
                for parent in node.parents:
                    dfs(parent)
                order.append(node)

        dfs(self)
        return order

    def __hash__(self) -> int:
        # so that class can be used in set
        return id(self)

    # magic methods:
    def __add__(self, other: Union[Node, np.ndarray, float, int]) -> Node:
        other = to_node(other)
        op = Add()
        new_val = op.forward(self.value, other.value)
        return Node(new_val, parents=[self, other], op=op)

    def __radd__(self, other: Union[Node, np.ndarray, float, int]) -> Node:
        return to_node(other).__add__(self)

    def __mul__(self, other: Union[Node, np.ndarray, float, int]) -> Node:
        other = to_node(other)
        op = Multiply()
        new_val = op.forward(self.value, other.value)
        return Node(new_val, parents=[self, other], op=op)

    def __rmul__(self, other: Union[Node, np.ndarray, float, int]) -> Node:
        return to_node(other).__mul__(self)

    def __matmul__(self, other: Union[Node, np.ndarray, float, int]) -> Node:
        other = to_node(other)
        op = MatMul()
        new_val = op.forward(self.value, other.value)
        return Node(new_val, parents=[self, other], op=op)

    def __rmatmul__(self, other: Union[Node, np.ndarray, float, int]) -> Node:
        return to_node(other).__matmul__(self)

    def __sub__(self, other: Union[Node, np.ndarray, float, int]) -> Node:
        other = to_node(other)
        op = Sub()
        new_val = op.forward(self.value, other.value)
        return Node(new_val, parents=[self, other], op=op)

    def __rsub__(self, other: Union[Node, np.ndarray, float, int]) -> Node:
        return to_node(other).__sub__(self)

    def __neg__(self) -> Node:
        op = Neg()
        new_val = op.forward(self.value)
        return Node(new_val, parents=[self], op=op)


class Variable(Node):
    """
    Special case of node that is a leaf node of the computational graph.
    Can be used as input or parameter to the graph
    """

    def __init__(self, value: Union[np.ndarray, float, int], name: Optional[str] = None) -> None:
        super().__init__(value, parents=None, op=None, name=name)


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

#! ========================
#!   Activation Functions
#! ========================


class ReLU(Operation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return np.maximum(x, 0)

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        x_val = node.parents[0].value
        grad = output_grad * (x_val > 0)
        return [grad]


def relu(x: Union[Node, np.ndarray, float, int]) -> Node:
    x = to_node(x)
    op = ReLU()
    val = op.forward(x.value)
    return Node(val, parents=[x], op=op)


class Sigmoid(Operation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        out = 1.0 / (1.0 + np.exp(-x))
        self.out = out
        return out

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        grad = output_grad * (self.out * (1.0 - self.out))
        return [grad]


def sigmoid(x: Union[Node, np.ndarray, float, int]) -> Node:
    x = to_node(x)
    op = Sigmoid()
    val = op.forward(x.value)
    return Node(val, parents=[x], op=op)


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

#! =====================
#!   Utility Functions
#! =====================


def to_node(x: Union[Node, np.ndarray, float, int]) -> Node:
    return x if isinstance(x, Node) else Node(x)


def unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    # reduce extra dimensions introduced by broadcasting:
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)

    # sum over dimensions that were originally 1:
    for axis, dimension in enumerate(shape):
        if dimension == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class no_grad:
    """Context manager to disable gradient tracking"""

    def __enter__(self) -> None:
        global _GRAD_ENABLED
        self.prev = _GRAD_ENABLED  # store previous state to allow nested contexts
        _GRAD_ENABLED = False

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        global _GRAD_ENABLED
        _GRAD_ENABLED = self.prev


def is_grad_enabled() -> bool:
    """function to check if grad is enabled

    Returns:
        bool: Whether grad is enabled
    """
    global _GRAD_ENABLED
    return _GRAD_ENABLED
