from __future__ import annotations
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from typing import Optional, Union, List

#! =========
#!   Types
#! =========
Array = npt.NDArray[np.float64]
OptionalArray = Optional[Array]


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
        self.parents = parents or []
        self.op = op
        self.grad = None  # will be computed in backprop
        self.name = name

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


class Variable(Node):
    """
    Special case of node that is a leaf node of the computational graph.
    Can be used as input or parameter to the graph
    """

    def __init__(self, value: Union[np.ndarray, float, int], name: Optional[str] = None) -> None:
        super().__init__(value, parents=[], op=None, name=name)


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
        return [output_grad, output_grad]


class Multiply(Operation):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        a_val, b_val = node.parents[0].value, node.parents[1].value
        return [output_grad * b_val, output_grad * a_val]


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
        return [output_grad, -output_grad]


class ReLU(Operation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return np.maximum(x, 0)

    def backward(self, output_grad, node):
        x_val = node.parents[0].value
        grad = output_grad * (x_val > 0)
        return [grad]


# used as a separate function since no operator to overload
def relu(x: Union[Node, np.ndarray, float, int]) -> Node:
    x = to_node(x)
    op = ReLU()
    val = op.forward(x.value)
    return Node(val, parents=[x], op=op)


#! =====================
#!   Utility Functions
#! =====================
def to_node(x: Union[Node, np.ndarray, float, int]) -> Node:
    return x if isinstance(x, Node) else Node(x)
