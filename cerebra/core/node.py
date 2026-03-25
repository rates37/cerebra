from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Optional, Union, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .ops import Operation

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
        from .ops import Add
        op = Add()
        new_val = op.forward(self.value, other.value)
        return Node(new_val, parents=[self, other], op=op)

    def __radd__(self, other: Union[Node, np.ndarray, float, int]) -> Node:
        return to_node(other).__add__(self)

    def __mul__(self, other: Union[Node, np.ndarray, float, int]) -> Node:
        other = to_node(other)
        from .ops import Multiply
        op = Multiply()
        new_val = op.forward(self.value, other.value)
        return Node(new_val, parents=[self, other], op=op)

    def __rmul__(self, other: Union[Node, np.ndarray, float, int]) -> Node:
        return to_node(other).__mul__(self)

    def __matmul__(self, other: Union[Node, np.ndarray, float, int]) -> Node:
        other = to_node(other)
        from .ops import MatMul
        op = MatMul()
        new_val = op.forward(self.value, other.value)
        return Node(new_val, parents=[self, other], op=op)

    def __rmatmul__(self, other: Union[Node, np.ndarray, float, int]) -> Node:
        return to_node(other).__matmul__(self)

    def __sub__(self, other: Union[Node, np.ndarray, float, int]) -> Node:
        other = to_node(other)
        from .ops import Sub
        op = Sub()
        new_val = op.forward(self.value, other.value)
        return Node(new_val, parents=[self, other], op=op)

    def __rsub__(self, other: Union[Node, np.ndarray, float, int]) -> Node:
        return to_node(other).__sub__(self)

    def __neg__(self) -> Node:
        from .ops import Neg
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
