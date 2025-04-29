from __future__ import annotations
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from typing import Optional

#! =========
#!   Types
#! =========
Array = npt.NDArray[np.float64]
OptionalArray = Optional[Array]


#! ===============
#!   Basic Nodes
#! ===============
class Node:
    def __init__(self, value, parents: list[Node] = None, op: Operation=None, name:str=None) -> None:
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        self.value = value
        self.parents = parents or []
        self.op = op
        self.grad = None  # will be computed in backprop
        self.name = name

    def backward(self, grad=None) -> None:
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

    
    def top_sort_ancestors(self) -> list[Node]:
        visited = set()
        order = []
        def dfs(node):
            if node not in visited:
                visited.add(node)
                for parent in node.parents:
                    dfs(parent)
                order.append(node)
        dfs(self)
        return order

    def __hash__(self):
        # so that class can be used in set
        return id(self)

#! ===================
#!     Operations
#! ===================
class Operation(ABC):
    @abstractmethod
    def forward(self, *inputs):
        pass
    
    @abstractmethod
    def backward(self, output_grad, node):
        pass
    
class Add(Operation):
    def forward(self, a, b):
        return a + b
    
    def backward(self, output_grad, node):
        a_val, b_val = node.parents[0].value, node.parents[1].value
        return [output_grad, output_grad]

class Multiply(Operation):
    def forward(self, a, b):
        return a*b
    
    def backward(self, output_grad, node):
        a_val, b_val = node.parents[0].value, node.parents[1].value
        return [output_grad * b_val, output_grad * a_val]

class MatMul(Operation):
    def forward(self, a, b):
        return a @ b
    
    def backward(self, output_grad, node):
        a_val = node.parents[0].value
        b_val = node.parents[1].value
        return [output_grad @ b_val.T, a_val.T @ output_grad]
