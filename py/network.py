import numpy as np
from abc import ABC, abstractmethod
from graph import Node, Variable, Operation, to_node
from typing import Union, Optional, Any, List


class Parameter(Variable):
    def __init__(self, value: Union[np.ndarray, float, int], name: Optional[str] = None) -> None:
        super().__init__(value, name=name)


class Module(ABC):
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def parameters(self) -> List[Parameter]:
        params = []
        for attribute in self.__dict__.values():
            if isinstance(attribute, Parameter):
                params.append(attribute)
            elif isinstance(attribute, Module):
                params.extend(attribute.parameters())
        return params


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        # Normal Xavier Initialisation
        self.weight = Parameter(
            np.random.randn(in_features, out_features) * np.sqrt(2 / (in_features+out_features)),
            name="weight"
        )

        if bias:
            self.bias = Parameter(np.zeros(out_features), name="bias")
        else:
            self.bias = None

    def forward(self, x: Node) -> Node:
        if self.bias is not None:
            return (x @ self.weight) + self.bias
        else:
            return (x @ self.weight)


class CrossEntropyLoss(Operation):
    def __init__(self, target: np.ndarray) -> None:
        self.target = target

    def forward(self, x: np.ndarray) -> np.ndarray:
        # presume x in format: (batch_size, num_classes)
        x_cpy = x.copy()
        x_max = np.max(x_cpy, axis=1, keepdims=True)
        x_exp = np.exp(x_cpy - x_max)
        sum_exp = np.sum(x_exp, axis=1, keepdims=True)
        self.softmax = x_exp / sum_exp
        batch_size = x_cpy.shape[0]
        log_probabilities = np.log(self.softmax[np.arange(
            batch_size), self.target] + 1e-12)  # prevent instabilities
        loss_val = -np.mean(log_probabilities)
        return np.array([loss_val])

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        batch_size, _ = self.softmax.shape
        one_hot_logits = np.zeros_like(self.softmax)
        one_hot_logits[np.arange(batch_size), self.target] = 1.0
        grad = (self.softmax - one_hot_logits) / batch_size
        grad = grad * output_grad
        return [grad]


# function for CELoss so can be used as functional block:
def cross_entropy_loss(x: Union[Node, np.ndarray, float, int], target: np.ndarray) -> Node:
    x = to_node(x)
    op = CrossEntropyLoss(target)
    val = op.forward(x.value)
    return Node(val, parents=[x], op=op)
