import numpy as np
from abc import ABC, abstractmethod
from .graph import Node, Variable, Operation, to_node
from typing import Union, Optional, Any, List, Tuple


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
            np.random.randn(in_features, out_features) * np.sqrt(2 / (in_features)),
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


# function to convert a convolutional filter(s) to columns:
def convert_to_col(x: np.ndarray, kernel_h: int, kernel_w: int, stride: int, padding: int) -> Tuple[np.ndarray, int, int]:
    # extract shape
    N, C, H, W = x.shape

    # apply padding:
    H += 2*padding
    W += 2*padding
    x = np.pad(x, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')

    # compute output shape
    out_h = (H - kernel_h) // stride + 1
    out_w = (W - kernel_w) // stride + 1

    # create output columns matrix:
    col = np.empty((N, C, kernel_h, kernel_w, out_h, out_w), dtype=x.dtype)

    # fill out cols:
    for y in range(kernel_h):
        y_max = y + out_h*stride

        for x_i in range(kernel_w):
            x_max = x_i + out_w*stride
            col[:, :, y, x_i, :, :] = x[:,:, y:y_max:stride, x_i:x_max:stride]
    col = col.reshape(N, C*kernel_h*kernel_w, out_h * out_w)
    return col, out_h, out_w


def convert_from_col(cols: np.ndarray, x_shape: Tuple[int, int, int, int], kernel_h: int, kernel_w: int) -> np.ndarray:
    pass

class Conv2d(Operation):
    def __init__(self, stride: int = 1, padding: int = 0) -> None:
        # todo: allow for differing x and y stride/paddings
        # todo: convert stride / padding type from int to Union[int, Tuple[int, int]]
        self.stride = stride
        self.padding = padding

    def forward(self, x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None) -> np.ndarray:
        # store shape of input and parameters:
        self.x_shape = x.shape
        N, *_ = x.shape
        C_out, C_in, kh, kw = weight.shape
        self.kernel_h, self.kernel_w = kh, kw

        # convert to col
        col, out_h, out_w = convert_to_col(x, kh, kw, self.stride, self.padding)
        # store shape of col matrix
        self.col = col
        self.out_h = out_h
        self.out_w = out_w

        weight_col = weight.reshape(C_out, -1)
        output = np.empty((N, C_out, out_h * out_w), dtype=x.dtype)
        for i in range(N):
            output[i] = weight_col @ col[i]
        
        # apply bias:
        if bias is not None:
            output += bias.reshape(1, C_out, 1)
        output = output.reshape(N, C_out, out_h, out_w)
        return output

    def backward(self, grad_output: np.ndarray, node: Node) -> List[np.ndarray]:
        pass


class Conv2dLayer(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: int = 1,
                 padding: int = 0,
                 bias: bool = True
                 ) -> None:
        pass

    def forward(self, x: Node) -> Node:
        pass
