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
            np.random.randn(in_features, out_features) *
            np.sqrt(2 / (in_features + out_features)),
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
    x = np.pad(x, ((0, 0), (0, 0), (padding, padding),
               (padding, padding)), mode='constant')

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
            col[:, :, y, x_i, :, :] = x[:, :, y:y_max:stride, x_i:x_max:stride]
    col = col.reshape(N, C*kernel_h*kernel_w, out_h * out_w)
    return col, out_h, out_w


def convert_from_col(cols: np.ndarray, x_shape: Tuple[int, int, int, int], kernel_h: int,
                     kernel_w: int, stride: int, padding: int) -> np.ndarray:
    # convert the tensor from the expanded column tensor back to original shape
    # x_shape = (N, C, H, W)
    N, C, H, W = x_shape
    h_padded = H + 2*padding
    w_padded = W + 2*padding
    h_out = (h_padded - kernel_h) // stride + 1
    w_out = (w_padded - kernel_w) // stride + 1

    cols_reshaped = cols.reshape(N, C, kernel_h, kernel_w, h_out, w_out)
    x_pad = np.zeros((N, C, h_padded, w_padded), dtype=cols.dtype)

    for y in range(kernel_h):
        max_y = y + stride*h_out
        for xi in range(kernel_w):
            max_x = xi + stride*w_out
            x_pad[:, :, y:max_y:stride,
                  xi:max_x:stride] += cols_reshaped[:, :, y, xi, :, :]
    if padding == 0:
        return x_pad
    return x_pad[:, :, padding:-padding, padding:-padding]


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
        col, out_h, out_w = convert_to_col(
            x, kh, kw, self.stride, self.padding)
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

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        N, C_out, _, _ = output_grad.shape
        output_grad_reshaped = output_grad.reshape(N, C_out, -1)

        # grad of W
        weight_grad = np.zeros(
            (C_out, self.col.shape[1]), dtype=output_grad.dtype)
        for i in range(N):
            weight_grad += output_grad_reshaped[i] @ self.col[i].T
        # convert back to shape of regular W:
        weight_grad = weight_grad.reshape(node.parents[1].value.shape)

        # compute bias grad if it exists:
        bias_grad = None
        if len(node.parents) == 3:
            bias_grad = output_grad.sum(axis=(0, 2, 3))

        # calculate input gradient:
        weight = node.parents[1].value
        weight_col = weight.reshape(C_out, -1)
        # empty faster than zeroslike since doesn't initialise values
        grad_col = np.empty_like(self.col)
        for i in range(N):
            grad_col[i] = weight_col.T @ output_grad_reshaped[i]
        x_grad = convert_from_col(
            grad_col, self.x_shape, self.kernel_h, self.kernel_w, self.stride, self.padding)
        if bias_grad is not None:
            return [x_grad, weight_grad, bias_grad]
        else:
            return [x_grad, weight_grad]


class Conv2dLayer(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: int = 1,
                 padding: int = 0,
                 bias: bool = True
                 ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        fan_in = in_channels * kernel_size[0] * kernel_size[1]
        fan_out = out_channels * kernel_size[0] * kernel_size[1]
        self.weight = Parameter(
            np.random.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]) *
            np.sqrt(2 / (fan_in + fan_out)),
            name="conv2d_weight"
        )
        if bias:
            self.bias = Parameter(np.zeros(out_channels), name="conv2d_bias")
        else:
            self.bias = None

    def forward(self, x: Node) -> Node:
        weight = to_node(self.weight)
        op = Conv2d(self.stride, self.padding)

        if self.bias is not None:
            bias = to_node(self.bias)
            output = op.forward(x.value, weight.value, bias.value)
            return Node(output, parents=[x, weight, bias], op=op)
        else:
            output = op.forward(x.value, weight.value)
            return Node(output, parents=[x, weight], op=op)


# 2D Pooling layer Operations/Functions:

class MaxPool2DOp(Operation):
    def __init__(self, kernel_size: Tuple[int, int], stride: int = 1, padding: int = 0) -> None:
        self.kh, self.kw = kernel_size
        self.stride = stride
        self.padding = padding

        self.x_shape: Tuple[int, int, int, int] = ()
        self.out_h = self.out_w = 0
        self.maxIdx: np.ndarray = np.array([])

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = N, C, H, W = x.shape
        cols, self.out_h, self.out_w = convert_to_col(
            x, self.kh, self.kw, self.stride, self.padding)
        # cols.shape is (N, C * kh * kw, oh*pw)
        # reshape to (N, C, kh*kw, oh*pw):
        cols_reshaped = cols.reshape(
            N, C, self.kh*self.kw, self.out_h*self.out_w)
        maxVals = cols_reshaped.max(axis=2)  # (N,C,oh*pw)
        # for backprop, contains indices [0 .. kh*kw-1]
        self.maxIdx = cols_reshaped.argmax(axis=2)
        return maxVals.reshape(N, C, self.out_h, self.out_w)

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        N, C, _, _ = output_grad.shape
        L = self.out_h*self.out_w

        # reshape gradient to (N,C,L):
        flattened_gradient = output_grad.reshape(N, C, L)

        # create masK:
        mask = np.zeros((N, C, self.kw*self.kh, L), dtype=output_grad.dtype)
        for i in range(N):
            for j in range(C):
                mask[i, j, self.maxIdx[i, j, :], np.arange(L)] = 1
        grad_col = mask * flattened_gradient[:, :, None, :]
        grad_col = grad_col.reshape(N, C*self.kh*self.kw, L)
        x_grad = convert_from_col(
            grad_col, self.x_shape, self.kh, self.kw, self.stride, self.padding)
        return [x_grad]


class AvgPool2DOp(Operation):
    def __init__(self, kernel_size: Tuple[int, int], stride: int = 1, padding: int = 0) -> None:
        self.kh, self.kw = kernel_size
        self.stride = stride
        self.padding = padding

        self.x_shape: Tuple[int, int, int, int] = ()
        self.out_h = self.out_w = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = N, C, H, W = x.shape
        cols, self.out_h, self.out_w = convert_to_col(
            x, self.kh, self.kw, self.stride, self.padding)
        cols_reshaped = cols.reshape(
            N, C, self.kh*self.kw, self.out_h*self.out_w)

        #
        meanVals = cols_reshaped.mean(axis=2)
        return meanVals.reshape(N, C, self.out_h, self.out_w)

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        N, C, _, _ = output_grad.shape
        L = self.out_h*self.out_w

        # reshape gradient to (N,C,L):
        flattened_gradient = output_grad.reshape(N, C, L)

        gradient_per_element = (flattened_gradient / (self.kh * self.kw))
        gradient_cols = np.repeat(
            gradient_per_element[:, :, None, :], self.kh * self.kw, axis=2)
        # grad_cols is shape (N,C,kh*kw,L)
        x_grad = convert_from_col(
            gradient_cols, self.x_shape, self.kh, self.kw, self.stride, self.padding)
        return [x_grad]

# 2D Pooling Modules:


class MaxPool2D(Module):
    def __init__(self,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: int = 1,
                 padding: int = 0
                 ):
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Node) -> Node:
        op = MaxPool2DOp(self.kernel_size, self.stride, self.padding)
        out = op.forward(x.value)
        return Node(out, parents=[x], op=op)


class AvgPool2D(Module):
    def __init__(self,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: int = 1,
                 padding: int = 0
                 ):
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Node) -> Node:
        op = AvgPool2DOp(self.kernel_size, self.stride, self.padding)
        out = op.forward(x.value)
        return Node(out, parents=[x], op=op)


# Dropout:
class DropoutOp(Operation):
    def __init__(self, p: float = 0.5, training: bool = True):
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
    x = to_node(x)
    op = DropoutOp(p=p, training=training)
    out = op.forward(x.value)
    return Node(out, parents=[x], op=op)


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        # todo: check p in [0, 1]
        self.p = p
        self.training = True

    def forward(self, x: Node) -> Node:
        return dropout(x, p=self.p, training=self.training)
