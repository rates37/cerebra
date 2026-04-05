import numpy as np
from typing import Union, Tuple, List
from ..core.node import Node
from ..core.ops import Operation
from .conv import convert_to_col, convert_from_col
from .module import Module

# 2D Pooling layer Operations/Functions:

class MaxPool2DOp(Operation):
    """2D Max Pooling operation."""
    def __init__(self, kernel_size: Tuple[int, int], stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0) -> None:
        """Initialises the 2D max pooling operation.

        Args:
            kernel_size (Tuple[int, int]): the size of the window to take a max over.
            stride (Union[int, Tuple[int, int]], optional): the stride of the window. Can be int or 
                tuple of two ints (representing horizontal and vertical stride). Defaults to 1.
            padding (Union[int, Tuple[int, int]], optional): implicit zero padding to be added on
                both sides. Can be int or tuple of two ints (representing horizontal and vertical
                padding). Defaults to 0.
        """
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
    """2D Average Pooling operation."""
    def __init__(self, kernel_size: Tuple[int, int], stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0) -> None:
        """Initialises the 2D average pooling operation.

        Args:
            kernel_size (Tuple[int, int]): the size of the window to take an average over.
            stride (Union[int, Tuple[int, int]], optional): the stride of the window. Can be int or 
                tuple of two ints (representing horizontal and vertical stride). Defaults to 1.
            padding (Union[int, Tuple[int, int]], optional): implicit zero padding to be added on
                both sides. Can be int or tuple of two ints (representing horizontal and vertical
                padding). Defaults to 0.
        """
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
    """2D Max Pooling layer.
    
    Applies a 2D max pooling over an input signal composed of several input planes.
    """
    def __init__(self,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0
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
    """2D Average Pooling layer.
    
    Applies a 2D average pooling over an input signal composed of several input planes.
    """
    def __init__(self,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0
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
