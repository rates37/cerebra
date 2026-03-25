import numpy as np
from typing import Union, Optional, Tuple, List
from ..core.node import Node, to_node
from ..core.ops import Operation
from .module import Parameter, Module


# function to convert a convolutional filter(s) to columns:
def convert_to_col(x: np.ndarray, kernel_h: int, kernel_w: int, stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]]) -> Tuple[np.ndarray, int, int]:
    # extract shape
    N, C, H, W = x.shape

    # handle stride/padding as tuples:
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    # apply padding:
    H += 2*pad_h
    W += 2*pad_w
    x = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h),
               (pad_w, pad_w)), mode='constant')

    # compute output shape
    out_h = (H - kernel_h) // stride_h + 1
    out_w = (W - kernel_w) // stride_w + 1

    # create output columns matrix:
    col = np.empty((N, C, kernel_h, kernel_w, out_h, out_w), dtype=x.dtype)

    # fill out cols:
    for y in range(kernel_h):
        y_max = y + out_h*stride_h

        for x_i in range(kernel_w):
            x_max = x_i + out_w*stride_w
            col[:, :, y, x_i, :, :] = x[:, :, y:y_max:stride_h, x_i:x_max:stride_w]
    col = col.reshape(N, C*kernel_h*kernel_w, out_h * out_w)
    return col, out_h, out_w


def convert_from_col(cols: np.ndarray, x_shape: Tuple[int, int, int, int], kernel_h: int,
                      kernel_w: int, stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]]) -> np.ndarray:
    # convert the tensor from the expanded column tensor back to original shape
    # x_shape = (N, C, H, W)
    N, C, H, W = x_shape

    # handle stride/padding as tuples:
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    h_padded = H + 2*pad_h
    w_padded = W + 2*pad_w
    h_out = (h_padded - kernel_h) // stride_h + 1
    w_out = (w_padded - kernel_w) // stride_w + 1

    cols_reshaped = cols.reshape(N, C, kernel_h, kernel_w, h_out, w_out)
    x_pad = np.zeros((N, C, h_padded, w_padded), dtype=cols.dtype)

    for y in range(kernel_h):
        max_y = y + stride_h*h_out
        for xi in range(kernel_w):
            max_x = xi + stride_w*w_out
            x_pad[:, :, y:max_y:stride_h,
                  xi:max_x:stride_w] += cols_reshaped[:, :, y, xi, :, :]

    if pad_h == 0 and pad_w == 0:
        return x_pad
    return x_pad[:, :, pad_h:H+pad_h, pad_w:W+pad_w]


class Conv2d(Operation):
    def __init__(self, stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0) -> None:
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
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
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
