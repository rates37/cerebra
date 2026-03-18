import unittest
import numpy as np
from typing import Callable
from cerebra import Node, Conv2d, Conv2dLayer, Parameter, Operation, no_grad

EPSILON = 1e-6


def numerical_gradient(func: Callable[[], Node], node: Node, h=1e-6) -> np.ndarray:
    """Compute gradient of node w.r.t. input

    Args:
        func (_type_): Callable that when invoked, re-evaluates computational graph
                        based on the current state of `node`.
        node (Node): The node whose value attribute will be altered to calculate the
                        gradient
        h (_type_, optional): Small perturbation value. Defaults to 1e-6.

    Returns:
        np.ndarray: Numpy array representing the numerically calculated gradient. Of same 
                        shape as `node.value`.
    """
    input_value = node.value.copy()
    grad = np.zeros_like(input_value, dtype=np.float64)

    for i in np.ndindex(input_value.shape):
        val = input_value[i]

        # calculate f(x+h):
        x_plus_h = input_value.copy()
        x_plus_h[i] = val + h
        node.value = x_plus_h
        f_x_plus_h = func().value.item()  # func always returns a scalar (eg total loss)

        # calculate f(x-h):
        x_minus_h = input_value.copy()
        x_minus_h[i] = val - h
        node.value = x_minus_h
        f_x_minus_h = func().value.item()

        # central difference method:
        grad[i] = (f_x_plus_h - f_x_minus_h) / (2 * h)

    # restore input value of node
    node.value = input_value
    return grad


class TestConv2d(unittest.TestCase):
    def setUp(self) -> None:
        self.default_rng = np.random.default_rng(69)

    def tearDown(self) -> None:
        pass

    #! =========================
    #!    Forward Conv Tests
    #! =========================

    def test_conv2d_forward_no_bias(self):
        x_val = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
        w_val = np.array([[[[1, 0], [0, 1]]]], dtype=np.float32)
        # expected value is: (1*1 + 2*0 + 3*0 + 4*1) = 5
        expected_val = np.array([5])
        op = Conv2d(stride=1, padding=0)
        output = op.forward(x=x_val, weight=w_val, bias=None)

        self.assertTrue(np.allclose(output, expected_val, atol=EPSILON))

    def test_conv2d_forward_with_bias(self):
        x_val = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
        w_val = np.array([[[[1, 0], [0, 1]]]], dtype=np.float32)
        b_val = np.array([0.5], dtype=np.float32)
        # expected value is: (1*1 + 2*0 + 3*0 + 4*1) + 0.5 = 5.5
        expected_val = np.array([5.5])
        op = Conv2d(stride=1, padding=0)
        output = op.forward(x=x_val, weight=w_val, bias=b_val)

        self.assertTrue(np.allclose(output, expected_val, atol=EPSILON))

    def test_conv2d_op_forward_padding_stride(self):
        op = Conv2d(stride=2, padding=1)
        x_val = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
        w_val = np.array([[[[1, 1], [1, 1]]]], dtype=np.float32)
        # Padded_x:
        # 0 0 0 0
        # 0 1 2 0
        # 0 3 4 0
        # 0 0 0 0

        # output dimensions:
        # out_h = (2+2*1-2)//2 + 1 = 2
        # out_w = (2+2*1-2)//2 + 1 = 2

        # output matrix entries:
        # Top left from padded x[0:2, 0:2] = [[0,0],[0,1]] -> sum = 1
        # Top right from padded x[0:2, 2:4] = [[0,0],[2,0]] -> sum = 2
        # Bottom left from padded x[2:4, 0:2] = [[0,3],[0,0]] -> sum = 3
        # Bottom right from padded x[2:4, 2:4] = [[4,0],[0,0]] -> sum = 4
        expected_out = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
        out = op.forward(x_val, w_val)
        self.assertTrue(np.allclose(out, expected_out, atol=EPSILON))

    def test_conv2d_op_forward_multi_channel_in(self):
        op = Conv2d(stride=1, padding=0)
        # C_in=2, H_in=1, W_in=1
        x_val = np.array([[[[10.]], [[20.]]]], dtype=np.float32)
        w_val = np.array([[[[2.]], [[3.]]]], dtype=np.float32)
        b_val = np.array([0.5], dtype=np.float32)
        # Expected: (10*2 + 20*3) + 0.5 = 80.5
        expected_out = np.array([[[[80.5]]]], dtype=np.float32)
        out = op.forward(x_val, w_val, b_val)
        self.assertTrue(np.allclose(out, expected_out, atol=EPSILON))

    def test_conv2d_op_forward_multi_channel_out(self):
        op = Conv2d(stride=1, padding=0)
        # C_out=2
        x_val = np.array([[[[5]]]], dtype=np.float32)
        w_val = np.array([[[[2]]], [[[3]]]], dtype=np.float32)
        b_val = np.array([0.1, 0.2], dtype=np.float32)

        # Expected:
        # Channel 0: (5*2) + 0.1 = 10.1
        # Channel 1: (5*3) + 0.2 = 15.2
        expected_out = np.array([[[[10.1]], [[15.2]]]], dtype=np.float32)
        out = op.forward(x_val, w_val, b_val)
        self.assertTrue(np.allclose(out, expected_out, atol=EPSILON))

    #! ==========================
    #!    Backward Conv Tests
    #! ==========================
    # generic function to test backward method:
    def _check_conv_op_backward(
            self,
            op: Operation,
            x_val: np.ndarray,
            w_val: np.ndarray,
            b_val=None,
            grad_val=None
    ) -> None:
        # copy input values:
        x_node = Node(x_val.copy())
        w_node = Node(w_val.copy())
        parents = [x_node, w_node]
        args = [x_node.value, w_node.value]

        if b_val is not None:
            b_node = Node(b_val.copy())
            parents.append(b_node)
            args.append(b_node.value)
        else:
            b_node = None
        # run the forward:
        output_from_forward = op.forward(*args)

        # generate random loss if necessary
        if grad_val is None:
            grad_val = self.default_rng.random(
                output_from_forward.shape).astype(np.float32)

        # dummy output node:
        dummy_output = Node(output_from_forward, parents=parents, op=op)
        grads = op.backward(grad_val, dummy_output)

        # def function for analytical gradient calc:
        def get_loss_as_node() -> Node:
            current_args = [p.value for p in parents]
            out_val = op.forward(*current_args)
            loss_val = (out_val * grad_val).sum()
            return Node(np.array([loss_val]))

        dx_numerical = numerical_gradient(get_loss_as_node, x_node)
        self.assertTrue(np.allclose(
            grads[0], dx_numerical, atol=EPSILON, rtol=EPSILON))

        dw_numerical = numerical_gradient(get_loss_as_node, w_node)
        self.assertTrue(np.allclose(
            grads[1], dw_numerical, atol=EPSILON, rtol=EPSILON))

        if b_val is not None and b_node is not None:
            db_numerical = numerical_gradient(get_loss_as_node, b_node)
            self.assertTrue(np.allclose(
                grads[2], db_numerical, atol=EPSILON, rtol=EPSILON))

    def test_conv2d_op_backward_no_bias(self):
        op = Conv2d(stride=1, padding=0)
        x_val = self.default_rng.random((1, 1, 3, 3)).astype(np.float64)
        w_val = self.default_rng.random((1, 1, 2, 2)).astype(np.float64)
        self._check_conv_op_backward(op, x_val, w_val, b_val=None)

    def test_conv2d_op_backward_with_bias(self):
        op = Conv2d(stride=1, padding=0)
        x_val = self.default_rng.random((1, 1, 3, 3)).astype(np.float64)
        w_val = self.default_rng.random((1, 1, 2, 2)).astype(np.float64)
        b_val = self.default_rng.random((1)).astype(np.float64)
        self._check_conv_op_backward(op, x_val, w_val, b_val)

    def test_conv2d_op_backward_padding_stride_bias_multichannel(self) -> None:
        N = 2  # batch size
        C_in = 3
        H = W = 5
        C_out = 4
        kh = kw = 3
        stride = 2
        padding = 1

        op = Conv2d(stride=stride, padding=padding)
        x_val = self.default_rng.random((N, C_in, H, W)).astype(np.float64)
        w_val = self.default_rng.random(
            (C_out, C_in, kh, kw)).astype(np.float64)
        b_val = self.default_rng.random((C_out,)).astype(np.float64)
        self._check_conv_op_backward(op, x_val, w_val, b_val)

    def test_conv2d_layer_no_grad(self):
        x_val = self.default_rng.random((2, 3, 5, 5)).astype(np.float64)
        x_node = Node(x_val)

        conv_layer = Conv2dLayer(in_channels=3, out_channels=4, kernel_size=3)

        with no_grad():
            output_node = conv_layer(x_node)

        self.assertEqual(len(output_node.parents), 0)
        self.assertIsNone(output_node.op)

    def test_conv2d_forward_asymmetric_stride(self):
        x_val = np.arange(16).reshape(1, 1, 4, 4).astype(np.float32)
        w_val = np.ones((1, 1, 2, 2), dtype=np.float32)
        op = Conv2d(stride=(1, 2), padding=0)
        output = op.forward(x_val, w_val)
        
        # Expected:
        # Padded X (no padding):
        #  0  1  2  3
        #  4  5  6  7
        #  8  9 10 11
        # 12 13 14 15
        # ---
        # 0,0: [[0,1],[4,5]] -> sum 10
        # 0,1: [[2,3],[6,7]] -> sum 18
        # 1,0: [[4,5],[8,9]] -> sum 26
        # 1,1: [[6,7],[10,11]] -> sum 34
        # 2,0: [[8,9],[12,13]] -> sum 42
        # 2,1: [[10,11],[14,15]] -> sum 50
        
        expected_out = np.array([[[[10, 18], [26, 34], [42, 50]]]], dtype=np.float32)
        self.assertTrue(np.allclose(output, expected_out, atol=EPSILON))

    def test_conv2d_forward_asymmetric_padding(self):
        x_val = np.ones((1, 1, 3, 3), dtype=np.float32)
        w_val = np.ones((1, 1, 2, 2), dtype=np.float32)
        # padding: (1, 0)
        # Padded X: (H=3+2=5, W=3+0=3)
        # 0 0 0
        # 1 1 1
        # 1 1 1
        # 1 1 1
        # 0 0 0
        op = Conv2d(stride=1, padding=(1, 0))
        output = op.forward(x_val, w_val)
        self.assertEqual(output.shape, (1, 1, 4, 2))
        
        # 0,0: [[0,0],[1,1]] -> 2
        # 0,1: [[0,0],[1,1]] -> 2
        # 1,0: [[1,1],[1,1]] -> 4
        # 1,1: [[1,1],[1,1]] -> 4
        # 2,0: [[1,1],[1,1]] -> 4
        # 2,1: [[1,1],[1,1]] -> 4
        # 3,0: [[1,1],[0,0]] -> 2
        # 3,1: [[1,1],[0,0]] -> 2
        expected_out = np.array([[[[2, 2], [4, 4], [4, 4], [2, 2]]]], dtype=np.float32)
        self.assertTrue(np.allclose(output, expected_out, atol=EPSILON))

    def test_conv2d_backward_asymmetric_stride_padding(self):
        op = Conv2d(stride=(2, 1), padding=(1, 2))
        x_val = self.default_rng.random((1, 1, 4, 4)).astype(np.float64)
        w_val = self.default_rng.random((1, 1, 3, 3)).astype(np.float64)
        b_val = self.default_rng.random((1,)).astype(np.float64)
        self._check_conv_op_backward(op, x_val, w_val, b_val)

    def test_conv2d_stride_larger_than_kernel(self):
        # Stride 3, Kernel 2. Input 5x5
        # oh = (5-2)//3 + 1 = 1 + 1 = 2
        # ow = (5-2)//3 + 1 = 2
        op = Conv2d(stride=3, padding=0)
        x_val = np.arange(25).reshape(1, 1, 5, 5).astype(np.float32)
        w_val = np.ones((1, 1, 2, 2), dtype=np.float32)
        output = op.forward(x_val, w_val)
        
        # Expected:
        # 0,0: [[0,1],[5,6]] -> sum 12
        # 0,1: [[3,4],[8,9]] -> sum 24
        # 1,0: [[15,16],[20,21]] -> sum 72
        # 1,1: [[18,19],[23,24]] -> sum 84
        expected_out = np.array([[[[12, 24], [72, 84]]]], dtype=np.float32)
        self.assertTrue(np.allclose(output, expected_out, atol=EPSILON))
        # backward
        self._check_conv_op_backward(op, x_val.astype(np.float64), w_val.astype(np.float64))

    def test_conv2d_kernel_larger_than_input_with_padding(self):
        # Input 2x2, Kernel 3x3, Padding 1.
        # Padded Input: 4x4 (0-padded)
        # 0 0 0 0
        # 0 1 1 0
        # 0 1 1 0
        # 0 0 0 0
        # oh = (2+2-3)//1 + 1 = 2
        op = Conv2d(stride=1, padding=1)
        x_val = np.ones((1, 1, 2, 2), dtype=np.float32)
        w_val = np.ones((1, 1, 3, 3), dtype=np.float32)
        output = op.forward(x_val, w_val)
        self.assertEqual(output.shape, (1, 1, 2, 2))
        
        # Windows of padded input:
        # (0,0): [[0,0,0],[0,1,1],[0,1,1]] -> 4
        # (0,1): [[0,0,0],[1,1,0],[1,1,0]] -> 4
        # (1,0): [[0,1,1],[0,1,1],[0,0,0]] -> 4
        # (1,1): [[1,1,0],[1,1,0],[0,0,0]] -> 4
        expected_out = np.array([[[[4, 4], [4, 4]]]], dtype=np.float32)
        self.assertTrue(np.allclose(output, expected_out, atol=EPSILON))
        self._check_conv_op_backward(op, x_val.astype(np.float64), w_val.astype(np.float64))

    def test_conv2d_non_square_all(self):
        # N=1, Cin=2, H=3, W=5
        # Cout=3, kh=2, kw=4
        # stride=(1, 2), padding=(1, 0)
        # oh = (3 + 2*1 - 2)//1 + 1 = 4
        # ow = (5 + 2*0 - 4)//2 + 1 = 1//2 + 1 = 1
        op = Conv2d(stride=(1, 2), padding=(1, 0))
        x_val = self.default_rng.random((1, 2, 3, 5)).astype(np.float64)
        w_val = self.default_rng.random((3, 2, 2, 4)).astype(np.float64)
        b_val = self.default_rng.random((3,)).astype(np.float64)
        output = op.forward(x_val, w_val, b_val)
        self.assertEqual(output.shape, (1, 3, 4, 1))
        self._check_conv_op_backward(op, x_val, w_val, b_val)

    def test_conv2d_large_channels_and_batch(self):
        N = 4
        C_in = 8
        C_out = 16
        H = W = 8
        kh = kw = 3
        op = Conv2d(stride=2, padding=1)
        x_val = self.default_rng.random((N, C_in, H, W)).astype(np.float64)
        w_val = self.default_rng.random((C_out, C_in, kh, kw)).astype(np.float64)
        b_val = self.default_rng.random((C_out,)).astype(np.float64)
        self._check_conv_op_backward(op, x_val, w_val, b_val)
