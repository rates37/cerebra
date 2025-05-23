import unittest
import numpy as np
from typing import Callable
from cerebra import Node, Conv2d, Parameter, Operation

EPSILON = 1e-6


def numerical_gradient(func: Callable[[None], Node], node: Node, h=1e-6) -> np.ndarray:
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
    grad = np.zeros_like(input_value, dtype=np.float32)

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
        self.assertTrue(np.allclose(grads[0], dx_numerical, atol=EPSILON, rtol=EPSILON))

        dw_numerical = numerical_gradient(get_loss_as_node, w_node)
        self.assertTrue(np.allclose(grads[1], dw_numerical, atol=EPSILON, rtol=EPSILON))

        if b_val is not None:
            db_numerical = numerical_gradient(get_loss_as_node, b_node)
            self.assertTrue(np.allclose(grads[2], db_numerical, atol=EPSILON, rtol=EPSILON))

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
