import unittest
import numpy as np
from typing import Callable, Optional
from cerebra import Node, MaxPool2DOp, MaxPool2D, AvgPool2DOp, AvgPool2D, no_grad

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


class TestMaxPool2d(unittest.TestCase):
    def setUp(self) -> None:
        self.default_rng = np.random.default_rng(69)

    def tearDown(self) -> None:
        pass

    def test_maxpool2d_op_forward(self):
        x_val = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
        pool = MaxPool2DOp(kernel_size=(2, 2), stride=1)
        out = pool.forward(x_val)
        expected_out = np.array(
            [[[[6., 7., 8.],
               [10., 11., 12.],
               [14., 15., 16.]
               ]]], dtype=np.float32)

        self.assertTrue(np.allclose(out, expected_out, atol=EPSILON))

    def test_maxpool2d_forward(self):
        x_val = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
        x = Node(x_val)
        pool = MaxPool2D(kernel_size=2, stride=1)
        out = pool(x)
        expected_out = np.array(
            [[[[6., 7., 8.],
               [10., 11., 12.],
               [14., 15., 16.]
               ]]], dtype=np.float32)

        self.assertTrue(np.allclose(out.value, expected_out, atol=EPSILON))

    def test_maxpool2d_op_forward_stride(self):
        x_val = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
        pool = MaxPool2DOp(kernel_size=(2, 2), stride=2)
        out = pool.forward(x_val)
        expected_out = np.array([[[[6., 8.], [14., 16.]]]], dtype=np.float32)

        self.assertTrue(np.allclose(out, expected_out, atol=EPSILON))

    def test_maxpool2d_forward_stride(self):
        x_val = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
        x = Node(x_val)
        pool = MaxPool2D(kernel_size=2, stride=2)
        out = pool(x)
        expected_out = np.array([[[[6., 8.], [14., 16.]]]], dtype=np.float32)

        self.assertTrue(np.allclose(out.value, expected_out, atol=EPSILON))

    def test_maxpool2d_op_forward_padding(self):
        x_val = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
        pool = MaxPool2DOp(kernel_size=(2, 2), stride=1, padding=1)
        out = pool.forward(x_val)
        expected_out = np.array(
            [[[[1., 2., 3., 3.],
               [4., 5., 6., 6.],
               [7., 8., 9., 9.],
               [7., 8., 9., 9.]
               ]]], dtype=np.float32)

        self.assertTrue(np.allclose(out, expected_out, atol=EPSILON))

    def test_maxpool2d_forward_padding(self):
        x_val = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
        x = Node(x_val)
        pool = MaxPool2D(kernel_size=2, stride=1, padding=1)
        out = pool(x)
        expected_out = np.array(
            [[[[1., 2., 3., 3.],
               [4., 5., 6., 6.],
               [7., 8., 9., 9.],
               [7., 8., 9., 9.]
               ]]], dtype=np.float32)

        self.assertTrue(np.allclose(out.value, expected_out, atol=EPSILON))

    def test_maxpool2d_op_forward_stride_and_padding(self):
        x_val = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
        pool = MaxPool2DOp(kernel_size=(2, 2), stride=2, padding=1)
        out = pool.forward(x_val)
        expected_out = np.array(
            [[[[1., 3.],
               [7., 9.],
               ]]], dtype=np.float32)

        self.assertTrue(np.allclose(out, expected_out, atol=EPSILON))

    def test_maxpool2d_forward_stride_and_padding(self):
        x_val = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
        x = Node(x_val)
        pool = MaxPool2D(kernel_size=2, stride=2, padding=1)
        out = pool(x)
        expected_out = np.array(
            [[[[1., 3.],
               [7., 9.],
               ]]], dtype=np.float32)

        self.assertTrue(np.allclose(out.value, expected_out, atol=EPSILON))

    #! ==========================
    #!    Backward Pool Tests
    #! ==========================
    def _check_maxpool_op_backward(
        self,
        op: MaxPool2DOp,
        x_val: np.ndarray,
        grad_val: Optional[np.ndarray] = None
    ) -> None:
        x_node = Node(x_val.copy())

        # run forward pass
        output_from_forward = op.forward(x_node.value)

        # generate random loss if necessary
        if grad_val is None:
            grad_val = self.default_rng.random(
                output_from_forward.shape).astype(np.float64)

        # dummy output node for backward call
        dummy_output = Node(output_from_forward, parents=[x_node], op=op)
        grads = op.backward(grad_val, dummy_output)
        dx_analytical = grads[0]

        # define function for numerical gradient calculation
        def get_loss_as_node() -> Node:
            out_val = op.forward(x_node.value)
            loss_val = (out_val * grad_val).sum()
            return Node(np.array([loss_val]))

        dx_numerical = numerical_gradient(get_loss_as_node, x_node)
        self.assertTrue(np.allclose(
            dx_analytical, dx_numerical, atol=EPSILON, rtol=EPSILON))

    def test_maxpool2d_op_backward(self):
        op = MaxPool2DOp(kernel_size=(2, 2), stride=1, padding=0)
        x_val = self.default_rng.random((1, 1, 4, 4)).astype(np.float64)
        self._check_maxpool_op_backward(op, x_val)

    def test_maxpool2d_op_backward_stride_padding_multichannel(self):
        N = 2
        C = 3
        H = W = 5
        kh = kw = 3
        stride = 2
        padding = 1

        op = MaxPool2DOp(kernel_size=(kh, kw), stride=stride, padding=padding)
        x_val = self.default_rng.random((N, C, H, W)).astype(np.float64)
        self._check_maxpool_op_backward(op, x_val)

    def test_maxpool2d_no_grad(self):
        x_val = self.default_rng.random((2, 3, 5, 5)).astype(np.float64)
        x_node = Node(x_val)

        pool_layer = MaxPool2D(kernel_size=2, stride=2)

        with no_grad():
            output_node = pool_layer(x_node)

        self.assertEqual(len(output_node.parents), 0)
        self.assertIsNone(output_node.op)


class TestAvgPool2d(unittest.TestCase):
    def setUp(self) -> None:
        self.default_rng = np.random.default_rng(69)

    def tearDown(self) -> None:
        pass

    def test_avgpool2d_op_forward(self):
        x_val = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
        pool = AvgPool2DOp(kernel_size=(2, 2), stride=1)
        out = pool.forward(x_val)
        expected_out = np.array(
            [[[[3.5, 4.5, 5.5],
               [7.5, 8.5, 9.5],
               [11.5, 12.5, 13.5]
               ]]], dtype=np.float32)

        self.assertTrue(np.allclose(out, expected_out, atol=EPSILON))

    def test_avgpool2d_forward(self):
        x_val = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
        x = Node(x_val)
        pool = AvgPool2D(kernel_size=2, stride=1)
        out = pool(x)
        expected_out = np.array(
            [[[[3.5, 4.5, 5.5],
               [7.5, 8.5, 9.5],
               [11.5, 12.5, 13.5]
               ]]], dtype=np.float32)

        self.assertTrue(np.allclose(out.value, expected_out, atol=EPSILON))

    def test_avgpool2d_op_forward_stride(self):
        x_val = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
        pool = AvgPool2DOp(kernel_size=(2, 2), stride=2)
        out = pool.forward(x_val)
        expected_out = np.array(
            [[[[3.5, 5.5], [11.5, 13.5]]]], dtype=np.float32)

        self.assertTrue(np.allclose(out, expected_out, atol=EPSILON))

    def test_avgpool2d_forward_stride(self):
        x_val = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
        x = Node(x_val)
        pool = AvgPool2D(kernel_size=2, stride=2)
        out = pool(x)
        expected_out = np.array(
            [[[[3.5, 5.5], [11.5, 13.5]]]], dtype=np.float32)

        self.assertTrue(np.allclose(out.value, expected_out, atol=EPSILON))

    def test_avgpool2d_op_forward_padding(self):
        x_val = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
        pool = AvgPool2DOp(kernel_size=(2, 2), stride=1, padding=1)
        out = pool.forward(x_val)
        expected_out = np.array(
            [[[[0.25, 0.75, 1.25, 0.75],
               [1.25, 3.0, 4.0, 2.25],
               [2.75, 6.0, 7.0, 3.75],
               [1.75, 3.75, 4.25, 2.25]
               ]]], dtype=np.float32)

        self.assertTrue(np.allclose(out, expected_out, atol=EPSILON))

    def test_avgpool2d_forward_padding(self):
        x_val = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
        x = Node(x_val)
        pool = AvgPool2D(kernel_size=2, stride=1, padding=1)
        out = pool(x)
        expected_out = np.array(
            [[[[0.25, 0.75, 1.25, 0.75],
               [1.25, 3.0, 4.0, 2.25],
               [2.75, 6.0, 7.0, 3.75],
               [1.75, 3.75, 4.25, 2.25]
               ]]], dtype=np.float32)

        self.assertTrue(np.allclose(out.value, expected_out, atol=EPSILON))

    def test_avgpool2d_op_forward_stride_and_padding(self):
        x_val = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
        pool = AvgPool2DOp(kernel_size=(2, 2), stride=2, padding=1)
        out = pool.forward(x_val)
        expected_out = np.array(
            [[[[0.25, 1.25],
               [2.75, 7.0],
               ]]], dtype=np.float32)

        self.assertTrue(np.allclose(out, expected_out, atol=EPSILON))

    def test_avgpool2d_forward_stride_and_padding(self):
        x_val = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
        x = Node(x_val)
        pool = AvgPool2D(kernel_size=2, stride=2, padding=1)
        out = pool(x)
        expected_out = np.array(
            [[[[0.25, 1.25],
               [2.75, 7.0],
               ]]], dtype=np.float32)

        self.assertTrue(np.allclose(out.value, expected_out, atol=EPSILON))

    #! ==========================
    #!    Backward Pool Tests
    #! ==========================
    def _check_avgpool_op_backward(
        self,
        op: AvgPool2DOp,
        x_val: np.ndarray,
        grad_val: Optional[np.ndarray] = None
    ) -> None:
        x_node = Node(x_val.copy())

        # run forward pass
        output_from_forward = op.forward(x_node.value)

        # generate random loss if necessary
        if grad_val is None:
            grad_val = self.default_rng.random(
                output_from_forward.shape).astype(np.float64)

        # dummy output node for backward call
        dummy_output = Node(output_from_forward, parents=[x_node], op=op)
        grads = op.backward(grad_val, dummy_output)
        dx_analytical = grads[0]

        # define function for numerical gradient calculation
        def get_loss_as_node() -> Node:
            out_val = op.forward(x_node.value)
            loss_val = (out_val * grad_val).sum()
            return Node(np.array([loss_val]))

        dx_numerical = numerical_gradient(get_loss_as_node, x_node)
        self.assertTrue(np.allclose(
            dx_analytical, dx_numerical, atol=EPSILON, rtol=EPSILON))

    def test_avgpool2d_op_backward(self):
        op = AvgPool2DOp(kernel_size=(2, 2), stride=1, padding=0)
        x_val = self.default_rng.random((1, 1, 4, 4)).astype(np.float64)
        self._check_avgpool_op_backward(op, x_val)

    def test_avgpool2d_op_backward_stride_padding_multichannel(self):
        N = 2
        C = 3
        H = W = 5
        kh = kw = 3
        stride = 2
        padding = 1

        op = AvgPool2DOp(kernel_size=(kh, kw), stride=stride, padding=padding)
        x_val = self.default_rng.random((N, C, H, W)).astype(np.float64)
        self._check_avgpool_op_backward(op, x_val)

    def test_avgpool2d_no_grad(self):
        x_val = self.default_rng.random((2, 3, 5, 5)).astype(np.float64)
        x_node = Node(x_val)

        pool_layer = AvgPool2D(kernel_size=2, stride=2)

        with no_grad():
            output_node = pool_layer(x_node)

        self.assertEqual(len(output_node.parents), 0)
        self.assertIsNone(output_node.op)
