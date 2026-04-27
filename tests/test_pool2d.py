import unittest
import numpy as np
from cerebra import Node, MaxPool2DOp, MaxPool2D, AvgPool2DOp, AvgPool2D, no_grad
from tests.utils import EPSILON, numerical_gradient
from typing import Optional

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

    def test_maxpool2d_op_forward_asymmetric_padding(self):
        x_val = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
        # padding (1, 0)
        # Padded X:
        # 0 0 0
        # 1 2 3
        # 4 5 6
        # 7 8 9
        # 0 0 0
        pool = MaxPool2DOp(kernel_size=(2, 2), stride=1, padding=(1, 0))
        out = pool.forward(x_val)
        self.assertEqual(out.shape, (1, 1, 4, 2))
        expected_out = np.array([[[[2, 3],
                                   [5, 6],
                                   [8, 9],
                                   [8, 9]]]], dtype=np.float32)
        self.assertTrue(np.allclose(out, expected_out, atol=EPSILON))

    def test_maxpool2d_op_forward_asymmetric_stride(self):
        x_val = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
        # stride (1, 2)
        # oh = (4-2)//1 + 1 = 3, ow = (4-2)//2 + 1 = 2
        pool = MaxPool2DOp(kernel_size=(2, 2), stride=(1, 2))
        out = pool.forward(x_val)
        self.assertEqual(out.shape, (1, 1, 3, 2))
        
        # Expected output:
        # kernel at (0,0): [[1,2],[5,6]] -> 6
        # kernel at (0,1): [[3,4],[7,8]] -> 8
        # kernel at (1,0): [[5,6],[9,10]] -> 10
        # kernel at (1,1): [[7,8],[11,12]] -> 12
        # kernel at (2,0): [[9,10],[13,14]] -> 14
        # kernel at (2,1): [[11,12],[15,16]] -> 16
        expected_out = np.array([[[[6, 8], [10, 12], [14, 16]]]], dtype=np.float32)
        self.assertTrue(np.allclose(out, expected_out, atol=EPSILON))

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

    def test_maxpool2d_asymmetric_stride_padding(self):
        op = MaxPool2DOp(kernel_size=(2, 2), stride=(2, 1), padding=(1, 0))
        x_val = self.default_rng.random((1, 1, 4, 4)).astype(np.float64)
        output = op.forward(x_val)
        self.assertEqual(output.shape, (1, 1, 3, 3))
        self._check_maxpool_op_backward(op, x_val)

    def test_maxpool2d_kernel_larger_than_input(self):
        op = MaxPool2DOp(kernel_size=(4, 4), stride=1, padding=1)
        x_val = self.default_rng.random((1, 1, 3, 3)).astype(np.float64)
        output = op.forward(x_val)
        self.assertEqual(output.shape, (1, 1, 2, 2))
        self._check_maxpool_op_backward(op, x_val)

    def test_maxpool2d_stride_larger_than_kernel(self):
        op = MaxPool2DOp(kernel_size=(2, 2), stride=3, padding=0)
        x_val = self.default_rng.random((1, 1, 5, 5)).astype(np.float64)
        output = op.forward(x_val)
        self.assertEqual(output.shape, (1, 1, 2, 2))
        self._check_maxpool_op_backward(op, x_val)


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

    def test_avgpool2d_op_forward_asymmetric_padding(self):
        x_val = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
        pool = AvgPool2DOp(kernel_size=(2, 2), stride=1, padding=(1, 0))
        out = pool.forward(x_val)
        self.assertEqual(out.shape, (1, 1, 4, 2))
        expected_out = np.array([[[[0.75, 1.25],
                                   [3.0, 4.0],
                                   [6.0, 7.0],
                                   [3.75, 4.25]]]], dtype=np.float32)
        self.assertTrue(np.allclose(out, expected_out, atol=EPSILON))

    def test_avgpool2d_op_forward_asymmetric_stride(self):
        x_val = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
        # stride (1, 2)
        pool = AvgPool2DOp(kernel_size=(2, 2), stride=(1, 2))
        out = pool.forward(x_val)
        self.assertEqual(out.shape, (1, 1, 3, 2))
        
        # Expected means:
        # kernel at (0,0): [[1,2],[5,6]] -> 14/4 = 3.5
        # kernel at (0,1): [[3,4],[7,8]] -> 22/4 = 5.5
        # kernel at (1,0): [[5,6],[9,10]] -> 30/4 = 7.5
        # kernel at (1,1): [[7,8],[11,12]] -> 38/4 = 9.5
        # kernel at (2,0): [[9,10],[13,14]] -> 46/4 = 11.5
        # kernel at (2,1): [[11,12],[15,16]] -> 54/4 = 13.5
        expected_out = np.array([[[[3.5, 5.5], [7.5, 9.5], [11.5, 13.5]]]], dtype=np.float32)
        self.assertTrue(np.allclose(out, expected_out, atol=EPSILON))

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

    def test_avgpool2d_asymmetric_stride_padding(self):
        op = AvgPool2DOp(kernel_size=(2, 2), stride=(2, 1), padding=(1, 0))
        x_val = self.default_rng.random((1, 1, 4, 4)).astype(np.float64)
        output = op.forward(x_val)
        self.assertEqual(output.shape, (1, 1, 3, 3))
        self._check_avgpool_op_backward(op, x_val)

    def test_avgpool2d_kernel_larger_than_input(self):
        op = AvgPool2DOp(kernel_size=(4, 4), stride=1, padding=1)
        x_val = self.default_rng.random((1, 1, 3, 3)).astype(np.float64)
        output = op.forward(x_val)
        self.assertEqual(output.shape, (1, 1, 2, 2))
        self._check_avgpool_op_backward(op, x_val)

    def test_avgpool2d_stride_larger_than_kernel(self):
        op = AvgPool2DOp(kernel_size=(2, 2), stride=3, padding=0)
        x_val = self.default_rng.random((1, 1, 5, 5)).astype(np.float64)
        output = op.forward(x_val)
        self.assertEqual(output.shape, (1, 1, 2, 2))
        self._check_avgpool_op_backward(op, x_val)
