import unittest
import numpy as np
from cerebra import Node, MaxPool2DOp, MaxPool2D

EPSILON = 1e-6


class TestPool2d(unittest.TestCase):
    def setUp(self) -> None:
        pass

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
