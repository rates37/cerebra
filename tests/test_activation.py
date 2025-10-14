import unittest
import numpy as np
from typing import Callable
from cerebra import Node, relu


EPSILON = 1e-6


class TestActivationFunction(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_relu_forward(self) -> None:
        x_vals = np.array([-3.0, -1.0, 0.0, 0.5, 2.0, 10.0], dtype=np.float64)
        x = Node(x_vals)
        expected = np.maximum(x_vals, 0.0)

        # test relu on ndarray:
        y = relu(x_vals)
        self.assertTrue(np.allclose(y.value, expected))

        # test relu on Node:
        y = relu(x)
        self.assertTrue(np.allclose(y.value, expected))

        # test relu on int:
        for x_val in x_vals:
            x_val = int(x_val)
            y = relu(x_val)
            expected = np.maximum(x_val, 0)
            self.assertTrue(np.allclose([y.value], [expected]))

        # test relu on floats:
        for x_val in x_vals:
            y = relu((x_val))
            expected = np.maximum(x_val, 0)
            self.assertTrue(np.allclose([y.value], [expected]))

    def test_relu_backward(self) -> None:
        x_vals = np.array([-3.0, -1.0, 0.0, 0.5, 2.0, 10.0], dtype=np.float64)
        x = Node(x_vals)

        y = relu(x)
        y.backward(np.ones_like(y.value))
        expected = (x_vals > 0).astype(np.float64)
        self.assertTrue(np.allclose(expected, x.grad, atol=EPSILON))
