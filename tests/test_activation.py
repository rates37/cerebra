import unittest
import numpy as np
from typing import Callable
from cerebra import Node, relu, sigmoid, tanh


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
        self.assertTrue(np.allclose(y.value, expected, atol=EPSILON))

        # test relu on Node:
        y = relu(x)
        self.assertTrue(np.allclose(y.value, expected, atol=EPSILON))

        # test relu on int:
        for x_val in x_vals:
            x_val = int(x_val)
            y = relu(x_val)
            expected = np.maximum(x_val, 0)
            self.assertTrue(np.allclose([y.value], [expected], atol=EPSILON))

        # test relu on floats:
        for x_val in x_vals:
            y = relu((x_val))
            expected = np.maximum(x_val, 0)
            self.assertTrue(np.allclose([y.value], [expected], atol=EPSILON))

    def test_relu_backward(self) -> None:
        x_vals = np.array([-3.0, -1.0, 0.0, 0.5, 2.0, 10.0], dtype=np.float64)
        x = Node(x_vals)

        y = relu(x)
        y.backward(np.ones_like(y.value))
        expected = (x_vals > 0).astype(np.float64)
        self.assertTrue(np.allclose(expected, x.grad, atol=EPSILON))

    def test_sigmoid_forward(self) -> None:
        x_vals = np.array([-3.0, -1.0, 0.0, 0.5, 2.0, 10.0], dtype=np.float64)
        x = Node(x_vals)
        expected = 1.0 / (1.0 + np.exp(-x_vals))

        # test sigmoid on ndarray:
        y = sigmoid(x_vals)
        self.assertTrue(np.allclose(y.value, expected, atol=EPSILON))

        # test sigmoid on Node:
        y = sigmoid(x)
        self.assertTrue(np.allclose(y.value, expected, atol=EPSILON))

        # test sigmoid on int:
        for x_val in x_vals:
            x_val = int(x_val)
            y = sigmoid(x_val)
            expected = 1.0 / (1.0 + np.exp(-x_val))
            self.assertTrue(np.allclose([y.value], [expected], atol=EPSILON))

        # test sigmoid on floats:
        for x_val in x_vals:
            y = sigmoid((x_val))
            expected = 1.0 / (1.0 + np.exp(-x_val))
            self.assertTrue(np.allclose([y.value], [expected], atol=EPSILON))

    def test_sigmoid_backward(self) -> None:
        x_vals = np.array([-3.0, -1.0, 0.0, 0.5, 2.0, 10.0], dtype=np.float64)
        x = Node(x_vals)

        y = sigmoid(x)
        y.backward(np.ones_like(y.value))
        expected_forward = 1.0 / (1.0 + np.exp(-x_vals))
        expected = expected_forward * (1.0 - expected_forward)
        self.assertTrue(np.allclose(expected, x.grad, atol=EPSILON))

    def test_tanh_forward(self) -> None:
        x_vals = np.array([-3.0, -1.0, 0.0, 0.5, 2.0, 10.0], dtype=np.float64)
        x = Node(x_vals)
        expected = np.tanh(x_vals)

        # test tanh on ndarray:
        y = tanh(x_vals)
        self.assertTrue(np.allclose(y.value, expected, atol=EPSILON))

        # test tanh on Node:
        y = tanh(x)
        self.assertTrue(np.allclose(y.value, expected, atol=EPSILON))

        # test tanh on int:
        for x_val in x_vals:
            x_val = int(x_val)
            y = tanh(x_val)
            expected = np.tanh(x_val)
            self.assertTrue(np.allclose([y.value], [expected], atol=EPSILON))

        # test tanh on floats:
        for x_val in x_vals:
            y = tanh((x_val))
            expected = np.tanh(x_val)
            self.assertTrue(np.allclose([y.value], [expected], atol=EPSILON))

    def test_tanh_backward(self) -> None:
        x_vals = np.array([-3.0, -1.0, 0.0, 0.5, 2.0, 10.0], dtype=np.float64)
        x = Node(x_vals)

        y = tanh(x)
        y.backward(np.ones_like(y.value))
        expected_forward = np.tanh(x_vals)
        expected = 1 - expected_forward**2
        self.assertTrue(np.allclose(expected, x.grad, atol=EPSILON))
