import unittest
import numpy as np
from cerebra import Node, Linear
from tests.utils import EPSILON


class TestLinear(unittest.TestCase):
    def test_initialisation(self) -> None:
        in_features, out_features = 10, 5
        layer = Linear(in_features, out_features)

        # Check shapes
        self.assertEqual(layer.weight.value.shape, (in_features, out_features))
        self.assertEqual(layer.bias.value.shape, (out_features,))

        # Check Xavier initialisation roughly
        # checks standard deviation of weights is reasonable
        std = np.std(layer.weight.value)
        expected_std = np.sqrt(2 / (in_features + out_features))
        self.assertAlmostEqual(std, expected_std, delta=0.1)

    def test_forward_with_bias(self) -> None:
        layer = Linear(2, 3)
        layer.weight.value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        layer.bias.value = np.array([0.1, 0.2, 0.3])

        x = Node(np.array([[1.0, 1.0]]))
        y = layer(x)  # x @ w + b

        expected = np.array(
            [
                [
                    1.0 * 1.0 + 1.0 * 4.0 + 0.1,
                    1.0 * 2.0 + 1.0 * 5.0 + 0.2,
                    1.0 * 3.0 + 1.0 * 6.0 + 0.3,
                ]
            ]
        )
        self.assertTrue(np.allclose(y.value, expected, atol=EPSILON))

    def test_forward_no_bias(self) -> None:
        layer = Linear(2, 3, bias=False)
        layer.weight.value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertIsNone(layer.bias)

        x = Node(np.array([[1.0, 1.0]]))
        y = layer(x)

        expected = np.array([[5.0, 7.0, 9.0]])
        self.assertTrue(np.allclose(y.value, expected, atol=EPSILON))

    def test_backward(self) -> None:
        layer = Linear(2, 2)
        layer.weight.value = np.array([[1.0, 2.0], [3.0, 4.0]])
        layer.bias.value = np.array([0.1, 0.2])

        x = Node(np.array([[1.0, 2.0]]))
        y = layer(x)
        y.backward(np.array([[1.0, 1.0]]))  # dL/dy = [1, 1]

        # dL/db = dL/dy = [1, 1]
        self.assertTrue(
            np.allclose(layer.bias.grad, np.array([1.0, 1.0]), atol=EPSILON)
        )

        # dL/dw = x.T @ (dL/dy) = [[1], [2]] @ [[1, 1]] = [[1, 1], [2, 2]]
        expected_w_grad = np.array([[1.0, 1.0], [2.0, 2.0]])
        self.assertTrue(np.allclose(layer.weight.grad, expected_w_grad, atol=EPSILON))

        # dL/dx = (dL/dy) @ w.T = [[1, 1]] @ [[1, 3], [2, 4]] = [[3, 7]]
        expected_x_grad = np.array([[3.0, 7.0]])
        self.assertTrue(np.allclose(x.grad, expected_x_grad, atol=EPSILON))
