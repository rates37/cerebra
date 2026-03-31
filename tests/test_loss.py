import unittest
import numpy as np
from cerebra import Node, cross_entropy_loss
from tests.utils import EPSILON


class TestCELoss(unittest.TestCase):
    def test_cross_entropy_value(self) -> None:
        # y = -mean(log(softmax(x)[target]))
        x = np.array([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4]])
        target = np.array([1, 2])

        loss_node = cross_entropy_loss(x, target)

        # Manual calculation
        def softmax(vals):
            exps = np.exp(vals - np.max(vals, axis=1, keepdims=True))
            return exps / np.sum(exps, axis=1, keepdims=True)

        sm = softmax(x)
        expected_loss = -np.mean(np.log(np.array([sm[0, 1], sm[1, 2]])))

        self.assertAlmostEqual(loss_node.value[0], expected_loss, places=6)

    def test_cross_entropy_stability(self) -> None:
        # Test with large values
        x = np.array([[100.0, -100.0]])
        target = np.array([0])
        # softmax should be [1, 0] roughly
        loss_node = cross_entropy_loss(x, target)
        self.assertFalse(np.isnan(loss_node.value[0]))
        self.assertAlmostEqual(loss_node.value[0], 0.0, places=4)

    def test_cross_entropy_backward(self) -> None:
        x_val = np.array([[0.5, -0.5], [1.0, 0.0]])
        target = np.array([0, 1])
        x = Node(x_val)

        loss = cross_entropy_loss(x, target)
        loss.backward()

        # dL/dx = (softmax(x) - one_hot(target)) / batch_size
        def softmax(vals):
            exps = np.exp(vals - np.max(vals, axis=1, keepdims=True))
            return exps / np.sum(exps, axis=1, keepdims=True)

        sm = softmax(x_val)
        one_hot = np.array([[1, 0], [0, 1]])
        expected_grad = (sm - one_hot) / 2.0

        self.assertTrue(np.allclose(x.grad, expected_grad, atol=EPSILON))
