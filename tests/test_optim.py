import unittest
import numpy as np
from cerebra import Parameter
from cerebra.optim import SGD

EPSILON = 1e-6


class TestOptim(unittest.TestCase):
    def test_sgd_step(self) -> None:
        # Initialise params
        p1_val = np.array([1.0, 2.0], dtype=np.float64)
        p2_val = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float64)
        p1 = Parameter(p1_val, name="p1")
        p2 = Parameter(p2_val, name="p2")

        params = [p1, p2]
        lr = 0.1
        optimiser = SGD(params, lr=lr)

        # Set gradients
        p1.grad = np.array([0.1, -0.2], dtype=np.float64)
        p2.grad = np.array([[0.01, 0.02], [-0.01, -0.02]], dtype=np.float64)

        # step
        optimiser.step()

        # Verify changed params
        expected_p1 = p1_val - lr * p1.grad
        expected_p2 = p2_val - lr * p2.grad

        self.assertTrue(np.allclose(p1.value, expected_p1, atol=EPSILON))
        self.assertTrue(np.allclose(p2.value, expected_p2, atol=EPSILON))

    def test_sgd_zero_grad(self) -> None:
        # Initialise params:
        p1_val = np.array([1.0, 2.0], dtype=np.float64)
        p1 = Parameter(p1_val, name="p1")
        p1.grad = np.array([0.5, 0.5], dtype=np.float64)

        optimiser = SGD([p1], lr=0.01)

        # Zero gradients
        optimiser.zero_grad()

        # Verify gradients are zero
        expected_grad = np.zeros_like(p1_val)
        self.assertTrue(np.allclose(p1.grad, expected_grad, atol=EPSILON))

    def test_sgd_zero_lr(self) -> None:
        # Ensure no parameter changes when lr=0
        p_val = np.array([1.0, 1.0], dtype=np.float64)
        p = Parameter(p_val, name="p")
        p.grad = np.array([10.0, 10.0], dtype=np.float64)

        optimiser = SGD([p], lr=0.0)
        optimiser.step()

        self.assertTrue(np.allclose(p.value, p_val, atol=EPSILON))

    def test_sgd_zero_grad_step(self) -> None:
        # Ensure no parameter changes when all gradients are zero
        p_val = np.array([1.0, 1.0], dtype=np.float64)
        p = Parameter(p_val, name="p")
        p.grad = np.array([0.0, 0.0], dtype=np.float64)

        optimiser = SGD([p], lr=0.1)
        optimiser.step()

        self.assertTrue(np.allclose(p.value, p_val, atol=EPSILON))

    def test_sgd_multiple_steps(self) -> None:
        # Ensure updates are applied correctly across multiple step() calls
        p_val = np.array([1.0], dtype=np.float64)
        p = Parameter(p_val, name="p")
        p.grad = np.array([1.0], dtype=np.float64)

        lr = 0.1
        optimiser = SGD([p], lr=lr)

        # 3 steps with the same gradient
        for _ in range(3):
            optimiser.step()

        expected_val = p_val - 3 * lr * p.grad
        self.assertTrue(np.allclose(p.value, expected_val, atol=EPSILON))
